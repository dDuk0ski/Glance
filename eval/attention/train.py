#!/usr/bin/env python3
"""Train and evaluate the Glance attention classifier.

Reads the feature CSVs in data/ (made by features_mp.py or the Apple Vision tool),
compares four frame-level models, evaluates them without leaking people between
train and test, and exports the logistic regression for the app.

Models compared (README "Evaluation"):
    face_detected   baseline 1: face found = looking
    head_threshold  baseline 2: head yaw/pitch within a tuned window around the
                    typical "looking" pose (thresholds tuned on the training fold)
    logreg_head     logistic regression on head/face-box features only (ablation)
    logreg_full     logistic regression with pupil + eye features (the model we ship; includes squared terms)

Evaluation:
    lopo      leave-one-person-out (needs >= 2 people; this is the real result)
    loso      leave-one-session-out (when there is only one person)
    blocked   grouped K-fold over dot/prompt segments (single session; pipeline test only)
    transfer  train on MacBook sessions, test on other laptops (if both exist)
    The mode is chosen automatically; blocked/loso do NOT measure generalization to new
    people, and the script says so.

Outputs:
    results/frame_metrics_folds.csv      per-fold accuracy / F1 / ROC-AUC per model
    results/frame_metrics_summary.csv    mean +/- std over folds, and pooled
    results/oof_scores.csv               out-of-fold scores per frame (input for smoothing.py)
    results/logreg_weights.csv           standardized weights of the final model
    results/roc_oof.png, results/f1_per_fold.png
    results/transfer_metrics.csv         (only with both MacBook and other-laptop data)
    attention_model.json                 the exported model for the app

Usage:
    python train.py                          # extractor "mp", automatic CV mode
    python train.py --extractor vision       # train on Apple Vision CSVs
    python train.py --cv lopo --C 0.3
    python train.py --no-export
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent

RAW_FEATURES = [
    "face_present", "yaw", "pitch", "roll",
    "face_x", "face_y", "face_w", "face_h",
    "pupil_lx", "pupil_ly", "pupil_rx", "pupil_ry",
    "eye_open_l", "eye_open_r",
]
# Squared terms. "Looking at the panel" means being CLOSE to a typical pose, while "not looking"
# means far from it on either side (left or right, up or down). A linear model on the signed
# value cannot express that (the two sides cancel out), but value + value^2 can (a bump).
SQUARED = ["yaw", "pitch", "roll", "pupil_lx", "pupil_ly", "pupil_rx", "pupil_ry"]
DERIVED = {f"{c}_sq": c for c in SQUARED}  # name -> raw column; value = raw * raw
FEATURES_FULL = RAW_FEATURES + list(DERIVED)
FEATURES_HEAD = RAW_FEATURES[:8] + ["yaw_sq", "pitch_sq", "roll_sq"]  # no pupil / eye features
META_COLS = ["person", "session", "frame", "timestamp", "label", "laptop", "target"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(data_dir: Path, extractor: str) -> pd.DataFrame:
    files = sorted(data_dir.glob(f"{extractor}_*.csv"))
    if not files:
        sys.exit(f"No {extractor}_*.csv files in {data_dir}. Run features_mp.py first.")
    dfs = []
    for f in files:
        d = pd.read_csv(f, dtype={"person": str, "session": str})
        missing = [c for c in META_COLS[:5] + RAW_FEATURES if c not in d.columns]
        if missing:
            sys.exit(f"{f.name} is missing columns: {missing}")
        d["source_file"] = f.name
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    if "laptop" not in df.columns:
        df["laptop"] = "unknown"
    if "target" not in df.columns:
        df["target"] = ""
    df["session_id"] = df["person"] + "_" + df["session"]
    for c in RAW_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for new_name, raw in DERIVED.items():
        df[new_name] = df[raw] * df[raw]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(files)} file(s): {len(df)} frames, {df['person'].nunique()} person(s), "
          f"{df['session_id'].nunique()} session(s), {df['laptop'].nunique()} laptop(s)")
    print(f"  looking: {int(df['label'].sum())} ({df['label'].mean():.0%}), "
          f"not looking: {int((1 - df['label']).sum())}")
    print(f"  face found in {df['face_present'].mean():.1%} of frames")
    return df


# --------------------------------------------------------------------------- #
# Models. Each has fit(df) and score(df) -> value in [0, 1]; predict = score >= 0.5
# --------------------------------------------------------------------------- #

class FaceDetected:
    name = "face_detected"

    def fit(self, df):
        return self

    def score(self, df):
        return df["face_present"].fillna(0).to_numpy(float)


class HeadThreshold:
    """Looking if |yaw - yaw0| <= ty and |pitch - pitch0| <= tp.

    yaw0 / pitch0 are the median head pose over training frames labelled "looking";
    ty / tp are grid-searched on the training fold to maximize F1. The score
    1 / (1 + d), with d = max(|dyaw| / ty, |dpitch| / tp), is 0.5 exactly on the
    decision boundary, so it also gives a usable ROC curve.
    """
    name = "head_threshold"

    def fit(self, df):
        look = df[(df["label"] == 1) & df["yaw"].notna() & df["pitch"].notna()]
        self.yaw0 = float(look["yaw"].median()) if len(look) else 0.0
        self.pitch0 = float(look["pitch"].median()) if len(look) else 0.0
        valid = df["yaw"].notna() & df["pitch"].notna()
        dy = (df["yaw"] - self.yaw0).abs().to_numpy()
        dp = (df["pitch"] - self.pitch0).abs().to_numpy()
        y = df["label"].to_numpy()
        best = (-1.0, 0.2, 0.2)
        grid = np.linspace(0.03, 0.9, 30)
        for ty in grid:
            for tp in grid:
                pred = valid.to_numpy() & (dy <= ty) & (dp <= tp)
                f = f1_score(y, pred, zero_division=0)
                if f > best[0]:
                    best = (f, ty, tp)
        _, self.ty, self.tp = best
        return self

    def score(self, df):
        dy = (df["yaw"] - self.yaw0).abs() / self.ty
        dp = (df["pitch"] - self.pitch0).abs() / self.tp
        d = np.maximum(dy, dp).to_numpy(float)
        s = 1.0 / (1.0 + d)
        return np.where(np.isnan(d), 0.0, s)  # no pose (e.g. no face) -> not looking


class LogReg:
    """Standardize (mean-impute missing values) + class-balanced logistic regression."""

    def __init__(self, name: str, features: list[str], C: float):
        self.name, self.features, self.C = name, features, C

    def _z(self, df):
        x = df[self.features].to_numpy(float)
        z = (x - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0)  # missing -> training mean

    def fit(self, df):
        x = df[self.features].to_numpy(float)
        self.mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.std = np.where(np.isnan(std) | (std < 1e-9), 1.0, std)
        self.clf = LogisticRegression(C=self.C, class_weight="balanced", max_iter=2000)
        self.clf.fit(self._z(df), df["label"].to_numpy())
        return self

    def score(self, df):
        return self.clf.predict_proba(self._z(df))[:, 1]


def make_models(C: float):
    return [
        FaceDetected(),
        HeadThreshold(),
        LogReg("logreg_head", FEATURES_HEAD, C),
        LogReg("logreg_full", FEATURES_FULL, C),
    ]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    pred = (s >= 0.5).astype(int)
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),  # share of looking frames kept open
        "roc_auc": auc,
    }


def make_folds(df: pd.DataFrame, mode: str):
    """Return (mode_used, [(fold_name, train_idx, test_idx), ...])."""
    n_people = df["person"].nunique()
    if mode == "auto":
        mode = "lopo" if n_people >= 2 else ("loso" if df["session_id"].nunique() >= 2 else "blocked")

    idx = np.arange(len(df))
    folds = []
    if mode in ("lopo", "loso"):
        key = "person" if mode == "lopo" else "session_id"
        for g in df[key].unique():
            test = idx[(df[key] == g).to_numpy()]
            train = idx[(df[key] != g).to_numpy()]
            folds.append((str(g), train, test))
    elif mode == "blocked":
        groups = (df["session_id"] + "|" + df["target"].astype(str)).to_numpy()
        k = min(5, len(np.unique(groups)))
        for i, (tr, te) in enumerate(GroupKFold(n_splits=k).split(idx, groups=groups)):
            folds.append((f"block{i + 1}", tr, te))
    else:
        sys.exit(f"Unknown cv mode {mode}")

    if mode == "lopo" and n_people < 2:
        sys.exit("lopo needs at least 2 people")
    return mode, folds


def cross_validate(df: pd.DataFrame, folds, C: float):
    names = [m.name for m in make_models(C)]
    fold_rows, oof = [], {n: np.full(len(df), np.nan) for n in names}
    for fold_name, tr, te in folds:
        train, test = df.iloc[tr], df.iloc[te]
        if train["label"].nunique() < 2:
            print(f"  skipping fold {fold_name}: training data has only one class")
            continue
        for m in make_models(C):
            m.fit(train)
            s = m.score(test)
            oof[m.name][te] = s
            fold_rows.append({"fold": fold_name, "model": m.name, **metrics(test["label"].to_numpy(), s)})
    return pd.DataFrame(fold_rows), oof


def summarize(fold_df: pd.DataFrame, df: pd.DataFrame, oof: dict) -> pd.DataFrame:
    rows = []
    y_all = df["label"].to_numpy()
    for name, s in oof.items():
        ok = ~np.isnan(s)
        pooled = metrics(y_all[ok], s[ok])
        sub = fold_df[fold_df["model"] == name]
        row = {"model": name, "n_folds": len(sub)}
        for k in ("accuracy", "f1", "roc_auc", "recall", "precision"):
            row[f"{k}_mean"] = sub[k].mean()
            row[f"{k}_std"] = sub[k].std(ddof=0)
            row[f"{k}_pooled"] = pooled[k]
        rows.append(row)
    return pd.DataFrame(rows)


def transfer_test(df: pd.DataFrame, train_laptop: str, C: float):
    is_train = df["laptop"].str.lower().str.contains(train_laptop.lower(), na=False)
    if is_train.all() or not is_train.any():
        return None
    train, test = df[is_train], df[~is_train]
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        return None
    rows = []
    for m in make_models(C):
        m.fit(train)
        rows.append({"model": m.name, "train": f"laptop contains '{train_laptop}'",
                     "test": ", ".join(sorted(test["laptop"].unique())),
                     **metrics(test["label"].to_numpy(), m.score(test))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots and export
# --------------------------------------------------------------------------- #

def plot_roc(df, oof, path):
    y = df["label"].to_numpy()
    plt.figure(figsize=(5.5, 5))
    for name, s in oof.items():
        ok = ~np.isnan(s)
        if len(np.unique(y[ok])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y[ok], s[ok]):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.xlabel("False positive rate (not looking, kept open)")
    plt.ylabel("True positive rate (looking, kept open)")
    plt.title("Out-of-fold ROC")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_f1(fold_df, path):
    piv = fold_df.pivot(index="fold", columns="model", values="f1")
    ax = piv.plot(kind="bar", figsize=(max(6, 1.2 * len(piv)), 4))
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Frame-level F1 per held-out fold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def export_model(df, features, C, args, extractor, path: Path, weights_path: Path):
    m = LogReg("final", features, C).fit(df)
    out = {
        "features": features,
        "mean": [float(v) for v in m.mean],
        "std": [float(v) for v in m.std],
        "weights": [float(v) for v in m.clf.coef_[0]],
        "bias": float(m.clf.intercept_[0]),
        "open_threshold": args.open_threshold,
        "close_threshold": args.close_threshold,
        "window_seconds": args.window_seconds,
        "dismiss_after_seconds": args.dismiss_after,
        # Extra keys (the app can ignore them):
        "derived_features": {k: f"{v} * {v}  (computed from the raw value, before standardizing)"
                             for k, v in DERIVED.items() if k in features},
        "missing_value_policy": "replace missing feature with its mean (z = 0)",
        "extractor": extractor,
        "trained_on": {"frames": int(len(df)), "people": int(df["person"].nunique()),
                       "sessions": int(df["session_id"].nunique())},
    }
    path.write_text(json.dumps(out, indent=2))
    pd.DataFrame({"feature": features, "weight_standardized": m.clf.coef_[0]}) \
        .sort_values("weight_standardized", key=np.abs, ascending=False) \
        .to_csv(weights_path, index=False)
    print(f"\nExported {path} ({len(features)} features, trained on all {len(df)} frames)")
    print("Top weights (standardized):")
    top = pd.read_csv(weights_path).head(6)
    for _, r in top.iterrows():
        print(f"    {r['feature']:<12s} {r['weight_standardized']:+.3f}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="mp", help="CSV prefix in data/: mp or vision (default mp)")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--cv", default="auto", choices=["auto", "lopo", "loso", "blocked"])
    ap.add_argument("--C", type=float, default=1.0, help="logistic regression inverse regularization")
    ap.add_argument("--train-laptop", default="macbook",
                    help="substring of the laptop name used as the training set in the transfer test")
    ap.add_argument("--export-model", choices=["full", "head"], default="full")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "attention_model.json")
    ap.add_argument("--open-threshold", type=float, default=0.6)
    ap.add_argument("--close-threshold", type=float, default=0.4)
    ap.add_argument("--window-seconds", type=float, default=1.0)
    ap.add_argument("--dismiss-after", type=float, default=1.5)
    args = ap.parse_args()

    df = load_data(args.data_dir, args.extractor)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if df["label"].nunique() < 2:
        sys.exit("Need both looking and not-looking frames to train.")

    mode, folds = make_folds(df, args.cv)
    print(f"\nCross-validation: {mode}, {len(folds)} fold(s)")
    if mode != "lopo":
        print("  NOTE: this does not test generalization to new people. For the report you need "
              "data from at least 2 (ideally 5+) people so leave-one-person-out can run.")

    fold_df, oof = cross_validate(df, folds, args.C)
    summary = summarize(fold_df, df, oof)

    fold_df.to_csv(args.results_dir / "frame_metrics_folds.csv", index=False)
    summary.to_csv(args.results_dir / "frame_metrics_summary.csv", index=False)

    oof_df = df[["person", "session", "frame", "timestamp", "label", "laptop", "target"]].copy()
    for name, s in oof.items():
        oof_df[f"score_{name}"] = s
    oof_df.to_csv(args.results_dir / "oof_scores.csv", index=False)

    plot_roc(df, oof, args.results_dir / "roc_oof.png")
    plot_f1(fold_df, args.results_dir / "f1_per_fold.png")

    print("\nFrame-level results (mean over folds | pooled out-of-fold):")
    print(f"  {'model':<15s} {'accuracy':>16s} {'F1':>16s} {'ROC-AUC':>16s}")
    for _, r in summary.iterrows():
        def cell(k):
            return f"{r[k + '_mean']:.3f} | {r[k + '_pooled']:.3f}"
        print(f"  {r['model']:<15s} {cell('accuracy'):>16s} {cell('f1'):>16s} {cell('roc_auc'):>16s}")

    tr = transfer_test(df, args.train_laptop, args.C)
    if tr is not None:
        tr.to_csv(args.results_dir / "transfer_metrics.csv", index=False)
        print(f"\nTransfer test (train: '{args.train_laptop}' laptops, test: others):")
        print(tr[["model", "accuracy", "f1", "roc_auc", "n"]].round(3).to_string(index=False))
    else:
        print(f"\nTransfer test skipped (needs both '{args.train_laptop}' and other-laptop data).")

    if not args.no_export:
        feats = FEATURES_FULL if args.export_model == "full" else FEATURES_HEAD
        export_model(df, feats, args.C, args, args.extractor, args.out,
                     args.results_dir / "logreg_weights.csv")
        if args.extractor != "vision":
            print("\nNOTE: the app uses Apple Vision features. Retrain with --extractor vision for "
                  "the final model; this export is from MediaPipe features.")
    print(f"\nResults written to {args.results_dir}")


if __name__ == "__main__":
    main()