#!/usr/bin/env python3
"""Dismiss logic for Glance's reply panel, replayed on recorded sessions.

train.py gives a per-frame "looking at the panel" score. A reply panel can't flicker
on every noisy frame, so this module turns the score stream into keep-open / dismiss
decisions, and measures how well that works at the SYSTEM level (README "Evaluation"):

    false dismissals   the panel closes while the user is still looking at it
    linger time        how long the panel stays open after the user looks away
    latency            compute time per frame of the smoothing logic

The decision logic (AttentionTracker) is deliberately small so it can be ported to Swift
line for line:

    1. Keep the scores of the last `window_seconds`; smoothed = their mean.
    2. Hysteresis: smoothed >= open_threshold  -> user is attending
                   smoothed <= close_threshold -> user is looking away
                   in between                  -> keep the previous state
    3. If the user has been looking away for `dismiss_after_seconds` -> DISMISS.
    4. After a dismissal the tracker re-arms as soon as the user attends again
       (smoothed >= open_threshold), which is how a replay can score every look-away.

Replay input is results/oof_scores.csv from train.py (out-of-fold scores, so the
classifier never saw the frames it is being replayed on).

Replay assumptions (read before quoting numbers):
  * Only labelled frames exist in the data. The ~0.8 s after each target change was not
    labelled (collect.py --settle), so those frames are missing. If the time gap between
    two frames exceeds --gap-reset (default 0.4 s) the smoothing window is cleared, as if
    the gap had been filled with frames of the new target; the dismiss timer keeps running.
  * Linger is measured from the first labelled frame of a look-away segment, so a real
    person's linger is up to about 0.8 s longer (time to move the eyes).

Usage:
    python smoothing.py                         # replay with defaults, compare all models
    python smoothing.py --sweep                 # search window / thresholds / dismiss delay
    python smoothing.py --sweep --update-model  # ...and write the best into attention_model.json
    python smoothing.py --window 0.5 --open 0.7 --close 0.3 --dismiss-after 1.0
"""

import argparse
import itertools
import json
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULTS = {"window_seconds": 1.0, "open_threshold": 0.6, "close_threshold": 0.4,
            "dismiss_after_seconds": 1.5}
MAX_DT = 0.5  # cap on a single frame's duration when summing "time spent" (seconds)


# --------------------------------------------------------------------------- #
# The logic that ships (port this to Swift)
# --------------------------------------------------------------------------- #

class AttentionTracker:
    def __init__(self, window_seconds, open_threshold, close_threshold, dismiss_after_seconds,
                 gap_reset=0.0):
        self.window = window_seconds
        self.open_t = open_threshold
        self.close_t = close_threshold
        self.dismiss_after = dismiss_after_seconds
        self.gap_reset = gap_reset
        self.buf = deque()          # (time, score)
        self.total = 0.0
        self.last_t = None
        self.attending = True       # the reply starts open
        self.away_since = None
        self.dismissed = False

    def update(self, t: float, score: float):
        """Feed one frame. Returns (smoothed_score, dismiss_event)."""
        if self.gap_reset and self.last_t is not None and t - self.last_t > self.gap_reset:
            self.buf.clear()
            self.total = 0.0
        self.last_t = t

        self.buf.append((t, score))
        self.total += score
        while self.buf[0][0] < t - self.window:
            self.total -= self.buf.popleft()[1]
        smoothed = self.total / len(self.buf)

        if smoothed >= self.open_t:
            self.attending = True
            self.away_since = None
            self.dismissed = False  # re-arm
        elif smoothed <= self.close_t and self.attending:
            self.attending = False
            self.away_since = t

        event = False
        if not self.attending and not self.dismissed and t - self.away_since >= self.dismiss_after:
            self.dismissed = True
            event = True
        return smoothed, event


# --------------------------------------------------------------------------- #
# Replay and metrics
# --------------------------------------------------------------------------- #

def runs(values: np.ndarray):
    """Consecutive runs of equal values -> list of (start_idx, end_idx_inclusive, value)."""
    if len(values) == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(values) - 1]))
    return [(int(s), int(e), values[s]) for s, e in zip(starts, ends)]


class Session:
    """One recorded session with its precomputed segments (independent of parameters)."""

    def __init__(self, key, d: pd.DataFrame, score_col: str):
        d = d.sort_values("timestamp")
        self.key = key
        self.d = d.reset_index(drop=True)
        self.t = self.d["timestamp"].to_numpy(float)
        self.s = self.d[score_col].to_numpy(float)
        self.label = self.d["label"].to_numpy(int)
        dt = np.diff(self.t, append=self.t[-1])
        self.dt = np.clip(dt, 0, MAX_DT)

        tgt = self.d["target"].astype(str).to_numpy() if "target" in self.d else self.label.astype(str)
        self.target_segs = runs(tgt)                  # one per dot/prompt occurrence
        self.label_runs = runs(self.label)            # merged same-label stretches


def replay(sess: Session, params: dict, gap_reset: float, time_frames: bool = False):
    n = len(sess.t)
    smoothed = np.zeros(n)
    event = np.zeros(n, bool)
    dismissed = np.zeros(n, bool)
    attending = np.zeros(n, bool)
    lat = np.zeros(n) if time_frames else None
    tr = AttentionTracker(gap_reset=gap_reset, **params)
    for i in range(n):
        if time_frames:
            t0 = time.perf_counter_ns()
        smoothed[i], event[i] = tr.update(sess.t[i], sess.s[i])
        if time_frames:
            lat[i] = (time.perf_counter_ns() - t0) / 1e3  # microseconds
        dismissed[i] = tr.dismissed
        attending[i] = tr.attending
    return smoothed, event, dismissed, attending, lat


def system_metrics(sessions: list[Session], params: dict, gap_reset: float, time_frames=False):
    n_look_targets = n_false_targets = n_false_events = 0
    look_seconds = closed_look_seconds = 0.0
    lingers, n_transitions, n_missed, n_already_closed = [], 0, 0, 0
    lats = []

    for sess in sessions:
        smoothed, event, dismissed, _, lat = replay(sess, params, gap_reset, time_frames)
        if time_frames:
            lats.append(lat)

        # False dismissals: dismiss events while the user is looking at the panel.
        false_ev = event & (sess.label == 1)
        n_false_events += int(false_ev.sum())
        look_seconds += float(sess.dt[sess.label == 1].sum())
        closed_look_seconds += float(sess.dt[(sess.label == 1) & dismissed].sum())
        for s, e, lab in sess.target_segs:
            if lab is not None and sess.label[s] == 1:
                n_look_targets += 1
                n_false_targets += int(false_ev[s:e + 1].any())

        # Linger: each look -> away transition (first labelled frame of the away run).
        for k, (s, e, lab) in enumerate(sess.label_runs):
            if lab != 0 or k == 0:
                continue
            n_transitions += 1
            hits = np.flatnonzero(event[s:e + 1])
            if len(hits):
                lingers.append(sess.t[s + hits[0]] - sess.t[s])
            elif dismissed[s]:
                lingers.append(0.0)  # already closed by an earlier (false) dismissal
                n_already_closed += 1
            else:
                n_missed += 1

    lingers = np.array(lingers)
    out = {
        "look_targets": n_look_targets,
        "false_dismissal_target_rate": n_false_targets / n_look_targets if n_look_targets else np.nan,
        "false_dismissals_per_look_min": n_false_events / (look_seconds / 60) if look_seconds else np.nan,
        "closed_while_looking_pct": 100 * closed_look_seconds / look_seconds if look_seconds else np.nan,
        "away_transitions": n_transitions,
        "missed_dismiss_rate": n_missed / n_transitions if n_transitions else np.nan,
        "linger_median_s": float(np.median(lingers)) if len(lingers) else np.nan,
        "linger_p90_s": float(np.percentile(lingers, 90)) if len(lingers) else np.nan,
        "linger_mean_s": float(lingers.mean()) if len(lingers) else np.nan,
    }
    if time_frames and lats:
        a = np.concatenate(lats)
        out["latency_mean_us"] = float(a.mean())
        out["latency_p99_us"] = float(np.percentile(a, 99))
    return out


# --------------------------------------------------------------------------- #
# Loading, sweep, plots
# --------------------------------------------------------------------------- #

def load_sessions(path: Path, score_col: str) -> list[Session]:
    df = pd.read_csv(path, dtype={"person": str, "session": str})
    if score_col not in df.columns:
        sys.exit(f"{path.name} has no column {score_col}. Columns: {list(df.columns)}")
    df = df.dropna(subset=[score_col, "timestamp", "label"])
    df["label"] = df["label"].astype(int)
    return [Session(k, g, score_col) for k, g in df.groupby(["person", "session"])]


def sweep(sessions, gap_reset, max_false_rate):
    grid = itertools.product(
        [0.5, 1.0, 1.5],              # window_seconds
        [0.5, 0.6, 0.7, 0.8],         # open_threshold
        [0.2, 0.3, 0.4, 0.5],         # close_threshold
        [0.5, 1.0, 1.5, 2.0, 3.0],    # dismiss_after_seconds
    )
    rows = []
    for w, o, c, d in grid:
        if c >= o:
            continue
        p = {"window_seconds": w, "open_threshold": o, "close_threshold": c, "dismiss_after_seconds": d}
        rows.append({**p, **system_metrics(sessions, p, gap_reset)})
    df = pd.DataFrame(rows)

    ok = df[(df["false_dismissal_target_rate"] <= max_false_rate) & (df["missed_dismiss_rate"] < 1)]
    if len(ok):
        best = ok.sort_values(["linger_median_s", "false_dismissal_target_rate", "missed_dismiss_rate"]).iloc[0]
        note = f"lowest median linger with false-dismissal target rate <= {max_false_rate:.0%}"
    else:
        best = df.sort_values(["false_dismissal_target_rate", "linger_median_s"]).iloc[0]
        note = f"no setting reached {max_false_rate:.0%} false dismissals; showing the lowest false-dismissal rate"
    return df, best, note


def plot_timeline(sess: Session, params, gap_reset, path: Path, seconds: float):
    smoothed, event, dismissed, _, _ = replay(sess, params, gap_reset)
    m = sess.t <= sess.t[0] + seconds
    t = sess.t[m] - sess.t[0]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for s, e, lab in sess.label_runs:
        if lab == 1 and sess.t[s] - sess.t[0] <= seconds:
            ax.axvspan(sess.t[s] - sess.t[0], sess.t[min(e, len(sess.t) - 1)] - sess.t[0],
                       color="tab:green", alpha=0.15, lw=0)
    ax.plot(t, sess.s[m], color="0.7", lw=0.8, label="frame score")
    ax.plot(t, smoothed[m], color="tab:blue", lw=1.6, label="smoothed")
    ax.axhline(params["open_threshold"], color="tab:green", ls="--", lw=0.8)
    ax.axhline(params["close_threshold"], color="tab:red", ls="--", lw=0.8)
    ev = m & event
    ax.scatter(sess.t[ev] - sess.t[0], np.full(ev.sum(), 0.03), marker="v", color="tab:red", zorder=5,
               label="dismiss")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("looking score")
    ax.set_title(f"Replay of {sess.key[0]} / {sess.key[1]}  (green = truly looking at the panel)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt_table(rows: pd.DataFrame) -> str:
    cols = {
        "model": "model",
        "false_dismissal_target_rate": "false-dismiss targets",
        "false_dismissals_per_look_min": "false/min",
        "closed_while_looking_pct": "closed while looking",
        "linger_median_s": "linger med (s)",
        "linger_p90_s": "linger p90 (s)",
        "missed_dismiss_rate": "missed",
    }
    out = rows[list(cols)].copy()
    for c in ("false_dismissal_target_rate", "missed_dismiss_rate"):
        out[c] = (out[c] * 100).map("{:.0f}%".format)
    out["closed_while_looking_pct"] = out["closed_while_looking_pct"].map("{:.1f}%".format)
    for c in ("false_dismissals_per_look_min", "linger_median_s", "linger_p90_s"):
        out[c] = out[c].map("{:.2f}".format)
    return out.rename(columns=cols).to_string(index=False)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=HERE / "results" / "oof_scores.csv")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--model-json", type=Path, default=HERE / "attention_model.json",
                    help="where default parameters are read from (and written with --update-model)")
    ap.add_argument("--model", default="logreg_full", help="score column to use for sweep/plot (score_<model>)")
    ap.add_argument("--window", type=float, dest="window_seconds")
    ap.add_argument("--open", type=float, dest="open_threshold")
    ap.add_argument("--close", type=float, dest="close_threshold")
    ap.add_argument("--dismiss-after", type=float, dest="dismiss_after_seconds")
    ap.add_argument("--gap-reset", type=float, default=0.4,
                    help="clear the smoothing window across gaps longer than this many seconds (0 = off)")
    ap.add_argument("--sweep", action="store_true", help="search parameters for the chosen model")
    ap.add_argument("--max-false-rate", type=float, default=0.05,
                    help="sweep: max share of looking targets with a false dismissal (default 0.05)")
    ap.add_argument("--update-model", action="store_true",
                    help="write the sweep's best parameters into attention_model.json")
    ap.add_argument("--plot-seconds", type=float, default=90.0)
    args = ap.parse_args()

    if not args.scores.exists():
        sys.exit(f"{args.scores} not found. Run train.py first (it writes oof_scores.csv).")

    params = dict(DEFAULTS)
    if args.model_json.exists():
        saved = json.loads(args.model_json.read_text())
        params.update({k: saved[k] for k in DEFAULTS if k in saved})
    for k in DEFAULTS:
        if getattr(args, k) is not None:
            params[k] = getattr(args, k)
    if params["close_threshold"] >= params["open_threshold"]:
        sys.exit("close threshold must be below the open threshold")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in pd.read_csv(args.scores, nrows=0).columns if c.startswith("score_")]
    if not cols:
        sys.exit("No score_* columns in the scores file.")
    print("Parameters:", ", ".join(f"{k}={v}" for k, v in params.items()), f"| gap_reset={args.gap_reset}")

    # --- replay every model with the same smoothing -------------------------
    rows = []
    for c in cols:
        sessions = load_sessions(args.scores, c)
        rows.append({"model": c.replace("score_", ""), **system_metrics(sessions, params, args.gap_reset, True)})
    table = pd.DataFrame(rows)
    table.to_csv(args.results_dir / "system_metrics.csv", index=False)
    n_sess = len(sessions)
    print(f"\nSystem-level results ({n_sess} session(s), {int(table['look_targets'].iloc[0])} looking targets, "
          f"{int(table['away_transitions'].iloc[0])} look-away transitions):")
    print(fmt_table(table))
    r = table[table["model"] == args.model.replace("score_", "")]
    if len(r) and "latency_mean_us" in r:
        print(f"\nSmoothing compute time per frame ({args.model}): mean {r['latency_mean_us'].iloc[0]:.1f} us, "
              f"p99 {r['latency_p99_us'].iloc[0]:.1f} us (Python; the Swift version is no slower).")
    print("Reading the table: 'false-dismiss targets' = share of looking dots/segments during which the panel "
          "would have closed; 'closed while looking' = share of looking time the panel was closed; "
          "'missed' = look-aways that never closed the panel.")

    # --- optional sweep ----------------------------------------------------
    chosen = params
    sessions = load_sessions(args.scores, f"score_{args.model.replace('score_', '')}")
    if args.sweep:
        df, best, note = sweep(sessions, args.gap_reset, args.max_false_rate)
        df.to_csv(args.results_dir / "smoothing_sweep.csv", index=False)
        print(f"\nSweep over {len(df)} settings for {args.model}: {note}")
        print(f"  window_seconds={best.window_seconds}, open_threshold={best.open_threshold}, "
              f"close_threshold={best.close_threshold}, dismiss_after_seconds={best.dismiss_after_seconds}")
        print(f"  -> false-dismiss targets {best.false_dismissal_target_rate:.0%}, "
              f"closed while looking {best.closed_while_looking_pct:.1f}%, "
              f"linger median {best.linger_median_s:.2f}s / p90 {best.linger_p90_s:.2f}s, "
              f"missed {best.missed_dismiss_rate:.0%}")
        print("  (Tuned and scored on the same sessions: expect slightly worse on new people.)")
        chosen = {k: float(best[k]) for k in DEFAULTS}
        if args.update_model:
            if not args.model_json.exists():
                print(f"  {args.model_json.name} not found; run train.py first. Not updated.")
            else:
                saved = json.loads(args.model_json.read_text())
                saved.update(chosen)
                args.model_json.write_text(json.dumps(saved, indent=2))
                print(f"  Updated {args.model_json.name} with these parameters.")

    # --- per-frame replay file and timeline plot ---------------------------
    frames = []
    for sess in sessions:
        sm, ev, dis, att, _ = replay(sess, chosen, args.gap_reset)
        f = sess.d[["person", "session", "frame", "timestamp", "label", "target"]].copy()
        f["score"], f["smoothed"], f["attending"] = sess.s, sm, att
        f["dismissed"], f["dismiss_event"] = dis, ev
        frames.append(f)
    pd.concat(frames).to_csv(args.results_dir / "replay_frames.csv", index=False)
    plot_timeline(sessions[0], chosen, args.gap_reset, args.results_dir / "smoothing_timeline.png",
                  args.plot_seconds)
    print(f"\nWrote system_metrics.csv, replay_frames.csv, smoothing_timeline.png"
          f"{', smoothing_sweep.csv' if args.sweep else ''} to {args.results_dir}")


if __name__ == "__main__":
    main()