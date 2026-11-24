#!/usr/bin/env python3
"""MediaPipe feature extraction for Glance attention detection.

Reads the recordings made by collect.py and writes one feature CSV per session
in the format described in the README:

    raw/<person>_<session>.mp4
    raw/<person>_<session>_labels.csv      -->   data/mp_<person>_<session>.csv

Only frames that have a label (i.e. not the settle / countdown frames) are
processed and written. Frame index and timestamp come straight from the labels
CSV, so the timing is the real capture time, not the video's playback time.

Features (one row per frame):
    face_present          1 if MediaPipe found a face, else 0
    yaw, pitch, roll      head angles in radians, from solvePnP on 6 face landmarks
                          (generic face model; fine for relative changes, not
                          calibrated absolute angles). Frontal face is ~0, 0, 0.
    face_x, face_y, face_w, face_h
                          bounding box of all landmarks, normalized to the frame
                          (x, y = top-left corner)
    pupil_lx, pupil_ly, pupil_rx, pupil_ry
                          iris centre relative to each eye's corners.
                          x: 0 = image-left corner, 1 = image-right corner (same
                             direction for both eyes, so "looking to image-left" is
                             low for both).
                          y: offset from the corner midpoint, positive = down, in
                             units of eye width (0 = centred).
                          "l"/"r" are the subject's left/right eye.
    eye_open_l, eye_open_r
                          lid gap divided by eye width (like the eye aspect ratio)

When no face is found, face_present is 0 and every other feature is left empty;
train.py should decide how to fill those (e.g. 0 after standardizing).

Usage:
    python features_mp.py                       # every session in raw/
    python features_mp.py mitchel_1 mitchel_2   # specific sessions
    python features_mp.py --force               # overwrite existing CSVs

Requires: opencv-python, numpy, mediapipe (legacy `mp.solutions.face_mesh`).
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
DATA_DIR = HERE / "data"

OUT_COLS = [
    "person", "session", "frame", "timestamp", "dot_x", "dot_y", "label", "laptop",
    "face_present", "yaw", "pitch", "roll",
    "face_x", "face_y", "face_w", "face_h",
    "pupil_lx", "pupil_ly", "pupil_rx", "pupil_ry",
    "eye_open_l", "eye_open_r",
    "target",
]
FEATURE_COLS = OUT_COLS[8:22]

# --- Landmark indices (MediaPipe face mesh with refine_landmarks=True) -------
# Head pose: nose tip, chin, outer eye corners, mouth corners.
# 33 / 61 are on the image-left side (the subject's right), 263 / 291 on the image-right.
POSE_IDX = [1, 152, 33, 263, 61, 291]
# Generic 3D face model (mm) in camera-style axes: x = image-right, y = down,
# z = away from the camera. A frontal face then gives a rotation close to identity.
POSE_MODEL = np.array([
    [0.0, 0.0, 0.0],          # nose tip
    [0.0, 63.6, 12.5],        # chin
    [-43.3, -32.7, 26.0],     # outer corner, image-left eye
    [43.3, -32.7, 26.0],      # outer corner, image-right eye
    [-28.9, 28.9, 24.1],      # mouth corner, image-left
    [28.9, 28.9, 24.1],       # mouth corner, image-right
], dtype=np.float64)

# Each eye: (image-left corner, image-right corner, upper lid, lower lid, iris centre)
EYE_R = (33, 133, 159, 145, 468)    # subject's right eye (image left)
EYE_L = (362, 263, 386, 374, 473)   # subject's left eye (image right)


# --------------------------------------------------------------------------- #
# Feature math
# --------------------------------------------------------------------------- #

def head_pose(pts: np.ndarray, w: int, h: int):
    """(yaw, pitch, roll) in radians from pixel landmarks, or None if solvePnP fails."""
    image_pts = pts[POSE_IDX].astype(np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    # Start from "frontal face, ~600 mm in front of the camera". Without this initial guess,
    # solvePnP often fails or flips on strongly turned heads (e.g. "look left" / "look right").
    ok, rvec, tvec = cv2.solvePnP(POSE_MODEL, image_pts, cam, np.zeros(4),
                                  rvec=np.zeros((3, 1)), tvec=np.array([[0.0], [0.0], [600.0]]),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:  # face must be in front of the camera
        return None
    rot, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = cv2.RQDecomp3x3(rot)[0]  # degrees about x, y, z
    return np.radians(yaw), np.radians(pitch), np.radians(roll)


def eye_features(pts: np.ndarray, idx) -> tuple[float, float, float]:
    """(pupil_x, pupil_y, openness) for one eye, from pixel landmarks."""
    left, right, up, low, iris = (pts[i] for i in idx)
    width = float(np.linalg.norm(right - left))
    if width < 1e-6:
        return float("nan"), float("nan"), float("nan")
    u = (right - left) / width
    v = np.array([-u[1], u[0]])  # perpendicular, pointing down in image coordinates
    centre = (left + right) / 2
    px = float(np.dot(iris - left, u) / width)
    py = float(np.dot(iris - centre, v) / width)
    openness = float(np.linalg.norm(up - low) / width)
    return px, py, openness


def frame_features(landmarks, w: int, h: int) -> dict:
    """All feature columns for one detected face."""
    n = len(landmarks)
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)
    feats = {"face_present": 1}

    pose = head_pose(pts, w, h)
    feats["yaw"], feats["pitch"], feats["roll"] = pose if pose else (None, None, None)

    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    feats.update(face_x=x0 / w, face_y=y0 / h, face_w=(x1 - x0) / w, face_h=(y1 - y0) / h)

    if n >= 478:  # iris landmarks exist only with refine_landmarks=True
        feats["pupil_lx"], feats["pupil_ly"], feats["eye_open_l"] = eye_features(pts, EYE_L)
        feats["pupil_rx"], feats["pupil_ry"], feats["eye_open_r"] = eye_features(pts, EYE_R)
    return feats


# --------------------------------------------------------------------------- #
# Session processing
# --------------------------------------------------------------------------- #

def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.6f}"
    return str(v)


def load_labels(path: Path) -> dict[int, dict]:
    with open(path, newline="") as f:
        return {int(r["frame"]): r for r in csv.DictReader(f)}


def process_session(stem: str, face_mesh, force: bool) -> bool:
    labels_path = RAW_DIR / f"{stem}_labels.csv"
    video_path = RAW_DIR / f"{stem}.mp4"
    out_path = DATA_DIR / f"mp_{stem}.csv"

    if not labels_path.exists() or not video_path.exists():
        print(f"[{stem}] missing {labels_path.name} or {video_path.name}, skipping")
        return False
    if out_path.exists() and not force:
        print(f"[{stem}] {out_path.name} exists (use --force to overwrite), skipping")
        return False

    labels = load_labels(labels_path)
    if not labels:
        print(f"[{stem}] labels file is empty, skipping")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[{stem}] cannot open video, skipping")
        return False

    last_needed = max(labels)
    rows = []
    idx = 0
    print(f"[{stem}] {len(labels)} labelled frames to process...")
    while idx <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in labels:
            h, w = frame.shape[:2]
            res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                feats = frame_features(res.multi_face_landmarks[0].landmark, w, h)
            else:
                feats = {"face_present": 0}
            lab = labels[idx]
            row = {c: lab.get(c, "") for c in
                   ("person", "session", "frame", "timestamp", "dot_x", "dot_y",
                    "label", "laptop", "target")}
            row.update({c: feats.get(c) for c in FEATURE_COLS})
            rows.append(row)
        idx += 1
        if idx % 500 == 0:
            print(f"[{stem}]   {idx} frames read")
    cap.release()

    if len(rows) < len(labels):
        print(f"[{stem}] warning: video ended early; got {len(rows)} of {len(labels)} labelled frames. "
              "(Was the video file cut off or re-encoded?)")

    DATA_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(OUT_COLS)
        for r in rows:
            wr.writerow([fmt(r.get(c)) if c in FEATURE_COLS else r.get(c, "") for c in OUT_COLS])

    summarize(stem, rows)
    print(f"[{stem}] wrote {out_path.relative_to(HERE)}")
    return True


def summarize(stem: str, rows: list[dict]):
    """Quick sanity check printed after each session."""
    def col(rs, c):
        return np.array([np.nan if r.get(c) is None else float(r[c]) for r in rs])

    print(f"[{stem}] summary (frames / face detected / mean yaw / mean pitch / mean pupil_x L,R):")
    for lab in ("1", "0"):
        rs = [r for r in rows if str(r["label"]) == lab]
        if not rs:
            continue
        det = col(rs, "face_present").mean() * 100
        with np.errstate(all="ignore"):
            yaw = np.nanmean(col(rs, "yaw"))
            pitch = np.nanmean(col(rs, "pitch"))
            plx = np.nanmean(col(rs, "pupil_lx"))
            prx = np.nanmean(col(rs, "pupil_rx"))
        name = "looking    " if lab == "1" else "not looking"
        print(f"    {name}: {len(rs):5d} frames, {det:5.1f}% face, "
              f"yaw {yaw:+.3f}, pitch {pitch:+.3f}, pupil_x {plx:.3f}, {prx:.3f}")
    total_det = col(rows, "face_present").mean() * 100
    if total_det < 90:
        print(f"[{stem}] warning: face found in only {total_det:.0f}% of frames. "
              "Check lighting / camera framing; these sessions may be poor training data.")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session stems, e.g. mitchel_1 (default: all in raw/)")
    ap.add_argument("--force", action="store_true", help="overwrite existing feature CSVs")
    ap.add_argument("--include-test", action="store_true", help="also process TEST_* quick runs")
    args = ap.parse_args()

    try:
        import mediapipe as mp
        face_mesh_mod = mp.solutions.face_mesh
    except Exception as e:  # ImportError, or AttributeError if this mediapipe build dropped solutions
        sys.exit(f"Could not load mediapipe's face mesh ({e}).\n"
                 "Install a version that includes the legacy solutions API, e.g.\n"
                 "    pip install \"mediapipe==0.10.21\"")

    if args.sessions:
        stems = args.sessions
    else:
        stems = sorted(p.name[: -len("_labels.csv")] for p in RAW_DIR.glob("*_labels.csv"))
        if not args.include_test:
            stems = [s for s in stems if not s.startswith("TEST_")]
    if not stems:
        sys.exit(f"No sessions found in {RAW_DIR}. Record one with collect.py first.")

    # static_image_mode=True: every frame is processed independently (no tracking
    # smoothing), which is closer to how the app will see each frame.
    with face_mesh_mod.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True, min_detection_confidence=0.5) as fm:
        done = sum(process_session(s, fm, args.force) for s in stems)
    print(f"\nDone: {done} of {len(stems)} session(s) processed.")
    print("Commit only data/*.csv. Never commit raw/ (video).")


if __name__ == "__main__":
    main()