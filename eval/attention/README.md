# Attention Detection

Decides whether the user is looking at Glance's reply panel (top center of the screen, just under the camera), so the reply stays open while they read and dismisses when they look away.

This folder is platform-independent Python. The app side (live webcam features and applying the model) is Swift and lives in `Glance/`.

## Pipeline

```
webcam frame → face features → logistic regression → smoothing → keep open / dismiss
```

The trained model is exported as `attention_model.json` and loaded by the app. No CoreML or deep-learning framework is needed.

## Setup

```bash
cd eval/attention
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, the first webcam capture asks for camera permission for Terminal / VS Code. Allow it, or capture fails silently.

## Layout

```
eval/attention/
  collect.py        # dot-target recording script (OpenCV)
  features_mp.py    # MediaPipe feature extraction (any OS)
  train.py          # training + leave-one-person-out evaluation
  smoothing.py      # dismiss logic, replayed on recorded sessions
  notebooks/        # exploration
  data/             # feature + label CSVs only (committed)
  raw/              # frames/video (git-ignored, keep in the private shared drive)
  results/          # plots and metric tables for the report
```

## Data collection

`collect.py` shows a full-screen window with a dot at a sequence of positions for about 3 s each, records the webcam, and labels every frame from the dot position:

- **looking (1):** the dot is inside the reply panel area, a top-center box about 640×200 points
- **not looking (0):** the dot is elsewhere on screen, or the screen shows a prompt such as "look at your phone", "look left", "look at your keyboard"

Record several sessions per person and vary lighting, glasses, distance and posture. Do most sessions on the MacBook (the real camera position); sessions on other laptops form a separate generalization test.

Store dot positions as fractions of the screen (0–1). Draw in actual screen pixels so labels stay correct on Retina displays.

**Privacy:** raw frames never go in this repo (it's public). Commit only CSVs.

## CSV format

`data/<extractor>_<person>_<session>.csv`, one row per frame:

| column | meaning |
|---|---|
| `person`, `session`, `frame`, `timestamp` | identifiers, timestamp in seconds |
| `dot_x`, `dot_y` | target position, 0–1 |
| `label` | 1 = looking at panel, 0 = not |
| `laptop` | which machine recorded it |
| `face_present` | 0/1 |
| `yaw`, `pitch`, `roll` | head angles, radians |
| `face_x`, `face_y`, `face_w`, `face_h` | face box, normalized to the frame |
| `pupil_lx`, `pupil_ly`, `pupil_rx`, `pupil_ry` | pupil position inside each eye outline, normalized |
| `eye_open_l`, `eye_open_r` | eye openness |

Two extractors produce this format: `mp` (MediaPipe, here) and `vision` (Apple Vision, a Swift tool run on the Mac over the same frames). The app uses Apple Vision, so the final model trains on `vision` CSVs. `mp` lets you build everything now and is an extra comparison in the report.

## Model export

`attention_model.json`:

```json
{
  "features": ["face_present", "yaw", "pitch", "..."],
  "mean": [0.0],
  "std": [1.0],
  "weights": [0.0],
  "bias": 0.0,
  "open_threshold": 0.6,
  "close_threshold": 0.4,
  "window_seconds": 1.0,
  "dismiss_after_seconds": 1.5
}
```

## Evaluation

Compare:
1. face detected = looking
2. fixed head-angle threshold
3. trained classifier (with pupil features)

Report:
- **Frame-level:** accuracy, F1, ROC-AUC with leave-one-person-out testing; also train on the MacBook and test on other laptops
- **System-level** (sessions replayed through `smoothing.py`): false dismissals (closed while still reading), linger time (open after looking away), per-frame latency
- **Comparison:** MediaPipe features vs Apple Vision features

## Deliverables

- `attention_model.json` handed to the app
- plots and tables in `results/`
- the attention sections of the report (Data, Methods, Experiments & Results)
