"""Before running the full script, run the following command to check the fps

"python collect.py --check-camera --width 640 --height 480"

If the fps is too low, add in "--exposure -5" or "--exposure -6" to the command above. Also be sure you are in a bright area.

use the command below to run this file, add in your name, session number, and computer name. If the testing before showed the fps is too low,
be sure to add "--exposure #" as well.

python collect.py --person <name> --session <session_number> --laptop <computer_name> --width 640 --height 480"""

import argparse
import csv
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "raw"
WINDOW = "Glance attention collection"

BG = (30, 30, 30)
FG = (240, 240, 240)
DOT = (60, 60, 230)  # BGR red

@dataclass
class Target:
    name: str
    kind: str                 # "dot" or "prompt"
    label: int                # 1 = looking at panel, 0 = not
    x: float | None = None    # fraction of screen width (dots only)
    y: float | None = None    # fraction of screen height (dots only)
    text: str | None = None   # prompt text (prompts only)

class Panel:
    """Top-center reply-panel box, in screen fractions."""

    def __init__(self, w_frac: float, h_frac: float):
        self.w = w_frac
        self.h = h_frac
        self.x0 = 0.5 - w_frac / 2
        self.x1 = 0.5 + w_frac / 2
        self.y0 = 0.0
        self.y1 = h_frac

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

PROMPTS = [
    "Look at your phone",
    "Look left",
    "Look right",
    "Look at your keyboard",
    "Look at the ceiling",
    "Look away from the screen",
]

def build_targets(panel: Panel, rounds: int, rng: random.Random) -> list[Target]:
    """Build a shuffled target list. Every dot's label comes from panel.contains()."""
    targets: list[Target] = []

    def clamp(v: float, lo: float = 0.03, hi: float = 0.97) -> float:
        return max(lo, min(hi, v))

    def jitter(v: float, amount: float = 0.02) -> float:
        return v + rng.uniform(-amount, amount)

    def add_dot(name: str, x: float, y: float):
        x, y = clamp(x), clamp(y)
        targets.append(Target(name, "dot", int(panel.contains(x, y)), x=x, y=y))

    for r in range(rounds):
        # Looking: spread across the panel (inset so labels are unambiguous).
        for fx in (0.12, 0.37, 0.63, 0.88):
            for fy in (0.25, 0.5, 0.75):
                x = panel.x0 + fx * panel.w
                y = panel.y0 + fy * panel.h
                add_dot(f"panel_{fx:.2f}_{fy:.2f}_r{r}", jitter(x, 0.01), jitter(y, 0.01))

        # Not looking: coarse grid over the screen, skipping anything near/in the panel.
        for gy in (0.3, 0.5, 0.7, 0.9):
            for gx in (0.1, 0.5, 0.9):
                x, y = jitter(gx), jitter(gy)
                if panel.contains(x, y) or y < panel.y1 + 0.05:
                    continue
                add_dot(f"grid_{gx:.1f}_{gy:.1f}_r{r}", x, y)

        # Hard negatives: just outside the panel box.
        margin = 0.06
        add_dot(f"below_panel_r{r}", jitter(0.5), panel.y1 + margin + rng.uniform(0, 0.03))
        add_dot(f"left_of_panel_r{r}", panel.x0 - margin - rng.uniform(0, 0.04), panel.h / 2)
        add_dot(f"right_of_panel_r{r}", panel.x1 + margin + rng.uniform(0, 0.04), panel.h / 2)
        add_dot(f"top_corner_l_r{r}", jitter(0.05, 0.01), jitter(0.05, 0.01))
        add_dot(f"top_corner_r_r{r}", jitter(0.95, 0.01), jitter(0.05, 0.01))

        # Off-screen prompts (always label 0).
        for i, text in enumerate(PROMPTS):
            targets.append(Target(f"prompt_{i}_r{r}", "prompt", 0, text=text))

    rng.shuffle(targets)

    # Avoid long runs of the same label: interleave a bit by a simple repair pass.
    for i in range(2, len(targets)):
        if targets[i].label == targets[i - 1].label == targets[i - 2].label:
            for j in range(i + 1, len(targets)):
                if targets[j].label != targets[i].label:
                    targets[i], targets[j] = targets[j], targets[i]
                    break
    return targets

def get_screen_points() -> tuple[int, int] | None:
    """Screen size in points (what macOS layout uses), via tkinter if available."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return None

def get_canvas_size(fallback: tuple[int, int]) -> tuple[int, int]:
    """Actual pixel size of the fullscreen window, falling back if unavailable."""
    for _ in range(10):
        cv2.waitKey(50)
        try:
            _, _, w, h = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            break
        if w > 0 and h > 0:
            return w, h
    return fallback

def draw_text_center(img, text: str, scale: float, thickness: int, y_off: int = 0):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2 + y_off),
                cv2.FONT_HERSHEY_SIMPLEX, scale, FG, thickness, cv2.LINE_AA)

def render_dot(canvas_size: tuple[int, int], x: float, y: float, t: float) -> np.ndarray:
    w, h = canvas_size
    img = np.full((h, w, 3), BG, np.uint8)
    cx, cy = int(x * w), int(y * h)
    base = max(10, h // 70)
    r = int(base * (1.0 + 0.15 * math.sin(t * 6)))  # gentle pulse keeps gaze on the dot
    cv2.circle(img, (cx, cy), r, DOT, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, r // 4), (255, 255, 255), -1, cv2.LINE_AA)
    return img

def render_prompt(canvas_size: tuple[int, int], text: str) -> np.ndarray:
    w, h = canvas_size
    img = np.full((h, w, 3), BG, np.uint8)
    draw_text_center(img, text, scale=h / 450, thickness=max(2, h // 400))
    return img

def render_message(canvas_size: tuple[int, int], lines: list[str]) -> np.ndarray:
    w, h = canvas_size
    img = np.full((h, w, 3), BG, np.uint8)
    step = h // 12
    top = -step * (len(lines) - 1) // 2
    for i, line in enumerate(lines):
        draw_text_center(img, line, scale=h / 700, thickness=max(1, h // 600), y_off=top + i * step)
    return img

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--person", default="test", help="participant id (no spaces), e.g. alice")
    p.add_argument("--session", default="0", help="session id, e.g. 1")
    p.add_argument("--laptop", default="unknown", help="machine name, e.g. macbook-air-m2")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0)")
    p.add_argument("--backend", choices=["dshow", "msmf", "any"], default="dshow",
                   help="Windows only: OpenCV camera backend (default dshow; try msmf or any if it fails)")
    p.add_argument("--width", type=int, default=1280, help="requested camera width (default 1280)")
    p.add_argument("--height", type=int, default=720, help="requested camera height (default 720)")
    p.add_argument("--fourcc", default="MJPG",
                   help="requested camera pixel format, e.g. MJPG (default) or 'none' to leave it alone")
    p.add_argument("--check-camera", action="store_true",
                   help="only open the camera, print the measured fps (~3 s), and exit")
    p.add_argument("--quick", action="store_true",
                   help="short test run (8 targets, ~1.5 s each, ~15 s total); files are prefixed TEST_")
    p.add_argument("--exposure", type=float, default=None,
                   help="turn off auto-exposure and set manual exposure (DirectShow: log2 seconds, "
                        "e.g. -5 = 1/32 s, -6 = 1/64 s). Short exposures allow higher fps.")
    p.add_argument("--rounds", type=int, default=2, help="repeats of the target set (default 2)")
    p.add_argument("--duration", type=float, default=3.0, help="seconds per target (default 3)")
    p.add_argument("--settle", type=float, default=0.8,
                   help="seconds after each target change before frames are labelled (default 0.8)")
    p.add_argument("--panel-points", type=float, nargs=2, default=(640, 200), metavar=("W", "H"),
                   help="reply panel size in points (default 640 200)")
    p.add_argument("--screen-points", type=int, nargs=2, metavar=("W", "H"),
                   help="screen size in points; auto-detected via tkinter if omitted "
                        "(e.g. 1512 982 for a 14-inch MacBook Pro)")
    p.add_argument("--seed", type=int, default=None, help="shuffle seed (default: random)")
    p.add_argument("--lighting", default="", help="free-text note, saved to the meta file")
    p.add_argument("--notes", default="", help="free-text note (glasses, posture, distance...)")
    p.add_argument("--no-fullscreen", action="store_true", help="debug: use a normal window")
    return p.parse_args()

def main():
    args = parse_args()

    for field in (args.person, args.session):
        if any(c.isspace() or c in "/\\" for c in field):
            sys.exit("person/session must not contain whitespace or slashes")

    screen_pts = tuple(args.screen_points) if args.screen_points else get_screen_points()
    if screen_pts is None:
        sys.exit("Could not detect screen size. Pass --screen-points W H "
                 "(System Settings > Displays shows the 'looks like' resolution).")
    panel = Panel(args.panel_points[0] / screen_pts[0], args.panel_points[1] / screen_pts[1])
    print(f"Screen: {screen_pts[0]}x{screen_pts[1]} pt | panel: "
          f"x {panel.x0:.3f}-{panel.x1:.3f}, y 0-{panel.y1:.3f} (fractions)")

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    targets = build_targets(panel, args.rounds, random.Random(seed))
    if args.quick:
        args.duration, args.settle = 1.5, 0.5
        looks = [t for t in targets if t.label == 1][:4]
        nots = [t for t in targets if t.label == 0][:4]
        keep = {id(t) for t in looks + nots}
        targets = [t for t in targets if id(t) in keep]
    n_look = sum(t.label for t in targets)
    print(f"{len(targets)} targets ({n_look} looking, {len(targets) - n_look} not), "
          f"~{len(targets) * args.duration / 60:.1f} min")

    print("Opening webcam (can take a few seconds)...", flush=True)
    if sys.platform == "win32":
        backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[args.backend]
        cap = cv2.VideoCapture(args.camera, backend)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Cannot open webcam. On macOS, allow camera access for your terminal / VS Code "
                 "in System Settings > Privacy & Security > Camera.")
    if args.fourcc != "none":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode on DirectShow
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    print("Webcam opened, reading first frame...", flush=True)
    ok, frame = cap.read()
    if not ok or frame is None:
        sys.exit("Webcam opened but returned no frames (check camera permission).")
    fh, fw = frame.shape[:2]
    n, tb = 0, time.monotonic()
    while time.monotonic() - tb < 1.5:
        if cap.read()[0]:
            n += 1
    fps = n / (time.monotonic() - tb)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"
    print(f"Camera: {fw}x{fh}, codec {codec}, measured {fps:.1f} fps", flush=True)
    if fps < 20:
        print("warning: camera is delivering under 20 fps. Try --width 640 --height 480, "
              "or --fourcc MJPG / --fourcc none, or --backend msmf.", file=sys.stderr)
    fps = max(fps, 1.0)
    if args.check_camera:
        cap.release()
        sys.exit(0)

    RAW_DIR.mkdir(exist_ok=True)
    stem = f"{'TEST_' if args.quick else ''}{args.person}_{args.session}"
    video_path = RAW_DIR / f"{stem}.mp4"
    labels_path = RAW_DIR / f"{stem}_labels.csv"
    meta_path = RAW_DIR / f"{stem}_meta.txt"
    if not args.quick and (video_path.exists() or labels_path.exists()):
        sys.exit(f"{stem} already exists in raw/. Pick a new --session or delete the old files.")

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    labels_f = open(labels_path, "w", newline="")
    labels = csv.writer(labels_f)
    labels.writerow(["person", "session", "frame", "timestamp", "dot_x", "dot_y",
                     "label", "laptop", "target"])

    print("Creating window...", flush=True)
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    if not args.no_fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW, np.full((200, 300, 3), BG, np.uint8))
    fallback = (screen_pts[0] * 2, screen_pts[1] * 2)  # assume Retina if the rect is unavailable
    canvas = get_canvas_size(fallback)
    if canvas[0] > 1280:
        sc = 1280 / canvas[0]
        canvas = (1280, int(canvas[1] * sc))
    print(f"Canvas: {canvas[0]}x{canvas[1]} px (stretched to fill the screen)")

    aborted = False
    frame_idx = 0
    t0 = None

    def grab_and_record(img_to_show, target: Target | None, labelled: bool):
        """Show an image, read one webcam frame, write video (+ label row if labelled)."""
        nonlocal frame_idx
        cv2.imshow(WINDOW, img_to_show)
        key = cv2.waitKey(1) & 0xFF
        ok, frm = cap.read()
        if not ok or frm is None:
            return key, False
        ts = time.monotonic() - t0
        writer.write(frm)
        if labelled and target is not None:
            labels.writerow([
                args.person, args.session, frame_idx, f"{ts:.4f}",
                "" if target.x is None else f"{target.x:.5f}",
                "" if target.y is None else f"{target.y:.5f}",
                target.label, args.laptop, target.name,
            ])
        frame_idx += 1
        return key, True

    try:
        # Wait for SPACE.
        intro = render_message(canvas, [
            "Sit as you normally would and keep your head natural.",
            "Follow the dot with your eyes. When text appears, do what it says.",
            "Press SPACE to start (ESC to abort)",
        ])
        while True:
            cv2.imshow(WINDOW, intro)
            key = cv2.waitKey(30) & 0xFF
            cap.read()  # keep the camera warm / auto-exposure settling
            if key == 32:
                break
            if key in (27, ord("q")):
                aborted = True
                break

        # 3-second countdown (recorded but unlabelled so frame indices stay aligned).
        t0 = time.monotonic()
        if not aborted:
            end = time.monotonic() + 3.0
            while time.monotonic() < end and not aborted:
                n = int(end - time.monotonic()) + 1
                key, _ = grab_and_record(render_message(canvas, [f"Starting in {n}"]), None, False)
                aborted = key in (27, ord("q"))

        for i, tgt in enumerate(targets):
            if aborted:
                break
            print(f"[{i + 1}/{len(targets)}] {tgt.name} (label={tgt.label})")
            tstart = time.monotonic()
            while True:
                el = time.monotonic() - tstart
                if el >= args.duration:
                    break
                if tgt.kind == "dot":
                    img = render_dot(canvas, tgt.x, tgt.y, el)
                else:
                    img = render_prompt(canvas, tgt.text)
                key, got = grab_and_record(img, tgt, labelled=el >= args.settle)
                if key in (27, ord("q")):
                    aborted = True
                    break
                if not got:
                    print("warning: dropped webcam frame", file=sys.stderr)
                    
    finally:
        cap.release()
        writer.release()
        labels_f.close()
        cv2.destroyAllWindows()

    elapsed = (time.monotonic() - t0) if t0 else 0.0
    eff_fps = frame_idx / elapsed if elapsed > 0 else 0.0
    meta_path.write_text(
        f"person={args.person}\nsession={args.session}\nlaptop={args.laptop}\n"
        f"lighting={args.lighting}\nnotes={args.notes}\n"
        f"screen_points={screen_pts[0]}x{screen_pts[1]}\ncanvas_px={canvas[0]}x{canvas[1]}\n"
        f"panel_points={args.panel_points[0]}x{args.panel_points[1]}\n"
        f"panel_fraction=x[{panel.x0:.4f},{panel.x1:.4f}] y[0,{panel.y1:.4f}]\n"
        f"camera_frame={fw}x{fh}@{fps:.1f}fps\nseed={seed}\naborted={aborted}\n"
        f"frames_recorded={frame_idx}\neffective_fps={eff_fps:.1f}\n"
    )

    print(f"\n{'Aborted; partial data kept.' if aborted else 'Done.'} "
          f"{frame_idx} frames (~{eff_fps:.1f} fps) -> {video_path.name}, {labels_path.name}")
    if eff_fps and eff_fps < 20:
        print("warning: effective frame rate is low; use the label CSV timestamps, not the video's.")
    print("Reminder: raw/ is git-ignored. Do not commit video; commit only feature CSVs.")


if __name__ == "__main__":
    main()