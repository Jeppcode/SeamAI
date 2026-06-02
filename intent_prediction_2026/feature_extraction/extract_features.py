#!/usr/bin/env python3
"""
Author: Jesper Malmgren
e-mail: malmgren.jesper@gmail.com

extract_features.py

Build the handcrafted feature table from a folder of collected clips (the output
of ../data_collection/pose_data_collection.py).

For every kept clip it computes the 30-dimensional handcrafted motion-feature
vector: 10 base features, each summarised over an observation window by its
mean, variance and latest value. The window ends tte_seconds before the event
frame (closest approach to the door for "enter", last visible frame for "pass").
One CSV row is written per (clip, tte). Several tte horizons can be requested in
one run.

The script is self-contained: it reads only the clip JSON files and writes a CSV,
with no imports from the rest of the project.

The 10 base features
  1 distance to door        6 step displacement
  2 closure rate            7 heading angle
  3 vx                      8 relative angle to door
  4 vy                      9 bounding-box aspect ratio
  5 absolute speed         10 bounding-box scale-change rate

Usage
-----
  python extract_features.py                                  # ../data, tte=2.0s, window=0.5s
  python extract_features.py --data-root ../data --out features.csv
  python extract_features.py --tte-seconds 1.0 1.5 2.0 2.5 3.0
  python extract_features.py --include-exit                   # also emit exit clips

Output CSV columns:
  source, label, fps, n_frames, tte_seconds, window_seconds, then the 30 feature
  columns named <feature>_<mean|var|latest>.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

# Used only if a clip JSON has no fps (clips collected by pose_data_collection.py
# always carry a measured fps, so this is effectively never hit).
DEFAULT_FPS = 13.0


# ---------------------------------------------------------------------------
# Parse raw frame data
# ---------------------------------------------------------------------------
def get_coordinates(frames):
    """Split the frame list into x, y, x1, y1, x2, y2 (NaN where not detected)."""
    x, y = [], []
    x1, y1 = [], []
    x2, y2 = [], []
    for frame in frames:
        center = frame.get("center")
        bbox = frame.get("bbox")
        if center is not None and len(center) == 2:
            x.append(float(center[0]))
            y.append(float(center[1]))
        else:
            x.append(float("nan"))
            y.append(float("nan"))
        if bbox is not None and len(bbox) == 4:
            x1.append(float(bbox[0]))
            y1.append(float(bbox[1]))
            x2.append(float(bbox[2]))
            y2.append(float(bbox[3]))
        else:
            x1.append(float("nan"))
            y1.append(float("nan"))
            x2.append(float("nan"))
            y2.append(float("nan"))
    return x, y, x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Locate the event frame
# ---------------------------------------------------------------------------
def find_event_frame(label, x, y, door_x, door_y):
    """
    Frame index of the key event:
      enter        -> frame closest to the door center
      pass / exit  -> last frame the person is visible
    Returns -1 if there are no detected frames.
    """
    valid = [i for i in range(len(x)) if not math.isnan(x[i]) and not math.isnan(y[i])]
    if len(valid) == 0:
        return -1
    if label in ("pass", "exit"):
        return valid[-1]
    if label == "enter":
        best_idx = valid[0]
        best_dist = float("inf")
        for i in valid:
            dist = math.sqrt((x[i] - door_x) ** 2 + (y[i] - door_y) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx
    return -1


# ---------------------------------------------------------------------------
# Feature time-series
# ---------------------------------------------------------------------------
def compute_distance_to_door(x, y, door_x, door_y):
    out = []
    for i in range(len(x)):
        if math.isnan(x[i]) or math.isnan(y[i]):
            out.append(float("nan"))
        else:
            out.append(math.sqrt((x[i] - door_x) ** 2 + (y[i] - door_y) ** 2))
    return out


def compute_closure_rate(d_door, fps, k):
    dt_k = k / fps
    out = []
    for i in range(len(d_door)):
        prev_i = i - k
        if prev_i < 0 or math.isnan(d_door[i]) or math.isnan(d_door[prev_i]):
            out.append(float("nan"))
        else:
            out.append((d_door[i] - d_door[prev_i]) / dt_k)
    return out


def compute_velocity(x, y, fps, k):
    dt_k = k / fps
    vx, vy = [], []
    for i in range(len(x)):
        prev_i = i - k
        if prev_i < 0 or math.isnan(x[i]) or math.isnan(x[prev_i]):
            vx.append(float("nan"))
        else:
            vx.append((x[i] - x[prev_i]) / dt_k)
        if prev_i < 0 or math.isnan(y[i]) or math.isnan(y[prev_i]):
            vy.append(float("nan"))
        else:
            vy.append((y[i] - y[prev_i]) / dt_k)
    return vx, vy


def compute_absolute_speed(vx, vy):
    out = []
    for i in range(len(vx)):
        if math.isnan(vx[i]) or math.isnan(vy[i]):
            out.append(float("nan"))
        else:
            out.append(math.sqrt(vx[i] ** 2 + vy[i] ** 2))
    return out


def compute_step_displacement(x, y):
    out = [float("nan")]
    for i in range(1, len(x)):
        if math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(x[i - 1]) or math.isnan(y[i - 1]):
            out.append(float("nan"))
        else:
            out.append(math.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2))
    return out


def compute_heading_and_relative_angle(vx, vy, x, y, door_x, door_y):
    heading, rel_angle = [], []
    for i in range(len(vx)):
        if math.isnan(vx[i]) or math.isnan(vy[i]):
            heading.append(float("nan"))
            rel_angle.append(float("nan"))
            continue
        theta = math.atan2(vy[i], vx[i])
        heading.append(theta)
        if math.isnan(x[i]) or math.isnan(y[i]):
            rel_angle.append(float("nan"))
        else:
            theta_door = math.atan2(door_y - y[i], door_x - x[i])
            diff = theta - theta_door
            diff_wrapped = math.atan2(math.sin(diff), math.cos(diff))
            rel_angle.append(abs(diff_wrapped))
    return heading, rel_angle


def compute_bbox_features(x1, y1, x2, y2, fps, k):
    dt_k = k / fps
    aspect_ratio, scale_change = [], []
    for i in range(len(x1)):
        if any(math.isnan(v) for v in [x1[i], y1[i], x2[i], y2[i]]):
            aspect_ratio.append(float("nan"))
        else:
            width = x2[i] - x1[i]
            height = y2[i] - y1[i]
            aspect_ratio.append(float("nan") if height == 0 else width / height)
        prev_i = i - k
        if prev_i < 0:
            scale_change.append(float("nan"))
        else:
            h_now = (y2[i] - y1[i]) if not (math.isnan(y1[i]) or math.isnan(y2[i])) else float("nan")
            h_prev = (y2[prev_i] - y1[prev_i]) if not (math.isnan(y1[prev_i]) or math.isnan(y2[prev_i])) else float("nan")
            if math.isnan(h_now) or math.isnan(h_prev):
                scale_change.append(float("nan"))
            else:
                scale_change.append((h_now - h_prev) / dt_k)
    return aspect_ratio, scale_change


# ---------------------------------------------------------------------------
# Window extraction and aggregation
# ---------------------------------------------------------------------------
def get_window_values(feature_series, w_start, w_end):
    window = []
    n = len(feature_series)
    for i in range(w_start, w_end + 1):
        window.append(float("nan") if (i < 0 or i >= n) else feature_series[i])
    return window


def aggregate_window(window, latest_value):
    valid_values = [v for v in window if not math.isnan(v)]
    if len(valid_values) == 0:
        mean_val = 0.0
        var_val = 0.0
    else:
        mean_val = sum(valid_values) / len(valid_values)
        var_val = sum((v - mean_val) ** 2 for v in valid_values) / len(valid_values)
    latest_val = latest_value if not math.isnan(latest_value) else 0.0
    return mean_val, var_val, latest_val


# ---------------------------------------------------------------------------
# Feature vector and usability
# ---------------------------------------------------------------------------
def extract_features(sample, tte_seconds=2.0, window_seconds=0.5):
    """Return the 30-dim feature vector (10 base features x mean/var/latest)."""
    fps = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
    label = str(sample.get("label", "")).lower()
    k = max(2, int(round(window_seconds * fps)))

    door = sample.get("door_center")
    if door is not None and len(door) == 2:
        door_x, door_y = float(door[0]), float(door[1])
    else:
        door_x = float(sample.get("frame_width", 640)) / 2.0
        door_y = float(sample.get("frame_height", 480)) - 1.0

    frames = sample.get("frames", [])
    if len(frames) == 0:
        return [0.0] * 30

    x, y, x1, y1, x2, y2 = get_coordinates(frames)
    n = len(x)

    t_event = find_event_frame(label, x, y, door_x, door_y)
    if t_event < 0:
        return [0.0] * 30

    tte_frames = int(round(tte_seconds * fps))
    t_predict = t_event - tte_frames
    t_predict = max(0, min(t_predict, n - 1))

    w_start = t_predict - k + 1
    w_end = t_predict

    d_door = compute_distance_to_door(x, y, door_x, door_y)
    closure_rate = compute_closure_rate(d_door, fps, k)
    vx, vy = compute_velocity(x, y, fps, k)
    abs_speed = compute_absolute_speed(vx, vy)
    step_disp = compute_step_displacement(x, y)
    heading, rel_angle = compute_heading_and_relative_angle(vx, vy, x, y, door_x, door_y)
    aspect_ratio, scale_change = compute_bbox_features(x1, y1, x2, y2, fps, k)

    all_features = [
        d_door, closure_rate, vx, vy, abs_speed,
        step_disp, heading, rel_angle, aspect_ratio, scale_change,
    ]

    feature_vector = []
    for series in all_features:
        window = get_window_values(series, w_start, w_end)
        latest = series[t_predict] if 0 <= t_predict < len(series) else float("nan")
        m, v, l = aggregate_window(window, latest)
        feature_vector.extend([m, v, l])
    return feature_vector


def is_sample_usable(sample, tte_seconds, window_seconds=0.5,
                     allowed_labels=("enter", "pass")):
    """
    True if the clip is long enough that T_predict = T_event - tte fits with a
    full observation window (so the features describe the intended moment).
    """
    fps = float(sample.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
    frames = sample.get("frames", [])
    if not frames:
        return False
    label = str(sample.get("label", "")).lower()
    if label not in allowed_labels:
        return False
    k = max(2, int(round(window_seconds * fps)))
    tte_frames = int(round(tte_seconds * fps))

    door = sample.get("door_center")
    if door is not None and len(door) == 2:
        door_x, door_y = float(door[0]), float(door[1])
    else:
        door_x = float(sample.get("frame_width", 640)) / 2.0
        door_y = float(sample.get("frame_height", 480)) - 1.0

    x, y, *_ = get_coordinates(frames)
    t_event = find_event_frame(label, x, y, door_x, door_y)
    if t_event < 0:
        return False
    t_predict = t_event - tte_frames
    return t_predict >= k - 1


def get_feature_names():
    """The 30 column names, in the order extract_features returns them."""
    base_features = [
        "dist_to_door", "closure_rate", "vx", "vy", "abs_speed",
        "step_displacement", "heading_angle", "rel_angle_to_door",
        "aspect_ratio", "scale_change_rate",
    ]
    stats = ["mean", "var", "latest"]
    names = []
    for feature in base_features:
        for stat in stats:
            names.append(f"{feature}_{stat}")
    return names


# ---------------------------------------------------------------------------
# Dataset walk + CSV
# ---------------------------------------------------------------------------
def find_clip_jsons(data_root: Path, labels) -> list[tuple[Path, str]]:
    """
    Return (json_path, label) for every kept clip under data_root.

    Looks for folders named after the labels (enter/pass/exit) anywhere below
    data_root, so it works whether data_root is the data/ folder or a single
    dated folder. Pose sidecars (*_pose.json) are skipped.
    """
    out: list[tuple[Path, str]] = []
    for label in labels:
        for d in sorted(data_root.rglob(label)):
            if not d.is_dir():
                continue
            for j in sorted(d.glob("*.json")):
                if j.stem.endswith("_pose"):
                    continue
                out.append((j, label))
    return out


def load_sample(json_path: Path) -> Optional[dict]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_table(data_root: Path, out_csv: Path, tte_list, window_seconds, labels):
    feature_names = get_feature_names()
    header = (["source", "label", "fps", "n_frames", "tte_seconds", "window_seconds"]
              + feature_names)

    clips = find_clip_jsons(data_root, labels)
    n_rows = 0
    n_clips_used = 0
    per_label = {lab: 0 for lab in labels}
    skipped = 0

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for json_path, folder_label in clips:
            sample = load_sample(json_path)
            if sample is None:
                skipped += 1
                continue
            label = str(sample.get("label", folder_label)).lower()
            fps = sample.get("fps")
            n_frames = sample.get("n_frames", len(sample.get("frames", [])))
            used_here = False
            for tte in tte_list:
                if not is_sample_usable(sample, tte, window_seconds, tuple(labels)):
                    continue
                vec = extract_features(sample, tte_seconds=tte, window_seconds=window_seconds)
                writer.writerow(
                    [json_path.name, label, fps, n_frames, tte, window_seconds]
                    + [round(v, 6) for v in vec]
                )
                n_rows += 1
                used_here = True
            if used_here:
                n_clips_used += 1
                per_label[label] = per_label.get(label, 0) + 1
            else:
                skipped += 1

    return {
        "clips_found": len(clips),
        "clips_used": n_clips_used,
        "rows": n_rows,
        "skipped": skipped,
        "per_label": per_label,
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Build the handcrafted feature table from collected clips.")
    p.add_argument("--data-root", type=Path, default=here.parent / "data",
                   help="Folder with the collected clips (default: ../data).")
    p.add_argument("--out", type=Path, default=here / "features.csv",
                   help="Output CSV path (default: features.csv next to this script).")
    p.add_argument("--tte-seconds", type=float, nargs="+", default=[2.0],
                   help="One or more time-to-event horizons in seconds (default: 2.0).")
    p.add_argument("--window-seconds", type=float, default=0.5,
                   help="Observation window length in seconds (default: 0.5).")
    p.add_argument("--include-exit", action="store_true",
                   help="Also extract features for exit clips (event = last visible frame).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    labels = ["enter", "pass"] + (["exit"] if args.include_exit else [])

    if not args.data_root.exists():
        print(f"Data folder not found: {args.data_root}")
        return 2

    print(f"[FEATURES] data-root : {args.data_root}")
    print(f"[FEATURES] labels    : {labels}")
    print(f"[FEATURES] tte (s)   : {args.tte_seconds}   window (s): {args.window_seconds}")

    stats = build_table(args.data_root, args.out, args.tte_seconds, args.window_seconds, labels)

    print(f"[FEATURES] clips found : {stats['clips_found']}")
    print(f"[FEATURES] clips used  : {stats['clips_used']}  {stats['per_label']}")
    print(f"[FEATURES] rows written: {stats['rows']}  (one per usable clip x tte)")
    print(f"[FEATURES] skipped     : {stats['skipped']}")
    print(f"[FEATURES] wrote       : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
