# RISE Open-House Demo — Pedestrian Intent Prediction

Real-time prediction of whether a person approaching a door will **enter** or **pass by**, running on an NVIDIA Jetson Nano with a single camera.

## Architecture

```
Camera → YOLOv8n-pose (BoT-SORT tracking)
              │
              ├─ Bounding box → dist_to_door, closure_rate, rel_angle_to_door
              ├─ Ear keypoints → left_ear_x/y, right_ear_x/y
              │
              └─ Core-3+head GRU (7 features, hidden=16) → P(enter)
                      │
                      └─ Decision logic → DOOR OPEN / CLOSED
```

## Setup on Jetson Nano

```bash
# 1. Install dependencies
pip install ultralytics torch torchvision numpy opencv-python

# 2. Download YOLO-Pose model (auto-downloads on first run)
#    Or pre-download: yolov8n-pose.pt

# 3. (Optional) Export to TensorRT for better FPS
yolo export model=yolov8n-pose.pt format=engine half=True device=0

# 4. Run demo (no trained model — shows features + skeleton)
python demo_live.py

# 5. Run with trained GRU model
python demo_live.py --model weights/gru_core3head.pt --norm weights/norm_stats.npz
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Camera index or video file path |
| `--yolo` | `yolov8n-pose.pt` | YOLO model (use .engine for TensorRT) |
| `--model` | None | Trained GRU weights (.pt file) |
| `--norm` | None | Normalisation stats (.npz file) |
| `--door-x` | frame center | Door X position in pixels |
| `--door-y` | frame bottom | Door Y position in pixels |
| `--threshold` | 0.75 | P(enter) threshold for opening |
| `--confirm` | 5 | Consecutive frames above threshold |
| `--fullscreen` | off | Fullscreen display |
| `--no-features` | off | Hide feature panel |
| `--no-skeleton` | off | Hide pose skeleton |
| `--no-log` | off | Disable CSV logging |
| `--log-dir` | `logs/` | Directory for log CSV files |

## Output Logs

Two CSV files are generated in `logs/`:

- **`latency_YYYYMMDD_HHMMSS.csv`** — Per-frame timing breakdown (for RQ2/RQ3):
  - `dt_yolo_ms`, `dt_features_ms`, `dt_gru_ms`, `dt_viz_ms`, `dt_total_ms`
  - `n_persons`, `fps_measured`

- **`features_YYYYMMDD_HHMMSS.csv`** — Per-person per-frame data (no video needed):
  - `track_id`, bounding box, all 7 features, `p_enter`, `door_decision`

## Keyboard Controls

- **q** — Quit
- **f** — Toggle fullscreen

## Training a GRU Model for the Demo

The demo can run without a trained model (shows P=0.50 for everyone).
To train and export a model for use with the demo, you need to:

1. Save the best model's `state_dict()` as a `.pt` file
2. Save the training set's feature mean/std as a `.npz` file:
   ```python
   np.savez("norm_stats.npz", mean=train_mean, std=train_std)
   ```

## Files

```
demo/
├── demo_live.py         # Main entry point — live camera demo
├── features.py          # Feature computation (Core-3 + head raw)
├── model.py             # GRU model definition + loading
├── train_demo_model.py  # (re)train the demo GRU from MasterData
├── test_camera.py       # quick camera sanity check
├── run.txt              # the command to launch the demo
├── weights/             # small trained model, included so the demo runs as-is
│   ├── gru_core3head.pt
│   └── norm_stats.npz
└── README.md
```

The trained demo weights are included, so step 5 above works out of the box.
The YOLO-Pose weights are not — they download automatically on first run.
