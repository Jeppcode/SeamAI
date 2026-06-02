# intent_prediction_2026

The **v2, pose-based** pipeline added by Jesper Malmgren's master's thesis (2026),
kept deliberately separate from the original `automatic_door/automatic_labeling/` scripts so the
existing code is untouched.

## What changed vs. the original (short version)

Compared to `automatic_door/automatic_labeling/data_collection.py` (Norberg), the headline
differences are:

1. **One pose model does everything.** A single `yolo11n-pose` pass yields the
   bounding box **and** the 17 COCO body keypoints per person — instead of
   detecting only the bounding box and adding a skeleton afterward with a
   separate MediaPipe step.
2. **`exit` is a real class.** Exit behavior (people walking *away* from the
   door) is detected and labeled at collection time, alongside `enter` and
   `pass`.

(Plus: clips are pre-sorted into `enter/pass/exit/removed` and nothing is
deleted, robust handling of people leaving and re-entering the frame, the real
measured fps and per-frame timestamps, and an optional `_pose.json` sidecar with
the skeleton as normalized named landmarks. See `data_collection/README.md` for
the full list and schema.)

## Structure

```
intent_prediction_2026/
  requirements.txt        # Python dependencies for the virtual environment (see Setup)
  data_collection/        # START HERE — the live collection script + its README
    pose_data_collection.py
    README.md
  data/                   # the collection script writes the dataset here (default output)
  feature_extraction/     # builds the handcrafted 30-dim feature table (CSV) from data/
    extract_features.py
    README.md
```

The collection script is **self-contained** (it imports no other project code)
and writes its dataset to `data/` by default.

## Setup

Create a virtual environment in this folder and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the Jetson, install PyTorch from NVIDIA's Jetson wheels **before** this step —
see the notes at the top of `requirements.txt`. The dependencies are only for
`data_collection/`; `feature_extraction/` runs on the standard library alone.

## Privacy

These scripts record identifiable video and pose data of people; handle it accordingly.
