# Evaluation

This folder contains the full pipeline, tooling, and results for the thesis evaluation of a VLM-based obstacle detection system for autonomous robotic lawnmowers.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Edit `config.py` to set the Ollama endpoint, VLM model, and any pipeline parameters, then run a pipeline:

```bash
python pipeline_seg_only.py
python pipeline_seg_vlm.py
```

Both scripts prompt for input at runtime, no arguments are needed.

## Pipelines

- `pipeline_seg_only.py`: runs a segmentation model on a video and logs obstacle detections per frame, without calling a VLM
- `pipeline_seg_vlm.py`: same segmentation front-end, but obstacle frames are forwarded to a VLM via Ollama

## Config and utilities

- `config.py`: all pipeline parameters: VLM model, Ollama endpoint, frame sampling rate, zone thresholds, blur filter, cooldown settings
- `seg_utils.py`: shared segmentation helpers used by both pipelines (model loading, zone detection, blur check, frame annotation)

## Tools

A browser-based tool for inspecting and reviewing pipeline output. It is a Flask app started with `python app.py`.

- `viewer/` (port 5050): frame-by-frame viewer for any pipeline run, showing segmentation and VLM results side by side
