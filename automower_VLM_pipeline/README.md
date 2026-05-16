# automower_VLM_pipeline

VLM-based obstacle detection pipeline for autonomous robotic lawnmowers.

## Folders

### `data_collection/`

Contains the camera server that runs on the Raspberry Pi mounted to the lawnmower. It streams the Pi camera live and lets you record video directly from a browser over the local network. See its README for setup instructions.

### `evaluation/`

Contains the full detection pipeline. Two scripts are available depending on whether you want VLM involvement:

- `pipeline_seg_only.py` runs segmentation-only obstacle detection on a video
- `pipeline_seg_vlm.py` does the same but forwards flagged frames to a VLM via Ollama

All pipeline parameters (model, thresholds, frame rate, Ollama endpoint) are set in `config.py`. 

A browser-based frame viewer is included under `viewer/` for inspecting results run by run. 

See the folder README for full usage instructions.
