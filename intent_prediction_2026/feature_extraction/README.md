# feature_extraction

`extract_features.py` turns a folder of collected clips (`../data`) into a
handcrafted feature table (CSV) for the logistic-regression / random-forest
models. It is self-contained (standard library only) and does not modify the
clips.

## What it computes

For each kept clip it builds the 30-dimensional feature vector: 10 motion
features, each summarised by its mean, variance and latest value over a 0.5 s
observation window ending `tte_seconds` before the event (closest approach to
the door for `enter`, last visible frame for `pass`). One CSV row is written per
(clip, tte horizon).

The 10 base features: distance to door, closure rate, vx, vy, absolute speed,
step displacement, heading angle, relative angle to door, bounding-box aspect
ratio, bounding-box scale-change rate.

## Run

```bash
python extract_features.py                                   # ../data, tte=2.0s, window=0.5s
python extract_features.py --data-root ../data --out features.csv
python extract_features.py --tte-seconds 1.0 1.5 2.0 2.5 3.0  # several horizons in one run
python extract_features.py --include-exit                    # also emit exit clips
```

Options: `--data-root`, `--out`, `--tte-seconds` (one or more), `--window-seconds`,
`--include-exit`.

## Output

A CSV with columns:

```
source, label, fps, n_frames, tte_seconds, window_seconds, <30 feature columns>
```

The feature columns are named `<feature>_<mean|var|latest>`. Clips too short to
fit the chosen `tte` + window are skipped; `removed/` clips and `_pose.json`
sidecars are ignored.

By default only `enter` and `pass` clips are included (the two-class task). Use
`--include-exit` to also extract `exit` clips, whose event is taken as the last
visible frame.
