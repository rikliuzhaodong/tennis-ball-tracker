# Tennis Ball Tracker

Detect a tennis ball in video, track it through short occlusions, and render a
fading trajectory trail while preserving the original audio.

The default detector is TrackNet V1, a three-frame neural network designed for
fast, tiny tennis balls. A classical colour/motion detector is also included as
a lightweight fallback.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 download_model.py
```

The downloaded weights are stored in `models/tracknet_weights.pth` and are not
committed to Git.

## Run

```bash
python3 track_tennis_ball.py input.mp4 outputs/tracked.mp4 \
  --debug-csv outputs/trajectory.csv \
  --preview outputs/preview.jpg \
  --trail 24 \
  --compile-model
```

Useful options:

- `--trail 24`: number of frames in the fading trail.
- `--analyze-only`: run detection and save CSV/preview without encoding video.
- `--trajectory-in path.csv`: reuse a detected trajectory without re-running
  the model.
- `--detector classical`: use the no-model colour/motion fallback.

The output is H.264 MP4 with the source audio remuxed as AAC. Detection uses a
short-gap interpolation policy: gaps of up to four frames are filled, while
longer uncertain sections are left blank instead of inventing a trajectory.

## Validated sample

The pipeline has been tested end to end on a 12.03-second indoor tennis clip:

- Source: 3840 x 2160, 30 fps, HEVC with AAC audio
- Output: 3840 x 2160, 30 fps, H.264 with AAC audio
- TrackNet detections/interpolations: 154 of 361 frames
- Visual result: highlighted ball position plus a 24-frame fading trail

Generated videos, previews, trajectory CSV files, and downloaded model weights
are intentionally ignored by Git. This keeps the repository lightweight while
allowing every artifact to be reproduced locally.

## Project structure

```text
track_tennis_ball.py  Detection, filtering, trail rendering, and video encoding
tracknet_model.py     TrackNet V1 neural-network architecture
download_model.py     Downloads the public pretrained tennis weights
requirements.txt      Python runtime dependencies
```

## Model attribution

- [TrackNet paper](https://arxiv.org/abs/1907.03698)
- [PyTorch reference implementation](https://github.com/yastrebksv/TrackNet)
- [MIT-licensed tennis weights](https://huggingface.co/vishnushenoy09/tracknet-v1-tennis)
