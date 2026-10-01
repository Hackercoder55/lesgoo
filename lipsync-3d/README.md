# Blender lip-sync automation

Lip-syncs only the speaking parts of 3D animated shorts through
[sync.so](https://sync.so), instead of sending the whole video. The plan
is built from the footage itself (face on screen + speech in the audio),
each segment is synced, composited back over the original, spliced in
frame-exactly, and cross-checked against the master.

Typical saving against sending the full video is 70-90% of credits.
Measured rate on `sync-3` is 0.5333 credits per frame.

## Use

```
python run.py source/video4.mp4          # plan and cost only - spends nothing
python run.py source/video4.mp4 --go     # generate, composite, splice, verify
```

Tuning: `--min-face` (smallest face to sync, % of frame area, default 2),
`--min-run`, `--merge-gap`.

It never retries a failed segment by itself. A segment that fails is
reported and the original is left in place.

## Setup

1. Python 3.11+ and ffmpeg/ffprobe on `PATH`.
2. `pip install -r requirements.txt`
3. Put the face-detection model in `models/`:
   ```
   curl -L -o models/yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
   ```
4. Create `.env` with your key (never commit it - it is gitignored):
   ```
   SYNC_API_KEY=your-key-here
   ```

## Status - read this before trusting it

- `run.py` plan-only mode is tested (on video 3 it found 33 segments,
  531 credits).
- `run.py --go` has **not** been run end to end. Every stage in it has run
  successfully, but as separate per-video scripts (`batch*.py`,
  `composite*.py`, `splice*.py`, `verify*.py`). Run it on a small video
  first and read the verify output.
- Lip quality is sync.so's, not this code's. What this code guarantees is
  that it checks its own output and does not silently damage the original.

## Files

| | |
|---|---|
| `run.py` | the one-command pipeline |
| `lipsync.py` | sync.so client: upload, submit, poll, output validation |
| `plan.py`, `plan2c.py`, `speakers3.py`, `scan3.py`, `refine.py` | segment planning, per video |
| `batch*.py` | extract and generate per video |
| `composite*.py` | put only the speaker's mouth back, level-matched |
| `splice*.py` | rebuild the full video |
| `verify*.py`, `align.py`, `seamcheck.py` | cross-checks against the master |

Everything except `run.py` and `lipsync.py` is the per-video working code
from building the tool: paths are hardcoded to `video2`/`video3`. The planning
scripts that embedded a client's dialogue are deliberately not in this repo;
`run.py` replaces them by deriving the plan from the footage.

## Hard-won rules (each one shipped a defect once)

- Decide what to sync from the footage, never from the transcript's
  grammar. Narration looked like voice-over on two videos and was mouthed
  on screen in the third.
- Name the speaker to sync.so explicitly. Left alone it syncs the most
  prominent face, which was a bystander twice - and the compositor must
  follow the same face.
- Never widen a segment into another voice's audio.
- Work in `yuv444p`; never pass `-color_trc` to the encoder (it converts
  pixels); label the finished file with the `h264_metadata` bitstream
  filter; use `setparams` inside the filter graph.
- Use the concat *filter*, rebuild timestamps with `setpts=N/fps/TB`, and
  output with `-fps_mode passthrough`, not CFR.
- sync.so returns frames darker than it was given and with wrong timing
  metadata. Composite over the original and rebuild timing.
- Measure on the raw Y plane with passthrough decoding; RGB and CFR both
  manufacture differences that are not in the file.
- A check that has never failed may be checking nothing. Prove each
  validator against a known-bad case.

The speaker-selection option is `snake_case` in the real API
(`active_speaker_detection`: `auto_detect`, `frame_number`, `coordinates`,
optional `face_image`); the published docs show camelCase, which is rejected.
