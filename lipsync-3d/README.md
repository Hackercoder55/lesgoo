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
`--min-run`, `--merge-gap`, `--face-score` (detector confidence, default
0.6; lower it for stylised faces), `--language` (`en` by default, `hi`,
or `auto`), `--whisper-model` (`small` by default; `medium` or `large-v3`
miss fewer words).

The plan output ends with **speech NOT synced**: every spoken stretch the
plan leaves out, with the reason (no face detected, face too small, too
short). It is also written to `work_<video>/missed.json`. Read it before
`--go`; the reason says which flag to change.

Cached analysis is keyed by its settings (`words_<model>_<lang>.json`,
`scan_<size>_<score>.npz`), so changing a flag re-runs that stage.

It never retries a failed segment by itself. A segment that fails is
reported and the original is left in place. To resubmit only the failed
ones:

```
python run.py source/video4.mp4 --go --retry
```

Segments that already synced and still match the plan are reused, not
paid for again. Without `--retry` a failed segment goes back under the
same idempotency key, and sync.so returns the same failed generation.

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

## Why it skipped dialogue (fixed)

- The planner dropped runs shorter than `--min-run` **before** joining
  them. Whisper leaves small gaps between words and the face detector
  misses odd frames, so a normal sentence arrived as 5-10 frame pieces and
  most of it was thrown away. It now joins first and drops only what is
  still short. On a synthetic 4 s sentence: before 14 frames synced, after 94.
- Faces were detected on a fixed 540x960 frame. That is right for 9:16
  shorts but squashes a 16:9 frame, and the detector then misses faces.
  The detection size now follows the video's aspect ratio.
- The frame rate was rounded (29.97 -> 30), which drifts the spliced
  picture against the audio by about a frame every 33 s. The exact rate
  is now used.

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
