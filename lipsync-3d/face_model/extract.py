#!/usr/bin/env python3
"""Training data from lip-synced videos: audio + the face's 52 ARKit
blendshape values on every frame, read by MediaPipe's face landmarker.

    python extract.py VIDEOS_DIR DATA_DIR [--model face_landmarker.task]

Good sources: sync.so results you already paid for, finished renders whose
lip sync was approved. One speaking face per clip works best (the largest
face is used). Clips already extracted are skipped.
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def probe(path):
    s = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), int(n) / int(d)


def face_track(path, landmarker_path, max_side=720):
    """(frames, 52) blendshapes and (frames,) bool mask of frames with a face."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    w, h, fps = probe(path)
    k = min(1.0, max_side / max(w, h))
    W, H = int(w * k) // 2 * 2, int(h * k) // 2 * 2
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(landmarker_path)),
        running_mode=vision.RunningMode.VIDEO, num_faces=2,
        output_face_blendshapes=True)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                          "-vf", f"scale={W}:{H}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    out, mask = [], []
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            b = p.stdout.read(W * H * 3)
            if len(b) < W * H * 3:
                break
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.frombuffer(b, np.uint8).reshape(H, W, 3))
            r = lm.detect_for_video(img, int(i * 1000 / fps))
            if r.face_blendshapes:
                # biggest face = the one the clip is about
                sizes = [np.ptp([q.x for q in f]) * np.ptp([q.y for q in f])
                         for f in r.face_landmarks]
                bs = r.face_blendshapes[int(np.argmax(sizes))]
                d = {c.category_name: c.score for c in bs}
                out.append([d.get(n, 0.0) for n in C.ARKIT])
                mask.append(True)
            else:
                out.append([0.0] * len(C.ARKIT))
                mask.append(False)
            i += 1
    p.stdout.close()
    p.wait()
    return np.asarray(out, np.float32), np.asarray(mask), fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos")
    ap.add_argument("out")
    ap.add_argument("--model", default=None, help="face_landmarker.task (downloaded if missing)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lmk = Path(a.model) if a.model else out / "face_landmarker.task"
    if not lmk.exists():
        print("downloading MediaPipe face landmarker...")
        urllib.request.urlretrieve(MODEL_URL, lmk)
    src = Path(a.videos)
    vids = [src] if src.is_file() else sorted(
        p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    print(f"{len(vids)} video(s)")
    for v in vids:
        dst = out / (v.stem + ".npz")
        if dst.exists():
            continue
        try:
            bs, mask, fps = face_track(v, lmk)
            mel = C.log_mel(C.load_audio(v))
        except Exception as e:
            print(f"  skip {v.name}: {e}")
            continue
        if mask.mean() < 0.3:
            print(f"  skip {v.name}: face found on only {mask.mean():.0%} of frames")
            continue
        np.savez_compressed(dst, mel=mel, bs=bs, mask=mask, fps=fps, src=str(v))
        print(f"  {v.name}: {len(bs)} frames @ {fps:.2f} fps, face on {mask.mean():.0%}, "
              f"{len(mel) / 100:.1f}s audio")


if __name__ == "__main__":
    main()
