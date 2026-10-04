#!/usr/bin/env python3
"""Dialogue audio -> face animation curves with the studio model.

    python predict.py model.pt dialogue.wav out.json --fps 24

out.json: {"fps": 24, "names": [52 ARKit names], "frames": [[52 values], ...]}
The Blender add-on (Lip-Sync Rig, engine "Studio AI model") runs this for
you and keys the values onto the character's shape keys.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def predict(model_path, audio_path, fps):
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    net = C.model(len(ck["names"]))
    net.load_state_dict(ck["state"])
    net.eval()
    mel = C.log_mel(C.load_audio(audio_path))
    x = torch.tensor((mel - ck["norm_mean"]) / ck["norm_std"], dtype=torch.float32)[None]
    with torch.no_grad():
        y = net(x)[0].numpy()
    n = int(len(mel) / 100 * fps)
    return ck["names"], C.sample_at(y, fps, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=24)
    a = ap.parse_args()
    names, frames = predict(a.model, a.audio, a.fps)
    Path(a.out).write_text(json.dumps({"fps": a.fps, "names": names,
                                       "frames": np.round(frames, 4).tolist()}))
    print(f"{len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()
