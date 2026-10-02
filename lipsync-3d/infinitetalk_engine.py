"""InfiniteTalk engine: video-to-video dubbing with MeiGen-AI/InfiniteTalk
(Apache-2.0, built on Wan2.1-I2V-14B).

Unlike LatentSync, which repaints only the mouth, InfiniteTalk regenerates
the whole clip it is given - lips, jaw, head and expression follow the
audio - and only roughly follows the original camera. Here it is given the
speaker crop, and app.blend() keeps as much of its output as the mask asks
for ("whole face" suits this engine best).

It lives in its own checkout and Python environment (its torch/flash-attn
versions differ from LatentSync's); deploy/infinitetalk_setup.sh installs
both. Paths come from the environment:
    INFINITETALK_DIR     checkout with weights/   (default /workspace/InfiniteTalk)
    INFINITETALK_PYTHON  its python               (default <dir>/.venv/bin/python)
    INFINITETALK_LOWVRAM 1 = --num_persistent_param_in_dit 0 (24 GB GPUs; slower)
"""

import json
import os
import subprocess
import time
from pathlib import Path

DIR = Path(os.environ.get("INFINITETALK_DIR", "/workspace/InfiniteTalk"))
PY = os.environ.get("INFINITETALK_PYTHON", str(DIR / ".venv" / "bin" / "python"))
W = {
    "ckpt": DIR / "weights" / "Wan2.1-I2V-14B-480P",
    "wav2vec": DIR / "weights" / "chinese-wav2vec2-base",
    "talk": DIR / "weights" / "InfiniteTalk" / "single" / "infinitetalk.safetensors",
    # optional: 8-step distillation LoRA, ~5x faster
    "lora": DIR / "weights" / "Wan2.1_I2V_14B_FusionX_LoRA.safetensors",
}


def problems():
    """What is missing for this engine, in plain words ([] = ready)."""
    out = []
    if not (DIR / "generate_infinitetalk.py").exists():
        out.append(f"InfiniteTalk is not installed in {DIR} - run "
                   f"lipsync-3d/webapp/deploy/infinitetalk_setup.sh")
        return out
    if not Path(PY).exists():
        out.append(f"InfiniteTalk's python not found: {PY}")
    for k in ("ckpt", "wav2vec", "talk"):
        if not W[k].exists():
            out.append(f"InfiniteTalk weights missing: {W[k]}")
    return out


def run(crop_in, wav, crop_out, seed, log, size="infinitetalk-480", quality="fast"):
    """Dub crop_in to wav; writes crop_out (25 fps, its own resolution -
    blend() scales it back onto the crop)."""
    gone = problems()
    if gone:
        raise RuntimeError(" | ".join(gone))
    crop_out = Path(crop_out)
    work = crop_out.parent
    job = work / "infinitetalk.json"
    job.write_text(json.dumps({
        "prompt": "A 3D animated character is talking, natural mouth movement, "
                  "same face, same lighting, steady camera",
        "cond_video": str(Path(crop_in).resolve()),
        "cond_audio": {"person1": str(Path(wav).resolve())}}))
    save = work / "infinitetalk_out"
    # its ffmpeg audio-crop step has no -y: a leftover file would stall it
    for f in work.glob("infinitetalk_out*"):
        f.unlink()
    cmd = [PY, "generate_infinitetalk.py",
           "--ckpt_dir", str(W["ckpt"]), "--wav2vec_dir", str(W["wav2vec"]),
           "--infinitetalk_dir", str(W["talk"]), "--input_json", str(job),
           "--size", size, "--mode", "streaming", "--motion_frame", "9",
           "--base_seed", str(int(seed)), "--save_file", str(save)]
    if quality == "fast" and W["lora"].exists():
        cmd += ["--lora_dir", str(W["lora"]), "--lora_scale", "1.0",
                "--sample_text_guide_scale", "1.0", "--sample_audio_guide_scale", "2.0",
                "--sample_steps", "8", "--sample_shift", "2"]
    else:
        # README: lip sync is best with audio guidance 3-5 without a LoRA
        cmd += ["--sample_steps", "40", "--sample_audio_guide_scale", "4.0"]
    if os.environ.get("INFINITETALK_LOWVRAM", "") == "1":
        cmd += ["--num_persistent_param_in_dit", "0"]
    log("InfiniteTalk: loading the 14B model and generating (this takes minutes)...")
    t = time.time()
    tail = []
    with subprocess.Popen(cmd, cwd=DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace") as p:
        for line in p.stdout:
            line = line.rstrip()
            print(f"[infinitetalk] {line}", flush=True)
            tail = (tail + [line])[-15:]
    if p.returncode:
        if any("out of memory" in x.lower() for x in tail):
            raise RuntimeError("InfiniteTalk ran out of GPU memory - set INFINITETALK_LOWVRAM=1 "
                               "and restart, or use a bigger GPU (A100/H100 80 GB)")
        raise RuntimeError("InfiniteTalk failed: " + " / ".join(tail[-4:])[-500:])
    made = next((f for f in (save.with_suffix(".mp4"), Path(str(save) + ".mp4"), save)
                 if f.exists()), None)
    if made is None:
        cands = sorted(work.glob("infinitetalk_out*.mp4"))
        made = cands[-1] if cands else None
    if made is None:
        raise RuntimeError("InfiniteTalk finished but wrote no video")
    made.replace(crop_out)
    log(f"InfiniteTalk: {time.time() - t:.0f}s")
