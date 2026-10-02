#!/usr/bin/env python3
"""Local lip-sync worker: runs LatentSync on a list of jobs, loading it once.

    python lipsync-3d/latentsync_worker.py --jobs jobs.json \
        --config configs/unet/stage2.yaml --ckpt checkpoints/latentsync_unet.pt

run.py starts this for you. It is a separate process on purpose:
LatentSync pins numpy 1.26 / torch 2.5 while run.py needs numpy 2, so the
two usually live in different environments (--ls-python in run.py).

It must run with the repository root as its working directory - LatentSync
reads "configs/", "checkpoints/" and writes "temp/" relative to it.

jobs.json is a list of
    {"id", "video", "audio", "out", "steps", "guidance", "seed"}
Each finished job writes <out>.json:
    {"ok", "error", "seconds", "peak_vram_gb", "settings"}
written as soon as that job ends, so a crash at job 8 keeps jobs 1-7.
Nothing here touches the network; the checkpoints must already be on disk.
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path


def load(config_path, ckpt, deepcache):
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    if not torch.cuda.is_available():
        sys.exit("LatentSync needs a CUDA GPU (torch.cuda.is_available() is False)")
    config = OmegaConf.load(config_path)
    fp16 = torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if fp16 else torch.float32
    whisper = {768: "checkpoints/whisper/small.pt",
               384: "checkpoints/whisper/tiny.pt"}[config.model.cross_attention_dim]
    for f in (ckpt, whisper):
        if not Path(f).exists():
            sys.exit(f"missing checkpoint {f} - download it during setup "
                     f"(see lipsync-3d/README.md), not during processing")

    audio_encoder = Audio2Feature(model_path=whisper, device="cuda",
                                  num_frames=config.data.num_frames,
                                  audio_feat_length=config.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), ckpt, device="cpu")
    pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder,
                           unet=unet.to(dtype=dtype),
                           scheduler=DDIMScheduler.from_pretrained("configs")).to("cuda")
    # decode 16 frames one at a time: a little slower, much less VRAM
    pipe.enable_vae_slicing()
    if deepcache:
        from DeepCache import DeepCacheSDHelper
        h = DeepCacheSDHelper(pipe=pipe)
        h.set_params(cache_interval=3, cache_branch_id=0)
        h.enable()
    return pipe, config, dtype


def run_job(pipe, config, dtype, job, io):
    import torch
    from accelerate.utils import set_seed

    # LatentSync builds its ffmpeg commands with shell=True and unquoted
    # paths, so a space anywhere in a path breaks it. Work on safe copies.
    io.mkdir(parents=True, exist_ok=True)
    v, a, o = io / f"{job['id']}.mp4", io / f"{job['id']}.wav", io / f"{job['id']}_out.mp4"
    shutil.copyfile(job["video"], v)
    shutil.copyfile(job["audio"], a)
    o.unlink(missing_ok=True)
    set_seed(int(job["seed"]))
    torch.cuda.reset_peak_memory_stats()
    pipe(video_path=str(v), audio_path=str(a), video_out_path=str(o),
         num_frames=config.data.num_frames,
         num_inference_steps=int(job["steps"]),
         guidance_scale=float(job["guidance"]),
         weight_dtype=dtype,
         width=config.data.resolution, height=config.data.resolution,
         mask_image_path=config.data.mask_image_path,
         temp_dir="temp")
    if not o.exists() or o.stat().st_size == 0:
        raise RuntimeError("LatentSync wrote no output")
    shutil.move(str(o), job["out"])
    return torch.cuda.max_memory_allocated() / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--config", default="configs/unet/stage2.yaml")
    ap.add_argument("--ckpt", default="checkpoints/latentsync_unet.pt")
    ap.add_argument("--deepcache", action="store_true")
    a = ap.parse_args()

    import torch

    jobs = json.loads(Path(a.jobs).read_text())
    t = time.time()
    pipe, config, dtype = load(a.config, a.ckpt, a.deepcache)
    print(f"[worker] LatentSync loaded in {time.time()-t:.0f}s on "
          f"{torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB), "
          f"{config.data.resolution}px, {len(jobs)} jobs", flush=True)
    io = Path("temp_ls_io")
    for job in jobs:
        t = time.time()
        res = {"settings": {k: job[k] for k in ("steps", "guidance", "seed")}}
        try:
            res["peak_vram_gb"] = round(run_job(pipe, config, dtype, job, io), 2)
            res.update(ok=True, error=None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            res.update(ok=False, error="CUDA out of memory - use the 256px model "
                       "(configs/unet/stage2.yaml + LatentSync 1.5 checkpoint), "
                       "fewer steps, or a smaller --crop-max")
        except Exception as e:
            res.update(ok=False, error=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        res["seconds"] = round(time.time() - t, 1)
        Path(job["out"] + ".json").write_text(json.dumps(res, indent=1))
        print(f"[worker] {job['id']} {'OK' if res['ok'] else 'FAIL ' + res['error']} "
              f"{res['seconds']}s vram={res.get('peak_vram_gb')}GB", flush=True)
    shutil.rmtree(io, ignore_errors=True)


if __name__ == "__main__":
    main()
