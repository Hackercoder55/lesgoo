#!/usr/bin/env python3
"""Train the studio's audio -> face model on extract.py's data.

    python train.py DATA_DIR model.pt [--epochs 60] [--val 0.1]

Clips are split into training and held-out validation by file, so the
validation score shows how it does on dialogue it has never heard. The best
model (lowest validation mouth error) is kept. Runs on a GPU when there is
one; a few hours of data trains in well under an hour on an RTX 4090.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def load(paths):
    clips = []
    for p in paths:
        d = np.load(p)
        mel, bs, mask, fps = d["mel"], d["bs"], d["mask"], float(d["fps"])
        # targets at 100 Hz to match the audio features
        t = np.arange(len(mel)) / 100 * fps
        fi = np.clip(np.round(t).astype(int), 0, len(bs) - 1)
        clips.append((mel, bs[fi], mask[fi] & (t < len(bs)), p.name))
    return clips


def batches(clips, n, seconds=4.0):
    T = int(seconds * 100)
    for _ in range(n):
        xs, ys, ms = [], [], []
        for _ in range(16):
            mel, y, m, _ = random.choice(clips)
            if len(mel) <= T:
                s = 0
            else:
                s = random.randrange(len(mel) - T)
            seg = slice(s, s + T)
            pad = T - len(mel[seg])
            xs.append(np.pad(mel[seg], ((0, pad), (0, 0)), constant_values=np.log(1e-6)))
            ys.append(np.pad(y[seg], ((0, pad), (0, 0))))
            ms.append(np.pad(m[seg], (0, pad)))
        yield (torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys)),
               torch.tensor(np.stack(ms)))


def loss_fn(pred, y, m, w):
    m = m.float()[..., None]
    err = ((pred - y).abs() * w * m).sum() / (m.sum() * w.sum() + 1e-6)
    vel = (((pred[:, 1:] - pred[:, :-1]) - (y[:, 1:] - y[:, :-1])).abs() * w
           * m[:, 1:]).sum() / (m.sum() * w.sum() + 1e-6)
    return err + 2 * vel


@torch.no_grad()
def evaluate(net, clips, dev, norm):
    net.eval()
    tot, n = 0.0, 0
    for mel, y, m, _ in clips:
        x = torch.tensor((mel - norm[0]) / norm[1])[None].to(dev)
        p = net(x)[0].cpu().numpy()
        e = np.abs(p - y)[m][:, C.MOUTH]
        tot += e.sum()
        n += e.size
    net.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=200, help="batches per epoch")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    files = sorted(Path(a.data).glob("*.npz"))
    if len(files) < 2:
        sys.exit("need at least 2 extracted clips (extract.py)")
    random.shuffle(files)
    nval = max(1, int(len(files) * a.val))
    val, tr = load(files[:nval]), load(files[nval:])
    allmel = np.concatenate([c[0] for c in tr])
    norm = (allmel.mean(0), allmel.std(0) + 1e-5)
    tr = [((m - norm[0]) / norm[1], y, k, n) for m, y, k, n in tr]
    hours = sum(len(c[0]) for c in tr) / 100 / 3600
    print(f"train: {len(tr)} clips, {hours:.2f} h | validation: {len(val)} clips")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = C.model().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    w = torch.ones(len(C.ARKIT), device=dev) * 0.3
    w[C.MOUTH] = 1.0
    best = 1e9
    for ep in range(1, a.epochs + 1):
        t0, run = time.time(), 0.0
        for x, y, m in batches(tr, a.steps):
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            loss = loss_fn(net(x), y, m, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += loss.item()
        sched.step()
        v = evaluate(net, val, dev, norm)
        mark = ""
        if v < best:
            best = v
            torch.save({"state": net.state_dict(), "norm_mean": norm[0], "norm_std": norm[1],
                        "names": C.ARKIT, "val_mouth_err": v, "epoch": ep}, a.out)
            mark = "  (saved)"
        print(f"epoch {ep:3d}  train {run / a.steps:.4f}  val mouth error {v:.4f}"
              f"  {time.time() - t0:.0f}s{mark}", flush=True)
    print(f"best validation mouth error {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
