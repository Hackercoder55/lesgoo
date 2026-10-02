#!/usr/bin/env python3
"""One command: video in, lip-synced video plus a QC report out.

    python run.py source/video4.mp4              # plan only
    python run.py source/video4.mp4 --go         # plan, then generate locally
    python run.py source/video4.mp4 --go --engine syncso   # paid sync.so API
    python run.py source/video4.mp4 --go --engine noop     # plumbing test, no GPU

Everything this pipeline knows is enforced here rather than remembered.
The notes below each stage say which defect the code is guarding
against, because every one of them shipped at least once and passed the
checks that existed at the time.

Engines: "latentsync" (default) runs on the local GPU through
latentsync_worker.py and costs nothing per video, so it retries a failed
segment by itself with the next settings in LADDER. "syncso" spends
credits, so it never retries on its own - only with --retry.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import lipsync as L

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent                # LatentSync lives at the repository root
RATE = 0.5333                     # credits per frame, sync-3, measured
DW, DH = 540, 960                 # face-detection scale, reset per video
FACE_SCORE = 0.6                  # YuNet confidence, --face-score
YUNET = ROOT / "models" / "yunet.onnx"


# ---------------------------------------------------------------- probe

def probe(src):
    """Read the master's real properties. Nothing downstream is hardcoded:
    one video was tagged iec61966-2-1 and the next bt709, and writing the
    wrong one back shifts the whole picture."""
    s = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,"
         "color_range,color_space,color_transfer,color_primaries",
         "-of", "json", str(src)], capture_output=True, text=True).stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    trc = s.get("color_transfer") or "bt709"
    prim = s.get("color_primaries") or "bt709"
    csp = s.get("color_space") or "bt709"
    rng = s.get("color_range") or "tv"
    h264_trc = {"bt709": 1, "iec61966-2-1": 13, "smpte170m": 6}.get(trc, 1)
    w, h = int(s["width"]), int(s["height"])
    return {
        "src": Path(src).resolve(), "w": w, "h": h,
        # exact rate, not rounded: 29.97 treated as 30 drifts the spliced
        # picture against the master audio by a frame every ~33 s
        "fps": int(num) / int(den), "rate": f"{num}/{den}",
        "total": int(s["nb_read_frames"]),
        "trc": trc, "prim": prim, "csp": csp, "range": rng, "h264_trc": h264_trc,
        "setparams": (f"setparams=color_primaries={prim}:color_trc={trc}"
                      f":colorspace={csp}:range={rng},"),
    }


def set_det_size(w, h, long_side=960):
    """Detect on the master's own aspect ratio. A fixed 540x960 is right for
    9:16 shorts but squashes a 16:9 frame to a third of its width, and the
    detector then misses most faces."""
    global DW, DH
    k = long_side / max(w, h)
    DW, DH = int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2


def work_dirs(cfg):
    stem = cfg["src"].stem
    w = ROOT / f"work_{stem}"
    w.mkdir(exist_ok=True)
    (ROOT / f"segments_{stem}" / "input").mkdir(parents=True, exist_ok=True)
    (ROOT / f"segments_{stem}" / "output").mkdir(parents=True, exist_ok=True)
    cfg["work"] = w
    cfg["segin"] = ROOT / f"segments_{stem}" / "input"
    cfg["segout"] = ROOT / f"segments_{stem}" / "output"
    cfg["out"] = ROOT / "final" / f"FINAL_{stem}.mp4"
    return cfg


# ------------------------------------------------------------- analysis

def transcribe(cfg, model="small", language="en"):
    wav = cfg["work"] / "audio16k.wav"
    # keyed by model and language so changing either re-transcribes instead
    # of silently reusing the old words
    words = cfg["work"] / f"words_{model}_{language or 'auto'}.json"
    if words.exists():
        return json.loads(words.read_text())
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(cfg["src"]), "-map", "0:a:0",
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    "-y", str(wav)], check=True)
    from faster_whisper import WhisperModel
    m = WhisperModel(model, device="cpu", compute_type="int8")
    segs, _ = m.transcribe(str(wav), word_timestamps=True, vad_filter=False,
                           language=language)
    out = [{"start": round(s.start, 2), "end": round(s.end, 2),
            "text": s.text.strip(),
            "words": [{"w": w.word.strip(), "s": round(w.start, 2),
                       "e": round(w.end, 2)} for w in (s.words or [])]}
           for s in segs]
    words.write_text(json.dumps(out, indent=1))
    return out


def cuts_of(cfg, th=0.10):
    f = cfg["work"] / "cuts.txt"
    if not f.exists():
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(cfg["src"]), "-map", "0:v:0",
             "-vf", f"select='gt(scene,{th})',metadata=print:file=-",
             "-an", "-f", "null", "-"], capture_output=True, text=True).stdout
        ts, t = [], None
        for line in out.splitlines():
            if "pts_time:" in line:
                t = float(line.split("pts_time:")[1].split()[0])
            elif "scene_score=" in line and t is not None:
                ts.append(f"{t:.2f}")
        f.write_text("\n".join(ts))
    return sorted(float(x) for x in f.read_text().split())


def scan(cfg):
    """Per frame: is a face on screen, how big, where, and is its mouth moving.

    Mouth movement is scored against whole-frame movement. Raw mouth
    difference just measures cuts and camera moves - a shot change scores
    higher than a spoken sentence."""
    npz = cfg["work"] / f"scan_{DW}x{DH}_{FACE_SCORE}.npz"
    if npz.exists():
        return dict(np.load(npz))
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (DW, DH), FACE_SCORE, 0.3, 5000)
    p = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(cfg["src"]), "-map", "0:v:0",
         "-fps_mode", "passthrough", "-vf", f"scale={DW}:{DH}",
         "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    size = DW * DH * 3
    n = cfg["total"]
    act = np.full(n, np.nan)
    face = np.zeros(n, bool)
    area = np.zeros(n)
    cen = np.zeros((n, 2))
    mouth = np.zeros((n, 4), int)
    prev = None
    for i in range(n):
        b = p.stdout.read(size)
        if len(b) < size:
            break
        f = np.frombuffer(b, np.uint8).reshape(DH, DW, 3)
        _, faces = det.detect(np.ascontiguousarray(f))
        box = None
        if faces is not None and len(faces):
            fa = max(faces, key=lambda r: r[2] * r[3])
            face[i] = True
            area[i] = float(fa[2] * fa[3])
            cen[i] = [float(fa[0] + fa[2] / 2), float(fa[1] + fa[3] / 2)]
            rx, ry, lx, ly = fa[10], fa[11], fa[12], fa[13]
            cx, cy = (rx + lx) / 2, (ry + ly) / 2
            half = max(abs(lx - rx), 8.0)
            box = (max(0, int(cx - half)), max(0, int(cy - half * .7)),
                   min(DW, int(cx + half)), min(DH, int(cy + half * .7)))
            mouth[i] = box
        if prev is not None and box is not None:
            A, B = prev.astype(np.int16), f.astype(np.int16)
            g = float(np.abs(A - B).mean())
            if g <= 40:
                x0, y0, x1, y1 = box
                if x1 > x0 and y1 > y0:
                    act[i] = float(np.abs(A[y0:y1, x0:x1]
                                          - B[y0:y1, x0:x1]).mean()) / (g + 1e-3)
        prev = f
        if i % 600 == 0:
            print(f"    scanning {i}/{n}", flush=True)
    p.stdout.close(); p.wait()
    d = {"act": act, "face": face, "area": area, "cen": cen, "mouth": mouth}
    np.savez(npz, **d)
    return d


# ----------------------------------------------------------------- plan

def runs_of(mask):
    """[first, last] inclusive for every run of True frames."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append([i, j])
            i = j + 1
        else:
            i += 1
    return out


def build_plan(cfg, words, cuts, sc, min_face_pct, min_run, merge_gap):
    """Segments come from the footage, not from the transcript's grammar.

    Reading narration as voice-over and skipping it was right on two videos
    and wrong on the third, where the characters mouth the narration too -
    that mistake left most of the video unsynced.

    Returns the plan and the spoken stretches it leaves out, with the reason
    for each, so a skipped line is reported instead of silently missing."""
    n = cfg["total"]
    fps = cfg["fps"]
    speech = np.zeros(n, bool)
    for s in words:
        for w in s["words"]:
            speech[int(w["s"] * fps):int(w["e"] * fps) + 1] = True
    big = sc["area"] > DW * DH * (min_face_pct / 100.0)
    cand = sc["face"] & big & speech

    def shot(f):
        t = f / fps
        return (max([c for c in cuts if c <= t], default=0.0),
                min([c for c in cuts if c > t], default=n / fps))

    # Join first, then drop what is still too short. Dropping first lost
    # most of the dialogue: whisper leaves gaps between words and the face
    # detector misses odd frames, so a normal sentence arrives as 5-10 frame
    # pieces, and each piece fell under min_run before it could be joined.
    merged = []
    for r in runs_of(cand):
        if merged and r[0] - merged[-1][1] <= merge_gap \
                and shot(merged[-1][1]) == shot(r[0]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    kept = [r for r in merged if r[1] - r[0] + 1 >= min_run]

    def said(a, b):
        return " ".join(w["w"] for s in words for w in s["words"]
                        if w["e"] > a / fps and w["s"] < (b + 1) / fps)

    plan = [{"n": k, "final": [a, b], "frames": b - a + 1, "text": said(a, b)[:60]}
            for k, (a, b) in enumerate(kept, 1)]

    covered = np.zeros(n, bool)
    for a, b in kept:
        covered[a:b + 1] = True
    why = np.where(~sc["face"], 0, np.where(~big, 1, 2))
    reasons = ("no face detected", f"face under --min-face {min_face_pct}%",
               f"shorter than --min-run {min_run}")
    missed = [{"frames": [a, b],
               "why": reasons[int(np.bincount(why[a:b + 1], minlength=3).argmax())],
               "text": said(a, b)[:60]}
              for a, b in runs_of(speech & ~covered)]
    return plan, missed


def speakers(cfg, plan):
    """Name the face to sync and track it for the compositor.

    Left alone sync.so takes the most prominent face, and the compositor
    used to mask the biggest one - between them they twice synced a
    bystander and then threw the correct result away."""
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (DW, DH), FACE_SCORE, 0.3, 5000)
    sx, sy = cfg["w"] / DW, cfg["h"] / DH
    out = []
    for p in plan:
        a, b = p["final"]
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(cfg["src"]), "-map", "0:v:0",
             "-fps_mode", "passthrough",
             "-vf", f"select='between(n,{a},{b})',scale={DW}:{DH}",
             "-pix_fmt", "bgr24", "-f", "rawvideo", "-"], capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, DH, DW, 3)
        tracks = []
        for i, f in enumerate(fr):
            _, faces = det.detect(np.ascontiguousarray(f))
            if faces is None:
                continue
            for fa in faces:
                x, y, w, h = fa[:4]
                cx, cy = float(x + w / 2), float(y + h / 2)
                rx, ry, lx, ly = fa[10], fa[11], fa[12], fa[13]
                mx, my = (rx + lx) / 2, (ry + ly) / 2
                half = max(abs(lx - rx), 8.0)
                mb = (int(mx - half), int(my - half * .7),
                      int(mx + half), int(my + half * .7))
                best, bd = None, 60
                for t in tracks:
                    if t["last"] == i:
                        continue
                    dd = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** .5
                    if dd < bd:
                        best, bd = t, dd
                if best is None:
                    tracks.append({"cx": cx, "cy": cy, "last": i, "n": 1,
                                   "frames": {i: mb}})
                else:
                    best.update(cx=cx, cy=cy, last=i, n=best["n"] + 1)
                    best["frames"][i] = mb
        tracks = [t for t in tracks if t["n"] >= max(3, len(fr) * .2)]
        if not tracks:
            out.append({**p, "coords": None, "track": {}})
            continue

        def score(t):
            vals = []
            for i in sorted(t["frames"]):
                if i + 1 >= len(fr) or (i + 1) not in t["frames"]:
                    continue
                A, B = fr[i].astype(np.int16), fr[i + 1].astype(np.int16)
                g = float(np.abs(A - B).mean())
                if g > 40:
                    continue
                x0, y0, x1, y1 = t["frames"][i]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(DW, x1), min(DH, y1)
                if x1 > x0 and y1 > y0:
                    vals.append(float(np.abs(A[y0:y1, x0:x1]
                                             - B[y0:y1, x0:x1]).mean()) / (g + 1e-3))
            return float(np.mean(vals)) if vals else 0.0

        best = max(tracks, key=score)
        # the frame where this face is biggest - a small face at the wrong
        # moment got a whole generation rejected
        fnum = max(best["frames"], key=lambda i:
                   (best["frames"][i][2] - best["frames"][i][0])
                   * (best["frames"][i][3] - best["frames"][i][1]))
        mb = best["frames"][fnum]
        out.append({**p, "faces": len(tracks),
                    "coords": [round(((mb[0] + mb[2]) / 2) * sx),
                               round(((mb[1] + mb[3]) / 2) * sy)],
                    "frame_number": int(fnum),
                    "face_box_small": [int(v) for v in mb],
                    "track": {str(i): [round(((v[0] + v[2]) / 2) / DW, 5),
                                       round(((v[1] + v[3]) / 2) / DH, 5)]
                              for i, v in best["frames"].items()}})
    return out


# ------------------------------------------------------------- generate

def extract_all(cfg, plan):
    """One decode pass for every segment. Running one ffmpeg per segment
    scans the whole master each time and was the slowest stage by far."""
    fc, maps = [], []
    for i, p in enumerate(plan):
        a, b = p["final"]
        t0, t1 = a / cfg["fps"], (b + 1) / cfg["fps"]
        d = cfg["segin"] / f"s{p['n']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        fc.append(f"[0:v:0]select='between(n,{a},{b})',setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[0:a:0]aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS,"
                  f"asplit=2[a{i}][w{i}]")
        maps += ["-map", f"[v{i}]", "-map", f"[a{i}]",
                 "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                 "-pix_fmt", "yuv420p", "-r", cfg["rate"],
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                 "-movflags", "+faststart", str(d / "seg.mp4")]
        maps += ["-map", f"[w{i}]", "-vn", "-c:a", "pcm_s16le", "-ar", "48000",
                 str(d / "seg.wav")]
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(cfg["src"]),
                    "-filter_complex", ";".join(fc)] + maps + ["-y"], check=True)
    for p in plan:
        got = L.ffprobe(cfg["segin"] / f"s{p['n']:02d}" / "seg.mp4")
        assert got["frames"] == p["frames"], \
            f"s{p['n']:02d}: extracted {got['frames']} != planned {p['frames']}"


def face_image(cfg, seg, frame_number, box):
    sx = cfg["w"] / DW
    x0, y0, x1, y1 = [v * sx for v in box]
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    half = int(max(x1 - x0, y1 - y0))
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(seg),
         "-vf", f"select='eq(n,{frame_number})',"
                f"crop={half*2}:{half*2}:{max(0,cx-half)}:{max(0,cy-half)},"
                f"scale=128:128", "-frames:v", "1", "-f", "webp", "-"],
        capture_output=True).stdout
    return "data:image/webp;base64," + base64.b64encode(out).decode() if out else None


def generate_syncso(cfg, plan, retry=False):
    """A segment whose synced clip from an earlier run still matches its
    frame range and passes validation is reused, not paid for again.

    Without --retry a failed segment is resubmitted under the same
    idempotency key, so sync.so hands back the same failed generation.
    --retry gives it a new attempt number and with it a new key."""
    def one(p):
        sid = f"s{p['n']:02d}"
        d = cfg["segin"] / sid
        out = cfg["segout"] / sid
        meta = out / "result.json"
        prev = json.loads(meta.read_text()) if meta.exists() else {}
        same = prev.get("final") == p["final"]
        if same and prev.get("ok") and (out / "synced.mp4").exists():
            checks, _, _ = L.validate(d / "seg.mp4", out / "synced.mp4")
            if all(c[2] for c in checks.values()):
                print(f"  {sid} reusing earlier result", flush=True)
                return sid, True
        attempt = prev.get("attempt", 0) if same else 0
        if retry and same and not prev.get("ok"):
            attempt += 1

        def record(ok, status):
            out.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps({"final": p["final"], "ok": ok,
                                        "status": status, "attempt": attempt}))

        try:
            v, a = L.upload(d / "seg.mp4"), L.upload(d / "seg.wav")
            asd = None
            if p.get("coords"):
                asd = {"auto_detect": False, "frame_number": p["frame_number"],
                       "coordinates": p["coords"]}
                fi = face_image(cfg, d / "seg.mp4", p["frame_number"],
                                p["face_box_small"])
                if fi:
                    asd["face_image"] = fi
            idem = str(uuid.uuid5(uuid.NAMESPACE_URL,
                                  f"{cfg['src'].stem}:{sid}:{p['final'][0]}:"
                                  f"{p.get('coords')}"
                                  + (f":retry{attempt}" if attempt else "")))
            g = L.poll(L.submit(v, a, idem, asd)["id"])
            if g["status"] != "COMPLETED":
                print(f"  {sid} {g['status']} {g.get('error') or ''} -> keeping original",
                      flush=True)
                record(False, g["status"])
                return sid, False
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-sfL", "-o", str(out / "synced.mp4"),
                            g["outputMediaUrl"]], check=True)
            checks, _, _ = L.validate(d / "seg.mp4", out / "synced.mp4")
            ok = all(c[2] for c in checks.values())
            print(f"  {sid} {'OK' if ok else 'FAILED ' + str([k for k, c in checks.items() if not c[2]])}",
                  flush=True)
            record(ok, "COMPLETED" if ok else "VALIDATION_FAILED")
            return sid, ok
        except Exception as e:
            print(f"  {sid} ERROR {e} -> keeping original", flush=True)
            record(False, f"ERROR {e}")
            return sid, False

    with ThreadPoolExecutor(max_workers=7) as ex:
        return [s for s, ok in ex.map(one, plan) if ok]


# ------------------------------------------------- generate, local GPU

# One entry per attempt. A local run is free, so a failed segment moves on
# to the next entry by itself (up to --attempts) instead of waiting for
# --retry. margin scales the crop around the speaker: wider when the face
# detector lost the face, tighter when another face crept into the crop.
LADDER = [
    {"steps": 20, "guidance": 1.5, "seed": 1247, "margin": 1.0},
    {"steps": 30, "guidance": 2.0, "seed": 4321, "margin": 1.4},
    {"steps": 20, "guidance": 1.0, "seed": 777, "margin": 0.8},
]
LS_FPS = 25                       # LatentSync is trained and runs at 25 fps


def speaker_crop(cfg, p, margin, crop_max):
    """Square box around the speaker, in master pixels.

    LatentSync syncs the biggest face it finds and gives up on the whole
    clip if any frame has none. Cropping to the speaker named by speakers()
    makes it the only face - the local stand-in for sync.so's
    active_speaker_detection - and upscaling the crop gets small or
    stylised faces past its 50x80 px minimum."""
    W, H = cfg["w"], cfg["h"]
    sx, sy = W / DW, H / DH
    trk = p.get("track") or {}
    if not trk or not p.get("face_box_small"):
        side = min(W, H)
        x, y = (W - side) // 2, (H - side) // 2
    else:
        x0, _, x1, _ = p["face_box_small"]
        eye = (x1 - x0) / 2 * sx             # box half-width = eye distance
        xs = [v[0] * W for v in trk.values()]
        ys = [v[1] * H + eye * .5 for v in trk.values()]   # eyes -> face centre
        side = margin * 5 * eye + max(max(xs) - min(xs), max(ys) - min(ys))
        side = int(min(side, W, H)) // 2 * 2
        x = int(np.clip((max(xs) + min(xs)) / 2 - side / 2, 0, W - side))
        y = int(np.clip((max(ys) + min(ys)) / 2 - side / 2, 0, H - side))
    size = int(np.clip(side, 512, crop_max)) // 2 * 2
    return {"x": x, "y": y, "side": side, "size": size}


def prep_crop(seg, box, dst):
    """The speaker crop at 25 fps - LatentSync's own rate - and its size."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(seg), "-an",
         "-vf", f"crop={box['side']}:{box['side']}:{box['x']}:{box['y']},"
                f"scale={box['size']}:{box['size']}:flags=lanczos,fps={LS_FPS}",
         "-c:v", "libx264", "-crf", "12", "-preset", "fast", "-pix_fmt", "yuv420p",
         "-y", str(dst)], check=True)


def uncrop(cfg, p, seg, box, crop_out, dst):
    """Put the synced crop back into the segment at the master's frame rate,
    exactly p["frames"] frames long, the segment's own pixels everywhere
    else. composite() then keeps only the speaker's mouth from it."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(seg), "-i", str(crop_out),
         "-filter_complex",
         f"[1:v]setpts=PTS-STARTPTS,fps={cfg['rate']},"
         f"scale={box['side']}:{box['side']}:flags=lanczos,"
         f"tpad=stop_mode=clone:stop_duration=2[c];"
         f"[0:v][c]overlay={box['x']}:{box['y']}:eof_action=repeat[v]",
         "-map", "[v]", "-map", "0:a?", "-frames:v", str(p["frames"]),
         "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-r", cfg["rate"],
         "-movflags", "+faststart", "-y", str(dst)], check=True)


def run_worker(jobs, a, log):
    """One worker process for the whole batch: the model loads once."""
    if a.engine == "noop":
        # no model: hand the crop back unchanged. Proves cropping, retiming,
        # compositing and splicing on any machine; verify() must then
        # report NO LIPSYNC on every segment, which is the point.
        for j in jobs:
            shutil.copyfile(j["video"], j["out"])
            Path(j["out"] + ".json").write_text(json.dumps(
                {"ok": True, "error": None, "seconds": 0.0, "peak_vram_gb": 0,
                 "settings": {k: j[k] for k in ("steps", "guidance", "seed")}}))
        return
    jf = log.parent / "jobs.json"
    jf.write_text(json.dumps(jobs, indent=1))
    cmd = [a.ls_python, str(ROOT / "latentsync_worker.py"), "--jobs", str(jf),
           "--config", a.ls_config, "--ckpt", a.ls_ckpt]
    if a.deepcache:
        cmd.append("--deepcache")
    with open(log, "a") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n"); lf.flush()
        pr = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        for line in pr.stdout:
            lf.write(line)
            if line.startswith("[worker]") or "Error" in line:
                print("   ", line.rstrip(), flush=True)
        pr.wait()
    if pr.returncode:
        print(f"    worker exited {pr.returncode}; see {log}", flush=True)


def generate_local(cfg, plan, a):
    """Segments that already passed are reused. A failed segment is retried
    with the next LADDER entry, up to --attempts in this run; one that has
    used them all is left for manual review until --retry, which continues
    the ladder where it stopped. The original stays in place for any
    segment that never passes."""
    ladder = [dict(LADDER[0], **{k: v for k, v in
                                 (("steps", a.steps), ("guidance", a.guidance),
                                  ("seed", a.seed)) if v is not None})] + LADDER[1:]
    log = cfg["work"] / "pipeline.log"
    state, todo, ok = {}, [], set()
    for p in plan:
        sid = f"s{p['n']:02d}"
        out = cfg["segout"] / sid
        meta = out / "result.json"
        prev = json.loads(meta.read_text()) if meta.exists() else {}
        same = prev.get("final") == p["final"] and prev.get("engine") == a.engine
        hist = prev.get("attempts", []) if same else []
        state[sid] = {"p": p, "out": out, "meta": meta, "hist": hist}
        if same and prev.get("ok") and (out / "synced.mp4").exists():
            checks, _, _ = L.validate(cfg["segin"] / sid / "seg.mp4", out / "synced.mp4")
            if all(c[2] for c in checks.values()):
                print(f"  {sid} reusing earlier result", flush=True)
                ok.add(sid)
                continue
        if hist and not a.retry:
            print(f"  {sid} failed before ({hist[-1]['error']}) - "
                  f"--retry to try again", flush=True)
            continue
        todo.append(sid)

    budget = {sid: a.attempts for sid in todo}
    while todo:
        jobs = []
        for sid in todo:
            st = state[sid]
            st["try"] = ladder[min(len(st["hist"]), len(ladder) - 1)]
            st["box"] = speaker_crop(cfg, st["p"], st["try"]["margin"], a.crop_max)
            st["out"].mkdir(parents=True, exist_ok=True)
            prep_crop(cfg["segin"] / sid / "seg.mp4", st["box"], st["out"] / "crop_in.mp4")
            (st["out"] / "crop_out.mp4.json").unlink(missing_ok=True)
            jobs.append({"id": sid, "video": str(st["out"] / "crop_in.mp4"),
                         "audio": str(cfg["segin"] / sid / "seg.wav"),
                         "out": str(st["out"] / "crop_out.mp4"),
                         **{k2: st["try"][k2] for k2 in ("steps", "guidance", "seed")}})
        print(f"  running {a.engine} on {len(jobs)} segment(s)...", flush=True)
        run_worker(jobs, a, log)

        nxt = []
        for sid in todo:
            st, p = state[sid], state[sid]["p"]
            rf = st["out"] / "crop_out.mp4.json"
            res = json.loads(rf.read_text()) if rf.exists() else \
                {"ok": False, "error": "worker produced no result (crashed?)"}
            if res["ok"]:
                try:
                    uncrop(cfg, p, cfg["segin"] / sid / "seg.mp4", st["box"],
                           st["out"] / "crop_out.mp4", st["out"] / "synced.mp4")
                    checks, _, _ = L.validate(cfg["segin"] / sid / "seg.mp4",
                                              st["out"] / "synced.mp4")
                    bad = [k for k, c in checks.items() if not c[2]]
                    if bad:
                        res.update(ok=False, error=f"validation failed: {bad}")
                except subprocess.CalledProcessError as e:
                    res.update(ok=False, error=f"rebuild failed: {e}")
            st["hist"].append({**st["try"], "ok": res["ok"], "error": res.get("error"),
                               "seconds": res.get("seconds"),
                               "peak_vram_gb": res.get("peak_vram_gb")})
            p["crop"] = st["box"]
            st["meta"].write_text(json.dumps(
                {"final": p["final"], "engine": a.engine, "ok": res["ok"],
                 "crop": st["box"], "attempts": st["hist"]}, indent=1))
            budget[sid] -= 1
            if res["ok"]:
                ok.add(sid)
                print(f"  {sid} OK (attempt {len(st['hist'])})", flush=True)
            else:
                more = budget[sid] > 0 and len(st["hist"]) < len(ladder)
                print(f"  {sid} FAILED attempt {len(st['hist'])}: {res.get('error')}"
                      f"{' - retrying with next settings' if more else ''}", flush=True)
                if more:
                    nxt.append(sid)
        todo = nxt

    # the compositor needs the crop of every usable segment, reused ones too
    for sid in ok:
        st = state[sid]
        prev = json.loads(st["meta"].read_text())
        st["p"]["crop"] = prev.get("crop")
    report = {sid: {"ok": sid in ok, "attempts": st["hist"],
                    "frames": st["p"]["frames"]} for sid, st in state.items()}
    (cfg["work"] / "report.json").write_text(json.dumps(report, indent=1))
    return sorted(ok), report


# ------------------------------------------------------------ composite

def composite(cfg, plan, usable):
    """Replace only the speaker's mouth, level-matched.

    sync.so returns frames darker than it was given, and more so outside
    the face than on it, so dropping them in whole makes every boundary
    flicker. All of this stays in yuv444p and writes no colour tags - a
    bgr24 roundtrip moved the picture +2.3 Y and tagging the encoder
    another -4.5 Y."""
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (DW, DH), FACE_SCORE, 0.3, 5000)
    W, H = cfg["w"], cfg["h"]
    scx, scy = W / DW, H / DH
    plane, frame = W * H, W * H * 3
    for p in plan:
        sid = f"s{p['n']:02d}"
        if sid not in usable:
            continue
        src, syn = cfg["segin"] / sid / "seg.mp4", cfg["segout"] / sid / "synced.mp4"
        dst = cfg["segout"] / sid / "composited.mp4"
        trk = {int(k): (v[0] * DW, v[1] * DH) for k, v in (p.get("track") or {}).items()}
        rd = lambda f: subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(f), "-fps_mode", "passthrough",
             "-pix_fmt", "yuv444p", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
        A, B = rd(src), rd(syn)
        out = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p",
             "-s", f"{W}x{H}", "-r", cfg["rate"], "-i", "-", "-i", str(src),
             "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "12",
             "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-b:a", "192k", "-fps_mode", "passthrough", "-y", str(dst)],
            stdin=subprocess.PIPE)
        last = np.zeros((H, W), np.float32)
        # A local engine changed only the speaker crop: everything outside it
        # is the original, so matching levels over the whole frame would
        # measure nothing. Match over the crop's own unmasked pixels instead.
        region = np.ones((H, W), bool)
        if p.get("crop"):
            c = p["crop"]
            region[:] = False
            region[c["y"]:c["y"] + c["side"], c["x"]:c["x"] + c["side"]] = True
        want, n = None, 0
        while True:
            ra, rb = A.stdout.read(frame), B.stdout.read(frame)
            if len(ra) < frame or len(rb) < frame:
                break
            po = [np.frombuffer(ra, np.uint8, plane, i * plane)
                  .reshape(H, W).astype(np.float32) for i in range(3)]
            ps = [np.frombuffer(rb, np.uint8, plane, i * plane)
                  .reshape(H, W).astype(np.float32) for i in range(3)]
            want = trk.get(n, want)
            small = cv2.cvtColor(np.stack([cv2.resize(x, (DW, DH)) for x in po],
                                          axis=-1).astype(np.uint8),
                                 cv2.COLOR_YUV2BGR)
            _, faces = det.detect(np.ascontiguousarray(small))
            if faces is not None and len(faces):
                if want is not None:
                    fa = min(faces, key=lambda r: (r[0] + r[2] / 2 - want[0]) ** 2
                                                  + (r[1] + r[3] / 2 - want[1]) ** 2)
                else:
                    fa = max(faces, key=lambda r: r[2] * r[3])
                x, y, w, h = [float(v) for v in fa[:4]]
                cx, cy = (x + w / 2) * scx, (y + h * .72) * scy
                ax, ay = w * .62 * scx, h * .42 * scy
                m = np.zeros((H, W), np.uint8)
                cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
                k = int(max(ax, ay) * .25) | 1
                last = cv2.GaussianBlur(m, (k, k), 0).astype(np.float32) / 255.0
            outside = (last < 0.02) & region
            keep = outside.sum() > 1000
            buf = bytearray()
            for c in range(3):
                s = ps[c]
                if keep:
                    oc, scc = po[c][outside], s[outside]
                    sd = scc.std()
                    g = float(np.clip((oc.std() / sd) if sd > 1e-3 else 1.0, .9, 1.1))
                    s = s * g + (oc.mean() - g * scc.mean())
                buf += np.clip(po[c] * (1 - last) + s * last, 0, 255).astype(np.uint8).tobytes()
            out.stdin.write(bytes(buf))
            n += 1
        for q in (A, B):
            q.stdout.close(); q.wait()
        out.stdin.close(); out.wait()
        print(f"  {sid} {n} frames {'OK' if n == p['frames'] else 'MISMATCH'}", flush=True)


# --------------------------------------------------------------- splice

def splice(cfg, plan, usable):
    segs = sorted((p["final"][0], p["final"][1], f"s{p['n']:02d}")
                  for p in plan if f"s{p['n']:02d}" in usable)
    inputs = ["-i", str(cfg["src"])]
    for _, _, sid in segs:
        c = cfg["segout"] / sid / "composited.mp4"
        inputs += ["-i", str(c if c.exists() else cfg["segout"] / sid / "synced.mp4")]
    sp = cfg["setparams"]
    num, den = cfg["rate"].split("/")
    pts = f"setpts=N*{den}/{num}/TB"
    fc, labels, cursor = [], [], 0
    for idx, (a, b, _) in enumerate(segs, start=1):
        if a > cursor:
            fc.append(f"[0:v:0]trim=start_frame={cursor}:end_frame={a},"
                      f"{sp}{pts}[o{idx}]")
            labels.append(f"o{idx}")
        fc.append(f"[{idx}:v]{sp}{pts}[s{idx}]")
        labels.append(f"s{idx}")
        cursor = b + 1
    if cursor < cfg["total"]:
        fc.append(f"[0:v:0]trim=start_frame={cursor}:end_frame={cfg['total']},"
                  f"{sp}{pts}[oz]")
        labels.append("oz")
    fc.append("".join(f"[{l}]" for l in labels) + f"concat=n={len(labels)}:v=1:a=0[out]")
    cfg["out"].parent.mkdir(exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", *inputs,
                    "-filter_complex", ";".join(fc),
                    "-map", "[out]", "-map", "0:a:0",
                    "-c:v", "libx264", "-crf", "14", "-preset", "fast",
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                    "-c:a", "aac", "-b:a", "192k",
                    "-bsf:v", f"h264_metadata=colour_primaries=1:"
                              f"transfer_characteristics={cfg['h264_trc']}:"
                              f"matrix_coefficients=1:video_full_range_flag=0",
                    "-movflags", "+faststart", "-y", str(cfg["out"])], check=True)
    print(f"  spliced {len(segs)} segments into {cfg['out']}")


# --------------------------------------------------------------- verify

def verify(cfg, plan, usable):
    w, h = 270, int(270 * cfg["h"] / cfg["w"])

    def y(path):
        # passthrough and the Y plane: a raw pipe defaults to CFR and adds a
        # frame, and reading through RGB applies the transfer tag - both once
        # made this very check report a corrupted baseline
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path),
                            "-map", "0:v:0", "-fps_mode", "passthrough",
                            "-vf", f"scale={w}:{h}", "-pix_fmt", "yuv420p",
                            "-f", "rawvideo", "-"], capture_output=True).stdout
        fs = w * h * 3 // 2
        n = len(r) // fs
        return (np.frombuffer(r[:n * fs], np.uint8).reshape(n, fs)[:, :w * h]
                .reshape(n, h, w).astype(np.float32))

    new, old = y(cfg["out"]), y(cfg["src"])
    n = min(len(new), len(old))
    fails = []
    print(f"\n  frames: final={len(new)} original={len(old)} "
          f"{'MATCH' if len(new) == len(old) == cfg['total'] else 'MISMATCH'}")
    if not (len(new) == len(old) == cfg["total"]):
        fails.append("frame count")

    segs = [p for p in plan if f"s{p['n']:02d}" in usable]
    inside = np.zeros(n, bool)
    for p in segs:
        a, b = p["final"]
        inside[a:b + 1] = True
    d = np.abs(new[:n] - old[:n]).mean(axis=(1, 2))
    stray = int(((~inside) & (d > 3)).sum())
    print(f"  untouched frames altered: {stray}  (mean {d[~inside].mean():.3f})"
          f"  {'PASS' if stray == 0 else 'FAIL'}")
    if stray:
        fails.append("untouched frames changed")

    worst = max(abs(float((new[p["final"][0]:p["final"][1] + 1]
                           - old[p["final"][0]:p["final"][1] + 1]).mean()))
                for p in segs) if segs else 0.0
    print(f"  worst level shift: {worst:.2f} Y  {'PASS' if worst < 1.0 else 'FAIL'}")
    if worst >= 1.0:
        fails.append("level shift (flicker)")

    dn = np.abs(np.diff(new, axis=0)).mean(axis=(1, 2))
    do = np.abs(np.diff(old, axis=0)).mean(axis=(1, 2))
    ws = 0.0
    for p in segs:
        a, b = p["final"]
        for i in (a - 1, b):
            if 0 <= i < len(dn):
                ws = max(ws, abs(dn[i] - do[i]))
    print(f"  worst seam delta: {ws:.2f}  (original std {do.std():.2f})  "
          f"{'PASS' if ws < do.std() else 'CHECK'}")
    if ws >= do.std():
        fails.append("seam")

    print("\n  lipsync present on the speaker:")
    for p in segs:
        a, b = p["final"]
        trk = p.get("track") or {}
        rd = lambda src: np.frombuffer(subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:v:0",
             "-fps_mode", "passthrough",
             "-vf", f"select='between(n,{a},{b})',scale={DW}:{DH}",
             "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
            capture_output=True).stdout, np.uint8).reshape(-1, DH, DW, 3).astype(np.float32)
        o, f = rd(cfg["src"]), rd(cfg["out"])
        m = min(len(o), len(f))
        vals = []
        for i in range(m):
            if str(i) not in trk:
                continue
            cx, cy = trk[str(i)][0] * DW, trk[str(i)][1] * DH
            x0, y0 = max(0, int(cx - 34)), max(0, int(cy - 26))
            x1, y1 = min(DW, int(cx + 34)), min(DH, int(cy + 26))
            if x1 > x0 and y1 > y0:
                vals.append(np.abs(f[i, y0:y1, x0:x1] - o[i, y0:y1, x0:x1]).mean())
        mouth = float(np.mean(vals)) if vals else 0.0
        bad = mouth < 3.0
        print(f"    s{p['n']:02d} mouth diff {mouth:6.2f}{'   NO LIPSYNC' if bad else ''}")
        if bad:
            fails.append(f"s{p['n']:02d} no lipsync")

    def count(path, filt, tok):
        e = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", filt,
                            "-an", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
        return e.count(tok)

    for lbl, p in (("original", cfg["src"]), ("final", cfg["out"])):
        print(f"  {lbl:9} black={count(p, 'blackdetect=d=0.03:pic_th=0.98', 'black_start')}"
              f" freeze={count(p, 'freezedetect=n=0.001:d=0.5', 'freeze_start')}")
    return fails


# ----------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--go", action="store_true", help="generate and build")
    ap.add_argument("--engine", choices=("latentsync", "syncso", "noop"),
                    default="latentsync",
                    help="latentsync: local GPU, free. syncso: paid API. "
                         "noop: no model, tests the plumbing")
    ap.add_argument("--min-face", type=float, default=2.0,
                    help="smallest face to sync, %% of frame area")
    ap.add_argument("--min-run", type=int, default=12)
    ap.add_argument("--merge-gap", type=int, default=10)
    ap.add_argument("--face-score", type=float, default=0.6,
                    help="face detector confidence; lower it for stylised faces")
    ap.add_argument("--language", default="en",
                    help="speech language for whisper, e.g. en, hi; 'auto' detects")
    ap.add_argument("--whisper-model", default="small",
                    help="small, medium, large-v3 - bigger misses fewer words")
    ap.add_argument("--retry", action="store_true",
                    help="try again segments that failed on an earlier --go")
    g = ap.add_argument_group("local engine")
    g.add_argument("--attempts", type=int, default=2,
                   help="tries per segment in one run, each with the next "
                        "LADDER settings (default 2)")
    g.add_argument("--steps", type=int, help="diffusion steps, first attempt (20)")
    g.add_argument("--guidance", type=float, help="guidance scale, first attempt (1.5)")
    g.add_argument("--seed", type=int, help="seed, first attempt (1247)")
    g.add_argument("--crop-max", type=int, default=768,
                   help="largest speaker crop fed to the model, px; lower it "
                        "on small GPUs")
    g.add_argument("--ls-python", default=os.environ.get("LATENTSYNC_PYTHON",
                                                         sys.executable),
                   help="python of the LatentSync environment "
                        "(or set LATENTSYNC_PYTHON)")
    g.add_argument("--ls-config", default="configs/unet/stage2.yaml",
                   help="stage2.yaml = 256px LatentSync 1.5 (8 GB); "
                        "stage2_512.yaml = 512px LatentSync 1.6 (18 GB)")
    g.add_argument("--ls-ckpt", default="checkpoints/latentsync_unet.pt",
                   help="relative to the repository root")
    g.add_argument("--deepcache", action="store_true",
                   help="faster LatentSync, slightly lower quality")
    a = ap.parse_args()
    paid = a.engine == "syncso"

    global FACE_SCORE
    FACE_SCORE = a.face_score
    cfg = work_dirs(probe(a.video))
    set_det_size(cfg["w"], cfg["h"])
    print(f"{cfg['src'].name}: {cfg['w']}x{cfg['h']} {cfg['fps']:.3f}fps "
          f"{cfg['total']} frames, transfer={cfg['trc']}")

    print("\n[1/7] transcribing")
    words = transcribe(cfg, a.whisper_model,
                       None if a.language == "auto" else a.language)
    print("[2/7] scene cuts")
    cuts = cuts_of(cfg)
    print(f"      {len(cuts)} cuts")
    print("[3/7] scanning faces and mouths")
    sc = scan(cfg)
    print("[4/7] planning from the footage")
    plan, missed = build_plan(cfg, words, cuts, sc, a.min_face, a.min_run,
                              a.merge_gap)
    plan = speakers(cfg, plan)
    (cfg["work"] / "plan.json").write_text(json.dumps(plan, indent=1))
    (cfg["work"] / "missed.json").write_text(json.dumps(missed, indent=1))

    total = sum(p["frames"] for p in plan)
    print(f"\n{'seg':6}{'time':>16}{'sec':>6}{'cr' if paid else '':>6}  words")
    for p in plan:
        x, b = p["final"]
        cr = f"{p['frames']*RATE:6.0f}" if paid else " " * 6
        print(f"s{p['n']:02d}  {x/cfg['fps']:7.2f}-{(b+1)/cfg['fps']:<8.2f}"
              f"{p['frames']/cfg['fps']:6.1f}{cr}  {p['text'][:44]}")
    print(f"\n{len(plan)} segments, {total} frames = {total/cfg['fps']:.1f}s "
          f"of {cfg['total']/cfg['fps']:.1f}s - "
          f"{(1-total/cfg['total'])*100:.0f}% of the video skipped")
    if paid:
        print(f"sync.so: {total*RATE:.0f} credits (${total*RATE/100:.2f}); "
              f"whole video would be {cfg['total']*RATE:.0f}")
    else:
        print(f"engine {a.engine}: local, no credits")

    long_missed = [m for m in missed if m["frames"][1] - m["frames"][0] + 1 >= 3]
    if long_missed:
        print(f"\nspeech NOT synced ({len(long_missed)} stretches) - check these:")
        for m in long_missed:
            x, b = m["frames"]
            print(f"  {x/cfg['fps']:7.2f}-{(b+1)/cfg['fps']:<8.2f}{m['why']:28}"
                  f"{m['text'][:40]}")

    if not a.go:
        print("\nplan only. re-run with --go to generate"
              + (" (spends credits)." if paid else "."))
        return

    print(f"\n[5/7] extracting and generating ({a.engine})")
    t0 = time.time()
    extract_all(cfg, plan)
    report = None
    if paid:
        usable = set(generate_syncso(cfg, plan, a.retry))
    else:
        usable, report = generate_local(cfg, plan, a)
        usable = set(usable)
    gen_s = time.time() - t0
    (cfg["work"] / "usable.json").write_text(json.dumps(sorted(usable)))
    failed = [f"s{p['n']:02d}" for p in plan if f"s{p['n']:02d}" not in usable]
    if failed:
        print(f"\n  not synced, original kept: {', '.join(failed)}"
              f"\n  run again with --go --retry to resubmit only these")
    if not usable:
        sys.exit("nothing usable - original left untouched")

    print("\n[6/7] compositing")
    composite(cfg, plan, usable)
    splice(cfg, plan, usable)

    print("\n[7/7] verifying")
    fails = verify(cfg, plan, usable)
    print("\n" + ("ALL CHECKS PASSED" if not fails else "FAILED: " + ", ".join(fails)))
    if report is not None:
        tries = [t for r in report.values() for t in r["attempts"]]
        vram = max([t.get("peak_vram_gb") or 0 for t in tries], default=0)
        print(f"\n  input duration      {cfg['total']/cfg['fps']:8.1f} s"
              f"\n  speaking (planned)  {total/cfg['fps']:8.1f} s"
              f"\n  segments synced     {len(usable):8d} / {len(plan)}"
              f"\n  model runs          {len(tries):8d}"
              f"  (retries {max(0, len(tries) - len(report))})"
              f"\n  peak VRAM           {vram:8.2f} GB"
              f"\n  generate time       {gen_s:8.0f} s"
              f"\n  report              {cfg['work'] / 'report.json'}")
        if failed:
            print(f"  manual review       {', '.join(failed)}")
    print(f"\n{cfg['out']}")


if __name__ == "__main__":
    main()
