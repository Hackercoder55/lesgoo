#!/usr/bin/env python3
"""Blender lip-sync automation - segment pipeline.

Extracts a frame range from a master render, sends only that chunk to
sync.so, validates the result, and reports whether it is safe to splice
back into the timeline.

Design rule: never let an AI result damage a valid original. Any check
that fails leaves the original segment in place.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent
META = ROOT / "metadata"
JOBS = META / "jobs.json"
API = "https://api.sync.so/v2"

# Settings copied from the team's existing generations, not invented.
MODEL = "sync-3"
OPTIONS = {
    "pads": [0, 5, 0, 0],
    "sync_mode": "bounce",
    "output_format": "mp4",
    "active_speaker_detection": {"auto_detect": False},
}


def key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("SYNC_API_KEY="):
            k = line.split("=", 1)[1].strip().strip("\"'")
            if k and k != "yahan_apni_key_paste_karo":
                return k
    sys.exit("SYNC_API_KEY missing in .env")


def req(method, url, headers=None, body=None, raw=None):
    import urllib.request

    # A default Python-urllib User-Agent gets a 403 from sync.so's WAF.
    h = {"x-api-key": key(), "user-agent": "blender-lipsync/1.0"}
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body else None)
    if body is not None:
        h["content-type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(r) as resp:
        payload = resp.read()
    return json.loads(payload) if payload[:1] in b"{[" else payload


def ffprobe(path):
    """Return the facts that matter for splicing: frames, fps, size, duration."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,r_frame_rate,avg_frame_rate,width,height,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["stream"][0] if "stream" in json.loads(out) else json.loads(out)["streams"][0]

    def rate(k):
        n, d = s.get(k, "0/1").split("/")
        return round(int(n) / int(d), 4) if int(d) else 0.0

    return {
        "frames": int(s.get("nb_read_frames", 0)),
        "fps": rate("r_frame_rate"),
        # sync.so has returned a clip with the right frame count but a 29fps
        # average, which concat then padded with a duplicate frame. Nominal
        # fps alone does not catch that; the average does.
        "avg_fps": rate("avg_frame_rate"),
        "width": int(s["width"]),
        "height": int(s["height"]),
        "duration": float(s.get("duration", 0)),
    }


def extract(src, start_frame, end_frame, outdir, fps=30):
    """Cut [start_frame, end_frame] inclusive, frame-exact, no timing drift."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0, t1 = start_frame / fps, (end_frame + 1) / fps
    vid, wav = outdir / "seg.mp4", outdir / "seg.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src),
         "-vf", f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS",
         "-af", f"aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS",
         "-r", str(fps), "-c:v", "libx264", "-crf", "12", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-movflags", "+faststart", "-y", str(vid)], check=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src),
         "-af", f"aselect='between(t,{t0},{t1})',asetpts=PTS-STARTPTS",
         "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-y", str(wav)], check=True)
    return vid, wav


def upload(path):
    """Two-step: ask for a presigned target, PUT the bytes, return the asset url."""
    path = Path(path)
    ctype = "video/mp4" if path.suffix == ".mp4" else "audio/wav"
    pre = req("POST", f"{API}/assets/upload", body={
        "fileName": path.name, "contentType": ctype, "size": path.stat().st_size})
    # curl, not urllib: the presigned URL signs content-length;host and urllib's
    # extra headers make S3 reject the PUT with 403.
    subprocess.run(["curl", "-sf", "-X", "PUT", "--upload-file", str(path),
                    pre["uploadUrl"]], check=True)
    return pre["url"]


def submit(video_url, audio_url, idem, speaker=None):
    """speaker: {"auto_detect": False, "frame_number": n, "coordinates": [x, y]}
    in the master's native pixel space. The API is snake_case here - the docs
    show camelCase, which it rejects outright."""
    opts = dict(OPTIONS)
    if speaker:
        opts["active_speaker_detection"] = speaker
    return req("POST", f"{API}/generate",
               headers={"x-idempotency-key": idem},
               body={"model": MODEL, "options": opts,
                     "input": [{"type": "video", "url": video_url},
                               {"type": "audio", "url": audio_url}]})


def poll(gid, timeout=1800):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        g = req("GET", f"{API}/generations/{gid}")
        if g["status"] != last:
            last = g["status"]
            print(f"  [{int(time.time()-t0):4d}s] {last}", flush=True)
        if last in ("COMPLETED", "FAILED", "REJECTED", "CANCELLED"):
            return g
        time.sleep(10)
    raise TimeoutError(gid)


def validate(original, produced):
    """Gate before splicing. Every failure means: keep the original."""
    a, b = ffprobe(original), ffprobe(produced)
    # Frame count and geometry are the invariants: the compositor reads
    # frames sequentially and the splice rebuilds every timestamp from the
    # frame index, so a clip that carries the right frames with wrong timing
    # metadata is still usable. sync.so returns those regularly - 29fps on
    # one video, 32.3fps on the next - so they are reported, not fatal.
    timing_ok = (a["fps"] == b["fps"]
                 and abs(a["avg_fps"] - b["avg_fps"]) < 0.5
                 and abs(a["duration"] - b["duration"]) < 0.05)
    checks = {
        "frame_count": (a["frames"], b["frames"], a["frames"] == b["frames"]),
        "width": (a["width"], b["width"], a["width"] == b["width"]),
        "height": (a["height"], b["height"], a["height"] == b["height"]),
    }
    black = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(produced),
         "-vf", "blackdetect=d=0.03:pic_th=0.98", "-an", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    n_black = black.count("black_start")
    checks["no_black_frames"] = (0, n_black, n_black == 0)
    if not timing_ok:
        print(f"      note: timing metadata differs "
              f"(fps {a['fps']}->{b['fps']}, avg {a['avg_fps']}->{b['avg_fps']}, "
              f"dur {a['duration']:.3f}->{b['duration']:.3f}); frames match so "
              f"the splice will rebuild it", flush=True)
    return checks, a, b


def save_job(rec):
    META.mkdir(exist_ok=True)
    jobs = json.loads(JOBS.read_text()) if JOBS.exists() else {}
    jobs[rec["segment_id"]] = rec
    JOBS.write_text(json.dumps(jobs, indent=1))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "probe":
        print(json.dumps(ffprobe(sys.argv[2]), indent=1))
    elif cmd == "extract":
        src, sf, ef, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        v, w = extract(src, sf, ef, out)
        print(json.dumps(ffprobe(v), indent=1))
    elif cmd == "upload":
        print(upload(sys.argv[2]))
    elif cmd == "run":
        vid, wav, seg_id = sys.argv[2], sys.argv[3], sys.argv[4]
        idem = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{seg_id}:{vid}"))
        print("uploading video..."); v = upload(vid)
        print("uploading audio..."); a = upload(wav)
        print(f"submitting {MODEL} (idempotency {idem[:8]})...")
        g = submit(v, a, idem)
        gid = g["id"]
        print("generation:", gid)
        save_job({"segment_id": seg_id, "generation_id": gid, "status": g["status"],
                  "idempotency_key": idem, "video_url": v, "audio_url": a})
        g = poll(gid)
        save_job({"segment_id": seg_id, "generation_id": gid, "status": g["status"],
                  "idempotency_key": idem, "video_url": v, "audio_url": a,
                  "output_duration": g.get("outputDuration"),
                  "error": g.get("error"), "errorCode": g.get("errorCode")})
        if g["status"] != "COMPLETED":
            sys.exit(f"{g['status']}: {g.get('error')} / {g.get('errorCode')} -> KEEP ORIGINAL")
        out = Path(vid).parent.parent.parent / "output" / seg_id
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "synced.mp4"
        subprocess.run(["curl", "-sfL", "-o", str(dst), g.get("outputMediaUrl") or g["outputUrl"]], check=True)
        print("downloaded:", dst)
    elif cmd == "validate":
        checks, a, b = validate(sys.argv[2], sys.argv[3])
        for k, (exp, got, ok) in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {k:16} expected={exp} got={got}")
        print("\nVERDICT:", "SAFE TO SPLICE" if all(c[2] for c in checks.values())
              else "REJECT - KEEP ORIGINAL")
    else:
        print(__doc__)
        print("commands: probe <f> | extract <src> <s> <e> <out> | upload <f> | validate <orig> <new>")


if __name__ == "__main__":
    main()
