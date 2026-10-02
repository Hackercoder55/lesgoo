"""Long-video auto mode: find where people speak, lip-sync only those parts,
and splice them back into the untouched original.

Speech is found from the audio's loudness (no model download needed), the
face for each stretch is the detector's most confident one inside it, and
every frame outside the synced stretches is copied from the original.
"""

import subprocess
import time
from pathlib import Path

import numpy as np

import app as core

DEFAULTS = {
    "min_speech": 0.30,     # s - shorter blips are not dialogue
    "merge_gap": 0.40,      # s - pauses shorter than this stay in one segment
    "pad_before": 0.15,     # s - start syncing a little before the first sound
    "pad_after": 0.20,
    "threshold_db": 12.0,   # dB above the noise floor that counts as speech
    "max_segment": 20.0,    # s - longer speech is split, keeps VRAM and retries small
}


def speech_segments(video, opts, log):
    """[(start_s, end_s), ...] where the soundtrack is louder than its own
    noise floor. Music beds raise the floor too, so the threshold follows
    the track rather than being a fixed level."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0",
                          "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True).stdout
    a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    if not len(a):
        raise RuntimeError("the video has no audio track")
    hop = 320                                    # 20 ms
    n = len(a) // hop
    rms = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10)
    db = 20 * np.log10(rms)
    floor = np.percentile(db, 10)
    on = db > max(floor + opts["threshold_db"], -50)
    # smooth over 100 ms so single loud frames do not flicker the mask
    on = np.convolve(on.astype(float), np.ones(5) / 5, "same") > .4
    segs, i = [], 0
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            segs.append([i * .02, (j + 1) * .02])
            i = j + 1
        else:
            i += 1
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= opts["merge_gap"]:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    dur = n * .02
    out = []
    for s, e in merged:
        if e - s < opts["min_speech"]:
            continue
        s, e = max(0, s - opts["pad_before"]), min(dur, e + opts["pad_after"])
        while e - s > opts["max_segment"] * 1.5:
            out.append((s, s + opts["max_segment"]))
            s += opts["max_segment"]
        out.append((s, e))
    # padding can make neighbours overlap; join them. Strictly overlapping
    # only: the max_segment pieces above touch end to start, and joining
    # those undid the split (a 92 s monologue went to the model in one go)
    final = []
    for s, e in out:
        if final and s < final[-1][1]:
            final[-1] = (final[-1][0], max(e, final[-1][1]))
        else:
            final.append((s, e))
    log(f"speech found in {len(final)} stretch(es), "
        f"{sum(e - s for s, e in final):.1f}s of {dur:.1f}s")
    return final


def best_face(seg, m, score):
    """Most confident face over a few frames of the segment, and when."""
    det = core.Detector(m["w"], m["h"], score)
    best = None
    for k in (.5, .25, .75, .1, .9):
        t = m["duration"] * k
        try:
            faces = det(core.frame_at(seg, t))
        except RuntimeError:
            continue
        if faces and (best is None or faces[0][4] > best[1][4]):
            best = (t, faces[0])
    if not best:
        return None
    t, (x, y, w, h, _) = best
    return {"t": t, "box": (x + w / 2, y + h / 2, w, h)}


def run(video, workdir, settings, ls, log, cancelled=lambda: False):
    """Returns the finished video and a per-segment report."""
    work = Path(workdir)
    m = core.probe(video)
    total = int(round(m["duration"] * m["fps"]))
    opts = {**DEFAULTS, **{k: v for k, v in (settings.get("auto") or {}).items()
                          if k in DEFAULTS}}
    segs = speech_segments(video, opts, log)
    fr = m["fps"]
    plan = []
    for s, e in segs:
        a, b = int(s * fr), min(total, int(round(e * fr)))
        if b - a >= 3:
            plan.append((a, b))                      # [a, b) in frames
    report, done = [], []
    todo = sum(b - a for a, b in plan) / fr
    spent = synced = 0.0
    for k, (a, b) in enumerate(plan, 1):
        if cancelled():
            raise RuntimeError("cancelled")
        sid = f"seg{k:03d}"
        d = work / sid
        d.mkdir(parents=True, exist_ok=True)
        seg = d / "source.mp4"
        eta = ""
        if synced:
            left = (todo - synced) * spent / synced
            eta = f" - about {left/60:.0f} min left" if left >= 60 else f" - about {left:.0f}s left"
        log(f"[{k}/{len(plan)}] {a/fr:.2f}-{b/fr:.2f}s{eta}")
        t0 = time.time()
        core.sh(["ffmpeg", "-v", "error", "-i", str(video), "-filter_complex",
                 f"[0:v:0]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v];"
                 f"[0:a:0]atrim={a/fr:.6f}:{b/fr:.6f},asetpts=PTS-STARTPTS[a]",
                 "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "12",
                 "-preset", "fast", "-pix_fmt", "yuv420p", "-r", m["rate"],
                 "-c:a", "pcm_s16le", "-y", str(seg)])
        sm = core.probe(seg)
        pick = best_face(seg, sm, settings["score"])
        rec = {"segment": sid, "start": round(a / fr, 3), "end": round(b / fr, 3)}
        if not pick:
            log("    no face found - left as the original")
            report.append({**rec, "status": "skipped", "why": "no face detected"})
            continue
        try:
            out = core.lipsync(seg, None, pick, "cut", settings["engine"],
                               settings["steps"], settings["guidance"],
                               settings["seed"], settings["crop_max"],
                               settings["score"], ls,
                               log=lambda s: log("    " + s), run=d / "run")
            n = count_frames(out)
            if abs(n - (b - a)) > 2:      # splice() evens out a frame or two
                raise RuntimeError(f"synced segment has {n} frames, expected {b - a}")
            done.append((a, b, out))
            report.append({**rec, "status": "synced",
                           "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            log(f"    FAILED: {e} - left as the original")
            report.append({**rec, "status": "failed", "why": str(e)})
        spent += time.time() - t0
        synced += (b - a) / fr
    dst = work / "lipsynced.mp4"
    splice(video, m, total, done, dst)
    log(f"spliced {len(done)} synced segment(s) into the full video")
    return dst, report


def count_frames(path):
    return int(core.sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", str(path)]).strip().split(",")[0])


def splice(video, m, total, done, dst):
    """Original frames everywhere except the synced segments; original audio
    untouched. Timestamps are rebuilt from the frame index so nothing drifts."""
    num, den = m["rate"].split("/")
    pts = f"setpts=N*{den}/{num}/TB"
    inputs, fc, labels, cur = ["-i", str(video)], [], [], 0
    for i, (a, b, path) in enumerate(done, 1):
        inputs += ["-i", str(path)]
        if a > cur:
            fc.append(f"[0:v:0]trim=start_frame={cur}:end_frame={a},{pts}[o{i}]")
            labels.append(f"o{i}")
        fc.append(f"[{i}:v:0]format=yuv420p,tpad=stop_mode=clone:stop={b - a},"
                  f"trim=end_frame={b - a},{pts}[s{i}]")
        labels.append(f"s{i}")
        cur = b
    if cur < total:
        fc.append(f"[0:v:0]trim=start_frame={cur},{pts}[oz]")
        labels.append("oz")
    fc.append("".join(f"[{x}]" for x in labels) + f"concat=n={len(labels)}:v=1:a=0[v]")
    core.sh(["ffmpeg", "-v", "error", *inputs, "-filter_complex", ";".join(fc),
             "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-crf", "14",
             "-preset", "medium", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-y", str(dst)])
