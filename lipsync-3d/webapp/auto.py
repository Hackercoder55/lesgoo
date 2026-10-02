"""Long-video auto mode: find where people speak, lip-sync only those parts,
and splice them back into the untouched original.

Speech is found from the audio's loudness (no model download needed), the
face for each stretch is the detector's most confident one inside it, and
every frame outside the synced stretches is copied from the original.
"""

import json
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
    "scene_cut": 0.3,       # ffmpeg scene score that counts as a shot change
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


# ------------------------------------------------------------- analysis

def scene_cuts(video, threshold):
    """Frame indices where the shot changes. A face track never crosses
    one: carrying a crop over a cut put the mouth mask on the wrong place."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0",
         "-vf", f"scale=320:-2,select='gt(scene,{threshold})',metadata=print:file=-",
         "-an", "-f", "null", "-"], capture_output=True, text=True).stdout
    cuts = []
    for line in out.splitlines():
        if line.startswith("frame:") and "pts_time:" in line:
            cuts.append(float(line.split("pts_time:")[1].split()[0]))
    return cuts


def pieces_of(segs, cuts, fr, total, opts):
    """Speech stretches, cut at every shot change and at max_segment, as
    [a, b) frame ranges."""
    cutf = sorted({int(round(c * fr)) for c in cuts})
    out = []
    for s, e in segs:
        a, b = int(s * fr), min(total, int(round(e * fr)))
        edges = [a] + [c for c in cutf if a < c < b] + [b]
        for x, y in zip(edges, edges[1:]):
            step = int(opts["max_segment"] * fr)
            while y - x > step * 1.5:
                out.append((x, x + step))
                x += step
            if (y - x) / fr >= opts["min_speech"]:
                out.append((x, y))
    return out


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    i = ix * iy
    return i / (aw * ah + bw * bh - i + 1e-6)


def analyze(video, opts, score, outdir, log, cancelled=lambda: False):
    """Plan a whole video: where speech is, cut into shots, and every face
    on screen in each piece, tracked, with a guess at who is speaking.

    The guess is mouth movement in the original render against the whole
    frame's movement - the speaking character's mouth already moves, a
    listener's does not. It is only the default; the site shows every face
    so the person can choose one or several per piece."""
    import cv2

    out = Path(outdir)
    (out / "thumbs").mkdir(parents=True, exist_ok=True)
    m = core.probe(video)
    fr = m["fps"]
    total = int(round(m["duration"] * fr))
    opts = {**DEFAULTS, **{k: v for k, v in (opts or {}).items() if k in DEFAULTS}}
    segs = speech_segments(video, opts, log)
    cuts = scene_cuts(video, opts["scene_cut"])
    pieces = pieces_of(segs, cuts, fr, total, opts)
    log(f"{len(cuts)} shot change(s); {len(pieces)} piece(s) to look at")

    owner = np.full(total + 1, -1)
    for k, (a, b) in enumerate(pieces):
        owner[a:b] = k
    det = core.Detector(m["w"], m["h"], score)
    sw, sh_ = det.size
    k_ = 1 / det.k
    dets = [[] for _ in pieces]               # per piece: (frame, box, act)
    prev = None
    for i, f in enumerate(core.frames(video, sw, sh_, vf=f"scale={sw}:{sh_}")):
        if i >= len(owner):
            break
        if i % 300 == 0:
            if cancelled():
                raise RuntimeError("cancelled")
            log(f"scanning faces {i}/{total}")
        pk = owner[i]
        if pk < 0:
            prev = None
            continue
        _, faces = det.net.detect(np.ascontiguousarray(f[:, :, ::-1]))
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.int16)
        g = float(np.abs(gray - prev).mean()) if prev is not None else None
        for r in (faces if faces is not None else []):
            x, y, w, h = [float(v) for v in r[:4]]
            act = None
            if g is not None and g < 40:
                x0, x1 = int(max(0, x + .2 * w)), int(min(sw, x + .8 * w))
                y0, y1 = int(max(0, y + .55 * h)), int(min(sh_, y + .95 * h))
                if x1 > x0 and y1 > y0:
                    act = float(np.abs(gray[y0:y1, x0:x1] - prev[y0:y1, x0:x1]).mean()) / (g + 1)
            dets[pk].append((i, (x * k_, y * k_, w * k_, h * k_), act))
        prev = gray

    plan = {"video": str(video), "fps": fr, "rate": m["rate"], "total": total,
            "width": m["w"], "height": m["h"], "opts": opts, "segments": []}
    for k, (a, b) in enumerate(pieces):
        tracks = []
        for i, box, act in dets[k]:
            best, bs = None, 0.3
            for t in tracks:
                if i - t["last"] > 8 or t["last"] == i:
                    continue
                lb = t["boxes"][t["last"]]
                sc = _iou(lb, box)
                cd = np.hypot(lb[0] + lb[2] / 2 - box[0] - box[2] / 2,
                              lb[1] + lb[3] / 2 - box[1] - box[3] / 2)
                if cd < .5 * max(lb[2], lb[3]):
                    sc = max(sc, .31)
                if sc > bs:
                    best, bs = t, sc
            if best is None:
                best = {"boxes": {}, "acts": []}
                tracks.append(best)
            best["boxes"][i] = box
            best["last"] = i
            if act is not None:
                best["acts"].append(act)
        n = b - a
        faces = []
        for t in tracks:
            if len(t["boxes"]) < max(8, .25 * n):
                continue
            fid = f"f{len(faces) + 1}"
            bi = max(t["boxes"], key=lambda i: t["boxes"][i][2] * t["boxes"][i][3])
            x, y, w, h = t["boxes"][bi]
            faces.append({"id": fid, "coverage": round(len(t["boxes"]) / n, 2),
                          "activity": round(float(np.median(t["acts"])), 3) if t["acts"] else 0.0,
                          "thumb_frame": bi, "thumb_box": [x, y, w, h],
                          "track": {str(i - a): [round(v, 1) for v in (bx + bw / 2, by + bh / 2, bw, bh)]
                                    for i, (bx, by, bw, bh) in t["boxes"].items()}})
        sid = f"seg{k + 1:03d}"
        pick = [max(faces, key=lambda f: f["activity"])["id"]] if faces else []
        plan["segments"].append({"id": sid, "a": a, "b": b, "start": round(a / fr, 3),
                                 "end": round(b / fr, 3), "faces": faces, "pick": pick})
        for f in faces:
            try:
                img = core.frame_at(video, f["thumb_frame"] / fr)
            except RuntimeError:
                continue
            x, y, w, h = f["thumb_box"]
            s = max(w, h) * 1.5
            cx, cy = x + w / 2, y + h / 2
            x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
            crop = img[y0:int(cy + s / 2), x0:int(cx + s / 2)]
            if crop.size:
                cv2.imwrite(str(out / "thumbs" / f"{sid}_{f['id']}.jpg"),
                            cv2.resize(crop, (160, 160))[:, :, ::-1])
    (out / "plan.json").write_text(json.dumps(plan))
    nf = sum(len(s["faces"]) for s in plan["segments"])
    log(f"plan: {len(plan['segments'])} piece(s), {nf} face track(s)")
    return plan


def runs_of(track, n, max_gap=6, min_len=8):
    """Stretches of frames where the face is on screen, as [a, b) with a
    box for every frame; short gaps (a missed detection) are interpolated,
    longer ones are left out - those frames keep the original."""
    have = sorted(int(k) for k in track)
    out, cur = [], []
    for i in have:
        if cur and i - cur[-1] > max_gap + 1:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    runs = []
    for r in out:
        a, b = r[0], r[-1] + 1
        if b - a < min_len:
            continue
        boxes = []
        for i in range(a, b):
            if str(i) in track:
                boxes.append(track[str(i)])
                continue
            lo = max(j for j in r if j < i)
            hi = min(j for j in r if j > i)
            u = (i - lo) / (hi - lo)
            boxes.append([p + (q - p) * u for p, q in zip(track[str(lo)], track[str(hi)])])
        runs.append((a, min(b, n), boxes[:min(b, n) - a]))
    return runs


def cut_clip(src, a, b, rate, fr, dst):
    core.sh(["ffmpeg", "-v", "error", "-i", str(src), "-filter_complex",
             f"[0:v:0]trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS[v];"
             f"[0:a:0]atrim={a/fr:.6f}:{b/fr:.6f},asetpts=PTS-STARTPTS[a]",
             "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "12",
             "-preset", "fast", "-pix_fmt", "yuv420p", "-r", rate,
             "-c:a", "pcm_s16le", "-y", str(dst)])


def run(video, workdir, settings, ls, log, cancelled=lambda: False,
        plan=None, picks=None):
    """Lip-sync the faces chosen in `plan` (analyze() is run when there is
    none). picks: {segment id: [face ids]} overrides the plan's own guess;
    several faces in one piece are synced one after the other.
    Returns the finished video and a per-segment report."""
    work = Path(workdir)
    m = core.probe(video)
    if plan is None:
        plan = analyze(video, settings.get("auto"), settings["score"], work / "plan",
                       log, cancelled)
    fr, total = plan["fps"], plan["total"]
    chosen = {s["id"]: (picks or {}).get(s["id"], s["pick"]) for s in plan["segments"]}
    todo_segs = [s for s in plan["segments"] if chosen[s["id"]]]
    report = [{"segment": s["id"], "start": s["start"], "end": s["end"],
               "status": "skipped", "why": "no face chosen" if s["faces"] else "no face found"}
              for s in plan["segments"] if not chosen[s["id"]]]
    todo = sum(s["b"] - s["a"] for s in todo_segs) / fr
    spent = synced = 0.0
    done = []
    log(f"syncing {len(todo_segs)} of {len(plan['segments'])} piece(s)")
    for k, sg in enumerate(todo_segs, 1):
        if cancelled():
            raise RuntimeError("cancelled")
        a, b, sid = sg["a"], sg["b"], sg["id"]
        d = work / sid
        d.mkdir(parents=True, exist_ok=True)
        eta = ""
        if synced:
            left = (todo - synced) * spent / synced
            eta = f" - about {left/60:.0f} min left" if left >= 60 else f" - about {left:.0f}s left"
        faces = {f["id"]: f for f in sg["faces"]}
        names = [x for x in chosen[sid] if x in faces]
        log(f"[{k}/{len(todo_segs)}] {sid} {a/fr:.2f}-{b/fr:.2f}s, face(s) {', '.join(names)}{eta}")
        t0 = time.time()
        rec = {"segment": sid, "start": sg["start"], "end": sg["end"], "faces": names}
        cur = d / "source.mp4"
        cut_clip(video, a, b, m["rate"], fr, cur)
        n = b - a
        ok_faces, errors = [], []
        for fid in names:
            runs = runs_of(faces[fid]["track"], n)
            if not runs:
                errors.append(f"{fid}: on screen too briefly")
                continue
            got = []
            for j, (ra, rb, boxes) in enumerate(runs):
                rc = d / f"{fid}_run{j}.mp4"
                cut_clip(cur, ra, rb, m["rate"], fr, rc)
                try:
                    o = core.lipsync(rc, None, None, "cut", settings["engine"],
                                     settings["steps"], settings["guidance"],
                                     settings["seed"], settings["crop_max"],
                                     settings["score"], ls,
                                     log=lambda s: log("    " + s),
                                     run=d / f"{fid}_run{j}", track_=boxes)
                    c = count_frames(o)
                    if abs(c - (rb - ra)) > 2:
                        raise RuntimeError(f"got {c} frames, expected {rb - ra}")
                    got.append((ra, rb, o))
                except Exception as e:
                    log(f"    {fid} {ra}-{rb}: FAILED {e}")
                    errors.append(f"{fid} frames {ra}-{rb}: {e}")
            if got:
                nxt = d / f"after_{fid}.mp4"
                splice(cur, m, n, got, nxt)
                cur = nxt
                ok_faces.append(fid)
        if ok_faces:
            done.append((a, b, cur))
            rec.update(status="synced" if not errors else "partly synced",
                       seconds=round(time.time() - t0, 1))
        else:
            rec.update(status="failed")
        if errors:
            rec["why"] = "; ".join(errors)[:400]
        report.append(rec)
        spent += time.time() - t0
        synced += n / fr
    report.sort(key=lambda r: r["start"])
    if todo_segs and not done:
        # nothing synced: say so instead of returning the original as "done"
        why = next((r.get("why") for r in report if r["status"] == "failed"), "")
        raise RuntimeError(f"no part could be lip-synced - {why}"[:600])
    dst = work / "lipsynced.mp4"
    splice(video, m, total, done, dst)
    log(f"spliced {len(done)} synced piece(s) into the full video")
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
