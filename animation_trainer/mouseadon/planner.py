"""Turn a short brief into a shot plan the Blender generator can execute.

    mouseadon plan --data /srv/mouseadon --brief "walk to the table, then pick up the cup" --out plan.json

The brief is split into beats ("then", "fir", "phir", commas, full stops).
Each beat is matched against the motion library, clips are laid out on the
timeline, and the learned pipeline + style are attached so the generator
works the way the animators do: layout > blocking > splining > polish > render.
"""

import json
import os
import re
import statistics
from collections import Counter, defaultdict

from .motion import Library

BEAT_SPLIT = re.compile(
    r"\s*(?:[.;\n,]|\b(?:and then|then|after that|aur fir|aur phir|fir|phir|uske baad|uske bad)\b)\s*",
    re.IGNORECASE,
)
EXECUTABLE = ("layout", "blocking", "splining", "polish", "review")


def split_beats(brief):
    return [part.strip() for part in BEAT_SPLIT.split(brief or "") if part and part.strip()]


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def choose_rig(library, beats):
    scores = defaultdict(float)
    for beat in beats:
        best = {}
        for score, cid in library.search(beat, limit=50):
            rig = library.index["clips"][cid]["rig"]
            if rig:
                best[rig] = max(best.get(rig, 0.0), score)
        for rig, score in best.items():
            scores[rig] += score
    if scores:
        return max(scores, key=lambda r: (scores[r], r))
    counts = Counter(m["rig"] for m in library.index["clips"].values() if m["rig"])
    return counts.most_common(1)[0][0] if counts else None


def _fallback_clip(library, rig, used):
    candidates = [(m["stats"]["key_frames"], cid) for cid, m in library.index["clips"].items()
                  if m["rig"] == rig and cid not in used]
    if not candidates:
        candidates = [(m["stats"]["key_frames"], cid) for cid, m in library.index["clips"].items() if m["rig"] == rig]
    return max(candidates)[1] if candidates else None


def generator_steps(pipeline, min_shots=10, min_share=0.15):
    """Which pipeline steps the generator runs, decided from the learned pipeline.

    Dependencies are fixed (you cannot spline before blocking). Optional
    phases are skipped when the studio rarely does them - e.g. a studio that
    keeps stepped/CONSTANT animation never splines, so neither does the tool.
    """
    pipeline = pipeline or {}
    learned = [p for p in pipeline.get("order", []) if p in EXECUTABLE]
    shares = {p["phase"]: p["shots_with_phase"] for p in pipeline.get("phases", [])}
    steps = []
    for phase in EXECUTABLE:
        optional = phase in ("splining", "polish")
        if optional and pipeline.get("shots", 0) >= min_shots and shares.get(phase, 0) < min_share:
            continue
        steps.append(phase)
    return steps, learned


def make_plan(data_root, brief, rig=None, fps=None, width=1280, height=720, gap=None):
    library = Library(data_root)
    if not library.index["clips"]:
        raise SystemExit("Motion library is empty - run `mouseadon daily` after animators have recorded work.")
    pipeline = _read_json(os.path.join(data_root, "brain", "pipeline.json"), {})
    style_all = _read_json(os.path.join(data_root, "brain", "style.json"), {})
    beats_text = split_beats(brief) or [brief]
    rig = rig or choose_rig(library, beats_text)
    if rig is None:
        raise SystemExit("No rigged clips in the motion library yet.")
    style = style_all.get(rig) or style_all.get("*") or {}

    rig_clips = [m for m in library.index["clips"].values() if m["rig"] == rig]
    fps = fps or (statistics.median(m["fps"] for m in rig_clips) if rig_clips else 24.0)
    if gap is None:
        gap = int(min(max(round(style.get("median_interval", 6) or 6), 4), 12))

    beats = []
    cursor = 1
    used = set()
    for text in beats_text:
        results = library.search(text, rig=rig)
        matched = bool(results)
        cid = results[0][1] if results else _fallback_clip(library, rig, used)
        if cid is None:
            continue
        used.add(cid)
        clip = library.load_clip(cid)
        scale = fps / (clip.get("fps") or fps)
        first = clip["stats"]["frame_start"]
        fcurves = []
        for fc in clip["fcurves"]:
            keys = []
            for key in fc["keys"]:
                key = list(key)
                for i in (0, 4, 6):
                    key[i] = round((key[i] - first) * scale, 4)
                keys.append(key)
            fcurves.append(dict(fc, keys=keys))
        length = max(1, round((clip["stats"]["frame_end"] - first) * scale))
        beats.append({
            "text": text, "matched": matched, "score": results[0][0] if results else 0.0,
            "clip": cid, "action": clip["action"], "animator": clip["animator"], "file": clip["file"],
            "start": cursor, "end": cursor + length, "fcurves": fcurves,
        })
        cursor += length + gap

    steps, learned = generator_steps(pipeline)
    rig_data = library.load_rig(rig) or {"sig": rig, "bones": []}
    return {
        "version": 1,
        "brief": brief,
        "fps": fps,
        "frame_start": 1,
        "frame_end": (beats[-1]["end"] + gap) if beats else 1,
        "resolution": [width, height],
        "gap": gap,
        "rig": rig_data,
        "steps": steps,
        "learned_order": learned,
        "style": style,
        "beats": beats,
    }


def save_plan(plan, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=1)
