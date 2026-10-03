"""Motion library: every action the animators finish becomes a reusable clip.

Clips are grouped by rig signature (same character rig => same signature), are
searchable by the words in their action and file names, and carry timing and
spacing statistics that describe the studio's animation style.
"""

import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict

from .events import KeyState

STOP_WORDS = {"action", "anim", "final", "new", "copy", "test", "blend", "shot", "sh", "sc", "scene",
              "the", "a", "an", "and", "to", "of", "with", "ka", "ki", "ke", "se", "aur", "fir", "phir",
              "then", "v", "ver", "wip", "ok", "armature", "rig", "unsaved"}


def stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    words = re.split(r"[^A-Za-z]+", text.lower())
    return [stem(w) for w in words if len(w) > 1 and w not in STOP_WORDS]


def clip_id(animator, file, action):
    return hashlib.sha1(("%s|%s|%s" % (animator, file, action)).encode()).hexdigest()[:16]


def clip_stats(fcurves):
    """Timing/spacing numbers that describe how a clip was animated."""
    intervals, ease, holds = [], [], 0
    interp = Counter()
    key_types = Counter()
    frames = set()
    peaks = 0
    for fc in fcurves:
        keys = fc["keys"]
        for i, key in enumerate(keys):
            frames.add(key[0])
            interp[key[2]] += 1
            key_types[key[3]] += 1
            if i:
                gap = key[0] - keys[i - 1][0]
                if gap > 0:
                    intervals.append(gap)
                    if abs(key[1] - keys[i - 1][1]) < 1e-5:
                        holds += 1
                    if key[2] == "BEZIER" or keys[i - 1][2] == "BEZIER":
                        ease.append(min((key[0] - key[4]) / gap, 1.0))
            if 0 < i < len(keys) - 1:
                before, after = keys[i - 1][1], keys[i + 1][1]
                if (key[1] - before) * (key[1] - after) > 0:
                    peaks += 1  # local extreme: a pose the motion swings through
    total = sum(interp.values()) or 1
    return {
        "frame_start": min(frames) if frames else 0,
        "frame_end": max(frames) if frames else 0,
        "keys": total if interp else 0,
        "key_frames": len(frames),
        "median_interval": statistics.median(intervals) if intervals else 0,
        "interpolation": {k: round(v / total, 3) for k, v in interp.items()},
        "breakdown_ratio": round(1 - key_types.get("KEYFRAME", 0) / total, 3) if interp else 0,
        "hold_ratio": round(holds / max(len(intervals), 1), 3),
        "ease": round(statistics.median(ease), 3) if ease else 0,
        "peaks": peaks,
    }


class Library:
    def __init__(self, data_root):
        self.root = os.path.join(data_root, "motion")
        self.index_path = os.path.join(self.root, "index.json")
        try:
            with open(self.index_path, encoding="utf-8") as handle:
                self.index = json.load(handle)
        except (OSError, ValueError):
            self.index = {"clips": {}, "rigs": {}}

    def clip_path(self, cid):
        return os.path.join(self.root, "clips", cid + ".json")

    def rig_path(self, sig):
        return os.path.join(self.root, "rigs", sig + ".json")

    def load_clip(self, cid):
        with open(self.clip_path(cid), encoding="utf-8") as handle:
            return json.load(handle)

    def load_rig(self, sig):
        try:
            with open(self.rig_path(sig), encoding="utf-8") as handle:
                return json.load(handle)
        except OSError:
            return None

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.index, handle, indent=1, sort_keys=True)
        os.replace(tmp, self.index_path)

    def _write(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))

    def add_rig(self, event):
        sig = event["sig"]
        entry = self.index["rigs"].setdefault(sig, {"names": [], "bones": len(event.get("bones", ()))})
        if event.get("object") and event["object"] not in entry["names"]:
            entry["names"].append(event["object"])
        if not os.path.exists(self.rig_path(sig)):
            self._write(self.rig_path(sig), {"sig": sig, "bones": event.get("bones", [])})

    def ingest_sessions(self, sessions, day):
        """Replay a day's sessions and store the latest state of each action."""
        latest = {}
        for (animator, _sid), events in sorted(sessions.items(), key=lambda kv: kv[1][0]["t"]):
            state = KeyState()
            for event in events:
                if event["type"] == "rig" and event.get("sig"):
                    self.add_rig(event)
                state.apply(event)
            for (file, action), entry in state.actions.items():
                latest[(animator, file, action)] = entry
        updated = []
        for (animator, file, action), entry in latest.items():
            fcurves = [fc for fc in entry["fcurves"].values() if fc.get("keys")]
            if not fcurves or max(len(fc["keys"]) for fc in fcurves) < 2:
                continue
            owners = entry.get("owners") or []
            rig = next((o["rig"] for o in owners if o.get("rig")), "")
            cid = clip_id(animator, file, action)
            last_t = entry.get("last_t", 0)
            if self.index["clips"].get(cid, {}).get("last_t", 0) > last_t:
                continue  # a newer version from a later day is already stored
            stats = clip_stats(fcurves)
            base = os.path.basename(file.replace("\\", "/"))
            meta = {
                "animator": animator, "file": file, "action": action, "rig": rig,
                "owner_type": owners[0].get("type", "") if owners else "",
                "fps": entry.get("fps", 24.0), "updated": day, "last_t": last_t,
                "tags": sorted(set(tokenize(action) + tokenize(os.path.splitext(base)[0]))),
                "bones": sorted({fc["bone"] for fc in fcurves if fc.get("bone")}),
                "stats": stats,
            }
            self._write(self.clip_path(cid), dict(meta, id=cid, fcurves=fcurves))
            self.index["clips"][cid] = meta
            updated.append(cid)
        self.save()
        return updated

    # -- search --------------------------------------------------------
    def search(self, query, rig=None, limit=5):
        """Rank clips for a text query (tf-idf over clip tags)."""
        words = tokenize(query)
        clips = {cid: meta for cid, meta in self.index["clips"].items() if not rig or meta["rig"] == rig}
        if not clips or not words:
            return []
        doc_freq = Counter(tag for meta in clips.values() for tag in set(meta["tags"]))
        n = len(clips)
        results = []
        for cid, meta in clips.items():
            tags = set(meta["tags"])
            score = 0.0
            for word in words:
                if word in tags:
                    score += 1.0 + (n / doc_freq[word]) ** 0.5
                elif any(t.startswith(word) or word.startswith(t) for t in tags if len(t) >= 3 and len(word) >= 3):
                    score += 0.5
            if score:
                score += min(meta["stats"]["key_frames"], 50) / 500.0  # prefer fuller clips on ties
                results.append((round(score, 4), cid))
        results.sort(key=lambda r: (-r[0], r[1]))
        return results[:limit]


def style_profile(index):
    """Studio animation style, overall and per rig, from all clips."""
    groups = defaultdict(list)
    for meta in index["clips"].values():
        groups["*"].append(meta["stats"])
        if meta["rig"]:
            groups[meta["rig"]].append(meta["stats"])
    profile = {}
    for key, stats in groups.items():
        def med(name):
            values = [s[name] for s in stats if s.get(name) is not None]
            return round(statistics.median(values), 3) if values else 0

        interp = Counter()
        for s in stats:
            interp.update(s["interpolation"])
        total = sum(interp.values()) or 1
        profile[key] = {
            "clips": len(stats),
            "median_interval": med("median_interval"),
            "breakdown_ratio": med("breakdown_ratio"),
            "hold_ratio": med("hold_ratio"),
            "ease": med("ease"),
            "interpolation": {k: round(v / total, 3) for k, v in interp.items()},
        }
    return profile
