"""Reading raw event logs and replaying animation state."""

import json
import os
import re

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def raw_dir(data_root):
    return os.path.join(data_root, "raw")


def list_days(data_root):
    root = raw_dir(data_root)
    days = set()
    if os.path.isdir(root):
        for animator in os.listdir(root):
            path = os.path.join(root, animator)
            if os.path.isdir(path):
                days.update(d for d in os.listdir(path) if DAY_RE.match(d))
    return sorted(days)


def read_jsonl(path):
    events = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue  # torn line after a crash; skip it
            if isinstance(event, dict) and "type" in event and "t" in event:
                events.append(event)
    return events


def load_day(data_root, day):
    """Return ``{(animator, session_id): [events sorted by time]}`` for a day."""
    sessions = {}
    root = raw_dir(data_root)
    if not os.path.isdir(root):
        return sessions
    for animator in sorted(os.listdir(root)):
        day_dir = os.path.join(root, animator, day)
        if not os.path.isdir(day_dir):
            continue
        for name in sorted(os.listdir(day_dir)):
            if name.endswith(".jsonl"):
                events = read_jsonl(os.path.join(day_dir, name))
                if events:
                    events.sort(key=lambda e: e["t"])
                    sessions[(animator, name[:-6])] = events
    return sessions


class KeyState:
    """Current F-curve contents of every action, rebuilt from ``keys`` events.

    Each ``keys`` event carries the full new state of the curves that changed,
    and every file load starts with a full baseline, so replaying events in
    order reproduces exactly what the animator had in Blender.
    """

    def __init__(self):
        self.actions = {}  # (file, action) -> {"fcurves": {key: fcurve}, "owners": [...], "fps": float}

    @staticmethod
    def curve_key(fcurve):
        return "%s|%s|%d" % (fcurve.get("slot", ""), fcurve["path"], fcurve["idx"])

    def apply(self, event):
        kind = event["type"]
        if kind == "file":
            for key in [k for k in self.actions if k[0] == event.get("file")]:
                del self.actions[key]
        elif kind == "keys":
            entry = self.actions.setdefault((event.get("file", ""), event["action"]),
                                            {"fcurves": {}, "owners": [], "fps": 24.0})
            if event.get("owners"):
                entry["owners"] = event["owners"]
            entry["fps"] = event.get("fps", entry["fps"])
            entry["last_t"] = event["t"]
            for key in event.get("removed", ()):
                entry["fcurves"].pop(key, None)
            for fcurve in event.get("fcurves", ()):
                entry["fcurves"][self.curve_key(fcurve)] = fcurve
        elif kind == "action_removed":
            self.actions.pop((event.get("file", ""), event["action"]), None)

    def file_stats(self, file):
        """Key counts for one file, used to recognise the pipeline phase."""
        total = constant = breakdown = 0
        frames = set()
        for (path, _action), entry in self.actions.items():
            if path != file:
                continue
            for fcurve in entry["fcurves"].values():
                for key in fcurve["keys"]:
                    total += 1
                    frames.add(key[0])
                    constant += key[2] == "CONSTANT"
                    breakdown += key[3] != "KEYFRAME"
        return {"keys": total, "constant": constant, "breakdown": breakdown, "frames": len(frames)}
