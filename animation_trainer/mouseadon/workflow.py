"""Workflow mining: how do the animators actually work?

1. Each session is cut into short windows (default 5 minutes).
2. Each window gets features (what kind of operators, which editors, how
   much of the animation is still stepped/CONSTANT, breakdowns, playback).
3. A transparent rule set labels each window with a phase
   (layout / blocking / splining / polish / review).
4. Across every shot of every animator the phases are put in the order
   animators actually use them, with typical durations, the operators and
   hotkeys that characterise each phase, and next-operator statistics.

The result (``pipeline.json``) is the studio pipeline the tool has learned,
rebuilt every day as more data arrives.
"""

import math
import statistics
from collections import Counter, defaultdict

from .events import KeyState
from .schema import PHASES, categorize, hotkey_label

WINDOW_SECONDS = 300
MIN_ACTIVITY = 3


def session_windows(animator, events, window_seconds=WINDOW_SECONDS):
    """Cut one session into feature windows."""
    if not events:
        return []
    state = KeyState()
    windows = []
    current_file = "<unsaved>"
    mode = ""
    start = events[0]["t"]
    window = None

    def new_window(index, file):
        return {
            "animator": animator, "sid": events[0].get("sid", ""), "file": file,
            "start": start + index * window_seconds, "end": start + (index + 1) * window_seconds,
            "cats": Counter(), "ops": Counter(), "op_names": {}, "bigrams": Counter(),
            "hotkeys": Counter(), "areas": Counter(), "key_events": 0, "curves_changed": 0,
            "frames": 0, "plays": 0, "clicks": 0, "last_op": None, "index": index,
        }

    def close(win):
        if win is None:
            return
        win.update(state.file_stats(win["file"]))
        del win["last_op"]
        windows.append(win)

    for event in events:
        index = int((event["t"] - start) // window_seconds)
        kind = event["type"]
        if kind == "file":
            current_file = event.get("file", current_file)
        if window is None or index != window["index"] or window["file"] != current_file:
            close(window)
            window = new_window(index, current_file)
        state.apply(event)
        if kind == "mode":
            mode = event.get("mode", "")
        elif kind == "op":
            op_id = event.get("id", "?")
            window["ops"][op_id] += 1
            window["cats"][categorize(op_id, event.get("area", ""), mode)] += 1
            if event.get("name"):
                window["op_names"][op_id] = event["name"]
            if window["last_op"]:
                window["bigrams"][window["last_op"] + ">" + op_id] += 1
            window["last_op"] = op_id
        elif kind == "key":
            window["hotkeys"][hotkey_label(event)] += 1
        elif kind == "click":
            window["clicks"] += 1
            if event.get("area"):
                window["areas"][event["area"]] += 3
        elif kind == "mouse":
            for area, value in event.get("areas", {}).items():
                window["areas"][area] += value.get("n", 0) / 10.0
        elif kind == "keys":
            window["key_events"] += 1
            window["curves_changed"] += len(event.get("fcurves", ()))
        elif kind == "frame":
            window["frames"] += 1
        elif kind == "play" and event.get("on"):
            window["plays"] += 1
    close(window)
    for win in windows:
        win["phase"] = classify(win)
    return windows


def classify(win):
    """Label a window with a phase. Rules are deliberately simple and visible."""
    cats = win["cats"]
    edits = sum(cats[c] for c in ("pose", "key", "timing", "curves")) + win["key_events"]
    total_ops = sum(cats.values())
    watching = win["plays"] * 3 + win["frames"] / 4.0
    if edits + total_ops + watching < MIN_ACTIVITY:
        return "idle"
    keys = win.get("keys", 0)
    constant = win.get("constant", 0) / keys if keys else 0.0
    areas = win["areas"]
    area_total = sum(areas.values()) or 1.0
    graph_share = areas.get("GRAPH_EDITOR", 0) / area_total
    if cats["scene"] + cats["rig"] > max(edits, 1) and keys < 20:
        return "layout"
    if watching > 2 * max(edits, 1):
        return "review"
    if keys and constant >= 0.6:
        return "blocking"
    curve_work = cats["curves"] / max(edits, 1)
    if graph_share >= 0.4 or curve_work >= 0.35:
        return "polish"
    if keys:
        return "splining"
    return "layout"


def _collapse(sequence):
    out = []
    for item in sequence:
        if not out or out[-1] != item:
            out.append(item)
    return out


def build_pipeline(windows, top_n=8):
    """Aggregate windows from every animator into the learned pipeline."""
    shots = defaultdict(list)
    for win in windows:
        if win["phase"] != "idle":
            shots[(win["animator"], win["file"])].append(win)

    positions = defaultdict(list)
    durations = defaultdict(list)
    transitions = Counter()
    sequences = Counter()
    for shot_windows in shots.values():
        shot_windows.sort(key=lambda w: w["start"])
        seq = _collapse(w["phase"] for w in shot_windows)
        sequences[" > ".join(seq)] += 1
        per_phase = Counter()
        for w in shot_windows:
            per_phase[w["phase"]] += (w["end"] - w["start"]) / 60.0
        for phase, minutes in per_phase.items():
            durations[phase].append(minutes)
            positions[phase].append(seq.index(phase) / max(len(seq) - 1, 1))
        for a, b in zip(seq, seq[1:]):
            transitions[(a, b)] += 1

    ops_by_phase = defaultdict(Counter)
    hotkeys_by_phase = defaultdict(Counter)
    areas_by_phase = defaultdict(Counter)
    bigrams_by_phase = defaultdict(Counter)
    op_names = {}
    all_ops = Counter()
    for win in windows:
        if win["phase"] == "idle":
            continue
        ops_by_phase[win["phase"]].update(win["ops"])
        hotkeys_by_phase[win["phase"]].update(win["hotkeys"])
        areas_by_phase[win["phase"]].update(win["areas"])
        bigrams_by_phase[win["phase"]].update(win["bigrams"])
        all_ops.update(win["ops"])
        op_names.update(win["op_names"])

    total_all = sum(all_ops.values()) or 1
    n_shots = len(shots) or 1
    phases = []
    for phase in sorted(positions, key=lambda p: (statistics.median(positions[p]), PHASES.index(p) if p in PHASES else 99)):
        ops = ops_by_phase[phase]
        total = sum(ops.values()) or 1
        signature = []
        for op_id, count in ops.items():
            lift = (count / total) / (all_ops[op_id] / total_all)
            signature.append((lift * math.log1p(count), op_id, count, lift))
        signature.sort(reverse=True)
        next_ops = defaultdict(Counter)
        for pair, count in bigrams_by_phase[phase].items():
            a, b = pair.split(">", 1)
            next_ops[a][b] += count
        area_total = sum(areas_by_phase[phase].values()) or 1
        outgoing = {b: c for (a, b), c in transitions.items() if a == phase}
        out_total = sum(outgoing.values()) or 1
        phases.append({
            "phase": phase,
            "shots_with_phase": round(len(durations[phase]) / n_shots, 3),
            "median_minutes": round(statistics.median(durations[phase]), 1),
            "median_position": round(statistics.median(positions[phase]), 3),
            "signature_ops": [
                {"id": op_id, "name": op_names.get(op_id, ""), "count": count, "lift": round(lift, 2)}
                for _, op_id, count, lift in signature[:top_n]
            ],
            "top_ops": [{"id": o, "name": op_names.get(o, ""), "count": c} for o, c in ops.most_common(top_n)],
            "hotkeys": [{"key": k, "count": c} for k, c in hotkeys_by_phase[phase].most_common(top_n)],
            "editors": {a: round(v / area_total, 3) for a, v in areas_by_phase[phase].most_common(5)},
            "next_op": {a: b.most_common(1)[0][0] for a, b in next_ops.items() if sum(b.values()) >= 3},
            "goes_to": {b: round(c / out_total, 3) for b, c in sorted(outgoing.items(), key=lambda x: -x[1])},
        })
    return {
        "shots": len(shots),
        "windows": sum(1 for w in windows if w["phase"] != "idle"),
        "order": [p["phase"] for p in phases],
        "phases": phases,
        "common_sequences": [{"sequence": s, "shots": c} for s, c in sequences.most_common(5)],
    }


def animator_stats(windows, sessions):
    """Per-animator activity for the daily report."""
    stats = defaultdict(lambda: {"active_minutes": 0.0, "ops": 0, "key_edits": 0, "shots": set(),
                                 "phases": Counter(), "editors": Counter(), "sessions": 0})
    for win in windows:
        if win["phase"] == "idle":
            continue
        entry = stats[win["animator"]]
        entry["active_minutes"] += (win["end"] - win["start"]) / 60.0
        entry["ops"] += sum(win["ops"].values())
        entry["key_edits"] += win["key_events"]
        entry["shots"].add(win["file"])
        entry["phases"][win["phase"]] += 1
        entry["editors"].update(win["areas"])
    for animator, _sid in sessions:
        stats[animator]["sessions"] += 1
    result = {}
    for animator, entry in stats.items():
        result[animator] = {
            "sessions": entry["sessions"],
            "active_minutes": round(entry["active_minutes"], 1),
            "ops": entry["ops"],
            "key_edits": entry["key_edits"],
            "shots": len(entry["shots"]),
            "main_phase": entry["phases"].most_common(1)[0][0] if entry["phases"] else "",
            "main_editor": entry["editors"].most_common(1)[0][0] if entry["editors"] else "",
        }
    return result


def windows_to_json(windows):
    """Counters -> plain dicts so day caches can be stored as JSON."""
    out = []
    for win in windows:
        item = dict(win)
        for key in ("cats", "ops", "bigrams", "hotkeys", "areas"):
            item[key] = dict(win[key])
        out.append(item)
    return out


def windows_from_json(items):
    for item in items:
        for key in ("cats", "ops", "bigrams", "hotkeys", "areas"):
            item[key] = Counter(item.get(key, {}))
    return items
