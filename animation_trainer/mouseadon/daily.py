"""The daily job: learn from everything recorded so far and rebuild the pipeline.

    mouseadon daily --data /srv/mouseadon

Run it once a day (cron). It
  * processes new/changed days of raw logs (cached per day, so it stays fast
    with 100 animators x 60 days),
  * updates the motion library with the latest version of every action,
  * rebuilds ``brain/pipeline.json`` (the learned workflow) and
    ``brain/style.json`` (timing/spacing style), keeping a dated history,
  * writes a Markdown report to ``reports/<day>.md``.
"""

import datetime
import json
import os
from collections import Counter

from . import events, motion, workflow

DEFAULT_TRAINING_DAYS = 45


def _paths(data_root):
    return {
        "days": os.path.join(data_root, "work", "days"),
        "brain": os.path.join(data_root, "brain"),
        "history": os.path.join(data_root, "brain", "history"),
        "reports": os.path.join(data_root, "reports"),
        "config": os.path.join(data_root, "config.json"),
    }


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _write_json(path, payload, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True, default=list)
    os.replace(tmp, path)


def load_config(data_root):
    return _read_json(_paths(data_root)["config"], {}) or {}


def _raw_mtime(data_root, day):
    newest = 0.0
    root = events.raw_dir(data_root)
    for animator in os.listdir(root):
        day_dir = os.path.join(root, animator, day)
        if os.path.isdir(day_dir):
            for name in os.listdir(day_dir):
                newest = max(newest, os.path.getmtime(os.path.join(day_dir, name)))
    return newest


def process_day(data_root, day, library):
    sessions = events.load_day(data_root, day)
    windows = []
    counts = Counter()
    for (animator, _sid), session_events in sessions.items():
        windows.extend(workflow.session_windows(animator, session_events))
        counts.update(e["type"] for e in session_events)
    clips = library.ingest_sessions(sessions, day)
    summary = {
        "day": day,
        "sessions": len(sessions),
        "event_counts": dict(counts),
        "animators": workflow.animator_stats(windows, sessions),
        "clips_updated": clips,
        "windows": workflow.windows_to_json(windows),
    }
    _write_json(os.path.join(_paths(data_root)["days"], day + ".json"), summary, indent=None)
    return summary


def readiness(data_root, days, pipeline, library, config):
    target = int(config.get("training_days", DEFAULT_TRAINING_DAYS))
    clips = library.index["clips"]
    rig_clips = Counter(meta["rig"] for meta in clips.values() if meta["rig"])
    full_shots = 0
    for item in pipeline.get("common_sequences", []):
        if len(item["sequence"].split(" > ")) >= 3:
            full_shots += item["shots"]
    checks = {
        "training_days": {"have": len(days), "need": target},
        "clips_on_best_rig": {"have": max(rig_clips.values()) if rig_clips else 0, "need": 50},
        "shots_seen": {"have": pipeline.get("shots", 0), "need": 100},
    }
    ready = all(c["have"] >= c["need"] for c in checks.values())
    return {"ready": ready, "checks": checks, "rigs": dict(rig_clips.most_common(10))}


def _diff_pipelines(old, new):
    notes = []
    if not old:
        return ["First version of the pipeline."]
    if old.get("order") != new.get("order"):
        notes.append("Phase order changed: %s -> %s" % (" > ".join(old.get("order", [])), " > ".join(new["order"])))
    old_sig = {p["phase"]: {o["id"] for o in p["signature_ops"]} for p in old.get("phases", [])}
    for phase in new["phases"]:
        added = [o["id"] for o in phase["signature_ops"] if o["id"] not in old_sig.get(phase["phase"], set())]
        if added:
            notes.append("%s: new characteristic tools %s" % (phase["phase"], ", ".join(added[:4])))
    return notes or ["No structural change; statistics refined with new data."]


def _previous_pipeline(history_dir, day):
    if not os.path.isdir(history_dir):
        return None
    older = sorted(n for n in os.listdir(history_dir) if n.startswith("pipeline-") and n[9:19] < day)
    return _read_json(os.path.join(history_dir, older[-1])) if older else None


def render_report(day, today_summary, pipeline, style, ready, changes, library):
    lines = ["# Mouseadon daily report - %s" % day, ""]
    animators = today_summary["animators"] if today_summary else {}
    lines += ["## Today", "",
              "- Animators recorded: **%d**" % len(animators),
              "- Sessions: **%d**" % (today_summary["sessions"] if today_summary else 0),
              "- Active hours: **%.1f**" % (sum(a["active_minutes"] for a in animators.values()) / 60.0),
              "- Clips added/updated in the motion library: **%d**" % (len(today_summary["clips_updated"]) if today_summary else 0),
              ""]
    if animators:
        lines += ["| Animator | Active min | Operators | Key edits | Shots | Mostly doing | Main editor |",
                  "|---|---:|---:|---:|---:|---|---|"]
        for name, a in sorted(animators.items()):
            lines.append("| %s | %.0f | %d | %d | %d | %s | %s |" % (
                name, a["active_minutes"], a["ops"], a["key_edits"], a["shots"], a["main_phase"], a["main_editor"]))
        lines.append("")
    lines += ["## Learned pipeline", "",
              "Order: **%s** (from %d shots)" % (" > ".join(pipeline["order"]) or "not enough data", pipeline["shots"]), ""]
    for phase in pipeline["phases"]:
        tools = ", ".join((o["name"] or o["id"]) for o in phase["signature_ops"][:5]) or "-"
        keys = ", ".join(h["key"] for h in phase["hotkeys"][:5]) or "-"
        editors = ", ".join("%s %d%%" % (k, v * 100) for k, v in list(phase["editors"].items())[:3]) or "-"
        lines += ["### %s" % phase["phase"].title(),
                  "- In %d%% of shots, median %.0f min per shot" % (phase["shots_with_phase"] * 100, phase["median_minutes"]),
                  "- Characteristic tools: %s" % tools,
                  "- Hotkeys: %s" % keys,
                  "- Editors: %s" % editors, ""]
    lines += ["## What changed since the last version", ""] + ["- " + n for n in changes] + [""]
    overall = style.get("*")
    if overall:
        lines += ["## Studio style", "",
                  "- Median frames between keys: %s" % overall["median_interval"],
                  "- Breakdown ratio: %s, holds: %s, ease: %s" % (overall["breakdown_ratio"], overall["hold_ratio"], overall["ease"]),
                  "- Interpolation mix: %s" % ", ".join("%s %d%%" % (k, v * 100) for k, v in overall["interpolation"].items()),
                  ""]
    lines += ["## Training readiness", ""]
    for name, check in ready["checks"].items():
        lines.append("- %s: %s / %s %s" % (name.replace("_", " "), check["have"], check["need"],
                                           "OK" if check["have"] >= check["need"] else ""))
    lines += ["", "**%s**" % ("Ready to generate on its own." if ready["ready"] else "Still training."), ""]
    lines.append("Motion library: %d clips across %d rigs." % (len(library.index["clips"]), len(library.index["rigs"])))
    return "\n".join(lines) + "\n"


def run_daily(data_root, day=None, force=False):
    paths = _paths(data_root)
    day = day or datetime.date.today().isoformat()
    config = load_config(data_root)
    library = motion.Library(data_root)
    all_days = [d for d in events.list_days(data_root) if d <= day]

    summaries = {}
    for d in all_days:
        cache = os.path.join(paths["days"], d + ".json")
        stale = (not os.path.exists(cache) or force
                 or _raw_mtime(data_root, d) > os.path.getmtime(cache))
        summaries[d] = process_day(data_root, d, library) if stale else _read_json(cache)

    windows = []
    for d in all_days:
        windows.extend(workflow.windows_from_json(summaries[d]["windows"]))
    pipeline = workflow.build_pipeline(windows)
    pipeline["built"] = day
    pipeline["days"] = len(all_days)
    style = motion.style_profile(library.index)

    changes = _diff_pipelines(_previous_pipeline(paths["history"], day), pipeline)
    ready = readiness(data_root, all_days, pipeline, library, config)
    pipeline["readiness"] = ready

    _write_json(os.path.join(paths["brain"], "pipeline.json"), pipeline)
    _write_json(os.path.join(paths["brain"], "style.json"), style)
    _write_json(os.path.join(paths["history"], "pipeline-%s.json" % day), pipeline)

    report = render_report(day, summaries.get(day), pipeline, style, ready, changes, library)
    os.makedirs(paths["reports"], exist_ok=True)
    report_path = os.path.join(paths["reports"], day + ".md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    return report_path, pipeline
