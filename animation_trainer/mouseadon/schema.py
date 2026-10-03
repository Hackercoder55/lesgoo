"""Shared vocabulary: event fields, operator categories and pipeline phases."""

import re

SCHEMA_VERSION = 1

CATEGORIES = ("pose", "key", "timing", "curves", "playback", "view", "select", "rig", "scene", "file", "other")

# Phases the daily pipeline can recognise. The *order* and *content* of the
# pipeline are learned from data; these are just the labels.
PHASES = ("layout", "blocking", "splining", "polish", "review")

TIME_EDITORS = {"DOPESHEET_EDITOR", "TIMELINE", "NLA_EDITOR"}

_RULES = [
    ("file", re.compile(r"^WM_OT_(save|open|read|link|append|recover|revert)")),
    ("playback", re.compile(r"^(SCREEN_OT_(animation_|frame_|keyframe_jump|marker_jump)|ANIM_OT_(change_frame|frame_))")),
    ("select", re.compile(r"_OT_\w*select(_|$)|_OT_(lasso|circle)$")),
    ("view", re.compile(r"^(VIEW3D_OT_(view|zoom|rotate|move|dolly|walk|fly|localview|camera)|VIEW2D_OT_|.*_OT_view_(all|selected|frame))")),
    ("key", re.compile(r"^ANIM_OT_(keyframe_|keying_set|keyframe_insert)|_OT_keyframe_(insert|delete)|^(ACTION|GRAPH)_OT_(copy|paste|delete|duplicate|duplicate_move)$|^POSE_OT_(copy|paste)$")),
    ("curves", re.compile(r"^GRAPH_OT_|_OT_(interpolation_type|handle_type|easing_type|euler_filter|smooth|clean|bake_keys|decimate)")),
    ("timing", re.compile(r"^(ACTION|NLA|MARKER)_OT_|_OT_keyframe_type$")),
    ("rig", re.compile(r"^(ARMATURE_OT_|CONSTRAINT_OT_)|^POSE_OT_(constraint|ik_|armature_apply|bone_layers)")),
    ("pose", re.compile(r"^POSE_OT_")),
    ("scene", re.compile(r"^(OBJECT|MESH|CURVE|SCENE|MATERIAL|CAMERA|LIGHT|COLLECTION|OUTLINER|NODE)_OT_")),
]


def categorize(op_id, area="", mode=""):
    """Map a Blender operator id to a workflow category.

    Transform operators depend on where they ran: moving keys in the dope
    sheet is *timing*, in the graph editor it is *curves*, in the viewport in
    pose mode it is *pose*.
    """
    if op_id.startswith("TRANSFORM_OT_"):
        if area in TIME_EDITORS:
            return "timing"
        if area == "GRAPH_EDITOR":
            return "curves"
        if mode == "POSE":
            return "pose"
        return "scene" if mode in ("OBJECT", "EDIT", "") else "pose"
    for category, pattern in _RULES:
        if pattern.search(op_id):
            return category
    return "other"


def hotkey_label(event):
    mods = "+".join(m for m in event.get("mods", ()) if m)
    return (mods + "+" if mods else "") + event.get("key", "?")


def humanize_op(op_id, name=""):
    if name:
        prefix = op_id.split("_OT_")[0].title() if "_OT_" in op_id else ""
        return (prefix + ": " if prefix else "") + name
    if "_OT_" in op_id:
        prefix, rest = op_id.split("_OT_", 1)
        return prefix.title() + ": " + rest.replace("_", " ")
    return op_id
