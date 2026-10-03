"""Turn Blender animation data into plain JSON-able dicts.

Everything here works on duck-typed objects so it can be exercised with the
``bpy`` module in tests. Supports the legacy ``Action.fcurves`` API (Blender
4.2-4.3) and slotted/layered actions (Blender 4.4+ and 5.x).
"""

import hashlib
import re

BONE_PATH = re.compile(r'pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(\w+)')

INTERPOLATIONS = ("CONSTANT", "LINEAR", "BEZIER")


def iter_action_fcurves(action):
    """Yield ``(slot_identifier, fcurve)`` for every F-curve of an action."""
    layers = getattr(action, "layers", None)
    if layers:
        for layer in layers:
            for strip in layer.strips:
                for bag in getattr(strip, "channelbags", ()):
                    slot = getattr(bag, "slot", None)
                    slot_id = getattr(slot, "identifier", "") if slot else ""
                    for fcurve in bag.fcurves:
                        yield slot_id, fcurve
        return
    for fcurve in getattr(action, "fcurves", ()):
        yield "", fcurve


def fcurve_key(slot_id, fcurve):
    return "%s|%s|%d" % (slot_id, fcurve.data_path, fcurve.array_index)


def serialize_fcurve(slot_id, fcurve):
    """Compact representation: one list per keyframe.

    ``[frame, value, interpolation, key_type, hl_x, hl_y, hr_x, hr_y,
    handle_left_type, handle_right_type, easing]``
    """
    keys = []
    for kp in fcurve.keyframe_points:
        keys.append([
            round(kp.co[0], 4), round(kp.co[1], 6), kp.interpolation, kp.type,
            round(kp.handle_left[0], 4), round(kp.handle_left[1], 6),
            round(kp.handle_right[0], 4), round(kp.handle_right[1], 6),
            kp.handle_left_type, kp.handle_right_type, kp.easing,
        ])
    match = BONE_PATH.match(fcurve.data_path)
    group = fcurve.group.name if getattr(fcurve, "group", None) else ""
    return {
        "slot": slot_id,
        "path": fcurve.data_path,
        "idx": fcurve.array_index,
        "bone": match.group(1) if match else "",
        "prop": match.group(2) if match else fcurve.data_path.rsplit(".", 1)[-1],
        "group": group,
        "keys": keys,
    }


def fcurve_digest(fcurve):
    """Cheap change-detection hash of an F-curve's keys and handles."""
    count = len(fcurve.keyframe_points)
    digest = hashlib.blake2b(digest_size=12)
    digest.update(str(count).encode())
    if count:
        for attr in ("co", "handle_left", "handle_right"):
            buffer = [0.0] * (count * 2)
            fcurve.keyframe_points.foreach_get(attr, buffer)
            digest.update(repr([round(x, 5) for x in buffer]).encode())
        digest.update("".join(kp.interpolation[0] + kp.type[0] for kp in fcurve.keyframe_points).encode())
    return digest.hexdigest()


def action_digests(action):
    return {fcurve_key(slot, fc): fcurve_digest(fc) for slot, fc in iter_action_fcurves(action)}


def diff_action(action, previous):
    """Compare an action against previous digests.

    Returns ``(changed_fcurves, removed_keys, new_digests)``.
    """
    current = {}
    changed = []
    for slot, fc in iter_action_fcurves(action):
        key = fcurve_key(slot, fc)
        digest = fcurve_digest(fc)
        current[key] = digest
        if previous.get(key) != digest:
            changed.append(serialize_fcurve(slot, fc))
    removed = sorted(set(previous) - set(current))
    return changed, removed, current


def rig_signature(armature):
    """Stable id for a rig: hash of its sorted bone names.

    The same character rig used across shots and animators gets the same
    signature, so motion learned on it can be pooled.
    """
    names = sorted(bone.name for bone in armature.bones)
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


def describe_rig(obj):
    """Rest-pose bone layout, enough to rebuild a proxy rig later."""
    armature = obj.data
    bones = []
    for bone in armature.bones:
        bones.append({
            "name": bone.name,
            "parent": bone.parent.name if bone.parent else "",
            "head": [round(v, 5) for v in bone.head_local],
            "tail": [round(v, 5) for v in bone.tail_local],
            "roll_z": [round(v, 5) for v in bone.matrix_local.col[2][:3]],
            "deform": bool(bone.use_deform),
            "connect": bool(bone.use_connect),
        })
    return {"object": obj.name, "armature": armature.name, "sig": rig_signature(armature), "bones": bones}


def action_owners(objects):
    """Map action name -> list of {object, rig} that currently use it."""
    owners = {}
    for obj in objects:
        anim = getattr(obj, "animation_data", None)
        action = getattr(anim, "action", None) if anim else None
        if action is None:
            continue
        rig = rig_signature(obj.data) if obj.type == "ARMATURE" else ""
        owners.setdefault(action.name, []).append({"object": obj.name, "rig": rig, "type": obj.type})
    return owners


def find_area(screen, x, y):
    """Return ``(area, rx, ry)`` for window coordinates or ``(None, 0, 0)``."""
    if screen is None:
        return None, 0.0, 0.0
    for area in screen.areas:
        if area.x <= x < area.x + area.width and area.y <= y < area.y + area.height:
            return area, (x - area.x) / max(area.width, 1), (y - area.y) / max(area.height, 1)
    return None, 0.0, 0.0


def op_idname(bl_idname):
    """'mesh.primitive_cube_add' -> 'MESH_OT_primitive_cube_add' (already-C ids pass through)."""
    if "_OT_" in bl_idname or "." not in bl_idname:
        return bl_idname
    prefix, name = bl_idname.split(".", 1)
    return prefix.upper() + "_OT_" + name


def op_props(op, limit=8):
    """A few simple operator settings (mode, value, type...) for context."""
    props = {}
    try:
        rna_props = op.properties.bl_rna.properties
    except (AttributeError, ReferenceError):
        return props
    for prop in rna_props:
        if prop.identifier == "rna_type" or len(props) >= limit:
            continue
        if prop.type not in {"BOOLEAN", "INT", "FLOAT", "ENUM", "STRING"} or prop.is_hidden:
            continue
        if prop.type == "STRING" and prop.subtype in {"PASSWORD", "FILE_PATH", "DIR_PATH", "FILE_NAME"}:
            continue
        try:
            value = getattr(op.properties, prop.identifier)
        except Exception:
            continue
        if prop.type == "ENUM" and prop.is_enum_flag:
            value = sorted(value)
        elif prop.type in {"INT", "FLOAT", "BOOLEAN"} and getattr(prop, "array_length", 0):
            value = [round(v, 4) if isinstance(v, float) else v for v in value][:4]
        elif isinstance(value, float):
            value = round(value, 4)
        elif isinstance(value, str):
            value = value[:60]
        props[prop.identifier] = value
    return props


# Keymaps that apply in every editor, checked after the editor's own keymaps.
GENERIC_KEYMAPS = ("Window", "Screen", "Frames", "Screen Editing", "Animation", "User Interface")
MODE_KEYMAPS = {"POSE": "Pose", "OBJECT": "Object Mode", "EDIT_ARMATURE": "Armature", "EDIT_MESH": "Mesh",
                "EDIT": "Mesh", "SCULPT": "Sculpt", "PAINT_WEIGHT": "Weight Paint"}
SPACE_KEYMAPS = {"VIEW_3D": ("3D View", "3D View Generic"), "GRAPH_EDITOR": ("Graph Editor", "Graph Editor Generic"),
                 "DOPESHEET_EDITOR": ("Dopesheet", "Dopesheet Generic"), "NLA_EDITOR": ("NLA Editor", "NLA Generic"),
                 "TIMELINE": ("Dopesheet", "Dopesheet Generic"), "OUTLINER": ("Outliner",),
                 "PROPERTIES": ("Property Editor",), "SEQUENCE_EDITOR": ("Sequencer",),
                 "IMAGE_EDITOR": ("Image",), "NODE_EDITOR": ("Node Editor",), "TEXT_EDITOR": ("Text",)}


def _mod_ok(kmi_value, pressed):
    # Blender 4.x: -1 = any, 0 = off, 1 = on (older: bool)
    if kmi_value is True or kmi_value is False:
        return bool(kmi_value) == pressed
    return kmi_value == -1 or bool(kmi_value) == pressed


def resolve_hotkey(keyconfig, area_type, mode, event_type, value, ctrl, shift, alt, oskey):
    """Best guess of the operator a key press triggered, including add-on hotkeys.

    Checks the editor's keymaps, then the mode keymap, then generic ones; any
    add-on keymap with the same name is merged into those by Blender.
    """
    if keyconfig is None:
        return None
    names = list(SPACE_KEYMAPS.get(area_type, ()))
    if area_type == "VIEW_3D" and mode in MODE_KEYMAPS:
        names.insert(0, MODE_KEYMAPS[mode])
    names += GENERIC_KEYMAPS
    keymaps = keyconfig.keymaps
    for name in names:
        keymap = keymaps.get(name)
        if keymap is None:
            continue
        for kmi in keymap.keymap_items:
            if not kmi.active or kmi.type != event_type or kmi.value not in (value, "ANY"):
                continue
            if getattr(kmi, "any", False) or (_mod_ok(kmi.ctrl, ctrl) and _mod_ok(kmi.shift, shift)
                                              and _mod_ok(kmi.alt, alt) and _mod_ok(kmi.oskey, oskey)):
                result = {"op": op_idname(kmi.idname), "keymap": name}
                menu = getattr(kmi.properties, "name", None) if kmi.properties else None
                if isinstance(menu, str) and menu:
                    result["menu"] = menu[:60]
                return result
    return None
