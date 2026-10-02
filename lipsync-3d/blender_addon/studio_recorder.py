# Studio Recorder - records how your animators animate, as training data.
#
# Install on each animator's Blender: Preferences > Add-ons > Install from
# Disk > this file, tick "Studio Recorder", set the data folder in the add-on
# preferences (a shared/network folder is best). Panel: 3D View > N > Recorder.
#
# What is recorded (Blender data only - never the screen, camera or mic):
#   - every save: the full animation of the scene (each animated bone /
#     shape key / property, every keyframe with interpolation and handles),
#     only when it changed since the last snapshot
#   - "Mark shot final": the same, labelled as the finished take
#   - shot id, script line, dialogue sound strips, markers, cameras, linked
#     character assets, fps and frame range
#   - every minute: whether the file is being worked on (time per shot)
#   - the Blender operators used (keyframe insert, rotate, graph editor
#     tools...), i.e. the order in which the animator works
# Each animator can pause recording at any time from the panel.
#
# Old finished projects can be harvested without the add-on installed:
#   blender -b shot.blend --python studio_recorder.py -- --harvest OUT_DIR
# (harvest_all.py runs that over a whole folder of .blend files).

bl_info = {
    "name": "Studio Recorder",
    "author": "Lip-Sync Studio",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar > Recorder",
    "description": "Records animation work (keyframes, edit history, shot info) as "
                   "training data for the studio's animation AI",
    "category": "Animation",
}

import getpass
import gzip
import hashlib
import json
import os
import socket
import sys
import time
import uuid

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty

FORMAT = 1
_state = {"last_hash": {}, "seen_ops": set(), "last_dirty": None, "session": uuid.uuid4().hex[:12]}


# -------------------------------------------------------------- where to

def _prefs():
    a = bpy.context.preferences.addons.get(__name__)
    return a.preferences if a else None


def data_dir():
    p = _prefs()
    d = bpy.path.abspath(p.data_dir) if p and p.data_dir else \
        os.path.join(os.path.expanduser("~"), "StudioRecorder")
    os.makedirs(d, exist_ok=True)
    return d


def who():
    p = _prefs()
    return (p.animator if p and p.animator else getpass.getuser()) + "@" + socket.gethostname()


def enabled():
    p = _prefs()
    return bool(p) and p.enabled and not bpy.app.background


def _write(rec, out_dir=None):
    """One JSON object per line, one gzip file per animator per day."""
    rec.setdefault("format", FORMAT)
    rec.setdefault("time", time.time())
    rec.setdefault("user", who() if not out_dir else "harvest")
    rec.setdefault("session", _state["session"])
    d = out_dir or data_dir()
    os.makedirs(d, exist_ok=True)
    name = f"{time.strftime('%Y-%m-%d')}_{rec['user'].replace('@', '_')}.jsonl.gz"
    with gzip.open(os.path.join(d, name), "at", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


# ------------------------------------------------------------- snapshot

def _fcurves(action):
    fcs = getattr(action, "fcurves", None)
    if fcs is not None:
        try:
            return list(fcs)
        except Exception:
            pass
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                out += list(bag.fcurves)
    return out


def _curves(action):
    out = []
    for fc in _fcurves(action):
        keys = [[round(k.co[0], 3), round(k.co[1], 5), k.interpolation[0],
                 round(k.handle_left[0], 3), round(k.handle_left[1], 5),
                 round(k.handle_right[0], 3), round(k.handle_right[1], 5)]
                for k in fc.keyframe_points]
        if keys:
            out.append({"path": fc.data_path, "index": fc.array_index,
                        "group": fc.group.name if fc.group else "",
                        "mods": [m.type for m in fc.modifiers], "keys": keys})
    return out


def _anim_of(idblock):
    ad = getattr(idblock, "animation_data", None)
    if not ad:
        return None
    res = {}
    if ad.action:
        res["action"] = ad.action.name
        res["curves"] = _curves(ad.action)
    nla = []
    for tr in ad.nla_tracks:
        for st in tr.strips:
            if st.action:
                nla.append({"track": tr.name, "strip": st.name, "action": st.action.name,
                            "start": st.frame_start, "end": st.frame_end,
                            "curves": _curves(st.action)})
    if nla:
        res["nla"] = nla
    return res or None


def _sounds(scene):
    se = scene.sequence_editor
    if not se:
        return []
    allx = getattr(se, "strips_all", None) or getattr(se, "sequences_all", [])
    out = []
    for s in allx:
        if s.type == "SOUND" and s.sound:
            out.append({"name": s.name, "path": bpy.path.abspath(s.sound.filepath),
                        "frame_start": s.frame_start,
                        "frame_final": [s.frame_final_start, s.frame_final_end],
                        "channel": s.channel, "volume": s.volume})
        elif s.type == "TEXT":
            out.append({"name": s.name, "text": getattr(s, "text", ""),
                        "frame_final": [s.frame_final_start, s.frame_final_end]})
    return out


def snapshot(reason, scene=None):
    """Everything an animation model needs to learn from this shot."""
    sc = scene or bpy.context.scene
    objs = []
    for ob in sc.objects:
        rec = {"name": ob.name, "type": ob.type,
               "library": bpy.path.abspath(ob.library.filepath) if ob.library else None,
               "data": ob.data.name if ob.data else None}
        if ob.override_library and ob.override_library.reference:
            ref = ob.override_library.reference
            rec["asset"] = {"name": ref.name,
                            "library": bpy.path.abspath(ref.library.filepath) if ref.library else None}
        a = _anim_of(ob)
        if a:
            rec["anim"] = a
        if ob.type == "ARMATURE":
            rec["bones"] = [b.name for b in ob.pose.bones]
        sk = getattr(getattr(ob, "data", None), "shape_keys", None)
        if sk:
            rec["shape_keys"] = [k.name for k in sk.key_blocks]
            a = _anim_of(sk)
            if a:
                rec["shape_anim"] = a
        if ob.type == "CAMERA":
            rec["lens"] = ob.data.lens
        if a or ob.type in ("ARMATURE", "CAMERA") or sk:
            objs.append(rec)
    props = getattr(sc, "studio_rec", None)
    return {"type": "snapshot", "reason": reason, "file": bpy.data.filepath,
            "scene": sc.name, "fps": sc.render.fps / sc.render.fps_base,
            "frame_range": [sc.frame_start, sc.frame_end],
            "shot": props.shot if props else "", "script": props.script if props else "",
            "camera": sc.camera.name if sc.camera else None,
            "markers": [{"name": m.name, "frame": m.frame,
                         "camera": m.camera.name if m.camera else None}
                        for m in sc.timeline_markers],
            "sounds": _sounds(sc), "objects": objs}


def record_snapshot(reason, force=False, out_dir=None):
    snap = snapshot(reason)
    body = json.dumps(snap["objects"], sort_keys=True).encode()
    h = hashlib.sha1(body).hexdigest()
    key = (bpy.data.filepath, snap["scene"])
    if not force and _state["last_hash"].get(key) == h:
        return False
    _state["last_hash"][key] = h
    snap["hash"] = h
    _write(snap, out_dir)
    return True


# ------------------------------------------------------------- handlers

@persistent
def on_save(_):
    if enabled():
        try:
            record_snapshot("save")
        except Exception as e:                      # never get in the animator's way
            print("Studio Recorder:", e)


@persistent
def on_load(_):
    if enabled():
        _write({"type": "open", "file": bpy.data.filepath})


def _tick():
    """Every 5 s: new operators. Every 60 s: a work heartbeat."""
    if not enabled():
        return 5.0
    try:
        wm = bpy.context.window_manager
        for op in list(wm.operators)[-20:]:
            k = op.as_pointer()
            if k in _state["seen_ops"]:
                continue
            _state["seen_ops"].add(k)
            _write({"type": "op", "op": op.bl_idname, "file": bpy.data.filepath,
                    "frame": bpy.context.scene.frame_current})
        if len(_state["seen_ops"]) > 5000:
            _state["seen_ops"] = set(list(_state["seen_ops"])[-500:])
        now = time.time()
        if now - _state.get("beat", 0) >= 60:
            _state["beat"] = now
            _write({"type": "beat", "file": bpy.data.filepath, "dirty": bpy.data.is_dirty,
                    "frame": bpy.context.scene.frame_current})
    except Exception as e:
        print("Studio Recorder:", e)
    return 5.0


# ------------------------------------------------------------------- ui

class SREC_Props(bpy.types.PropertyGroup):
    shot: StringProperty(name="Shot", description="Shot / scene id from the script, e.g. EP03_SC12_SH04")
    script: StringProperty(name="Script line", description="What happens / what is said in this shot")


class SREC_OT_final(bpy.types.Operator):
    bl_idname = "studio_rec.mark_final"
    bl_label = "Mark shot final"
    bl_description = "Record this shot's animation as the finished, approved version"

    def execute(self, context):
        if not bpy.data.filepath:
            self.report({"ERROR"}, "Save the file first")
            return {"CANCELLED"}
        record_snapshot("final", force=True)
        self.report({"INFO"}, "Final take recorded")
        return {"FINISHED"}


class SREC_OT_toggle(bpy.types.Operator):
    bl_idname = "studio_rec.toggle"
    bl_label = "Pause / resume recording"

    def execute(self, context):
        p = _prefs()
        if p.enabled:
            _write({"type": "paused", "file": bpy.data.filepath})
            p.enabled = False
        else:
            p.enabled = True
            _write({"type": "resumed", "file": bpy.data.filepath})
        return {"FINISHED"}


class SREC_PT_panel(bpy.types.Panel):
    bl_label = "Studio Recorder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Recorder"

    def draw(self, context):
        p = _prefs()
        lay = self.layout
        on = bool(p) and p.enabled
        row = lay.row()
        row.label(text="Recording" if on else "Paused", icon="REC" if on else "PAUSE")
        row.operator("studio_rec.toggle", text="Pause" if on else "Resume")
        props = context.scene.studio_rec
        lay.prop(props, "shot")
        lay.prop(props, "script")
        lay.operator("studio_rec.mark_final", icon="CHECKMARK")
        lay.label(text=f"Data: {data_dir()}", icon="FILE_FOLDER")


class SREC_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    enabled: BoolProperty(name="Record", default=True)
    data_dir: StringProperty(name="Data folder", subtype="DIR_PATH",
                             description="Where recordings go - a shared network folder "
                                         "collects every animator's data in one place")
    animator: StringProperty(name="Animator name", description="Defaults to the login name")

    def draw(self, context):
        self.layout.prop(self, "enabled")
        self.layout.prop(self, "data_dir")
        self.layout.prop(self, "animator")
        self.layout.label(text="Records Blender animation data only - no screen, camera or "
                               "microphone.", icon="INFO")


CLASSES = (SREC_Props, SREC_OT_final, SREC_OT_toggle, SREC_PT_panel, SREC_Prefs)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.studio_rec = bpy.props.PointerProperty(type=SREC_Props)
    bpy.app.handlers.save_post.append(on_save)
    bpy.app.handlers.load_post.append(on_load)
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=5.0, persistent=True)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    for h, f in ((bpy.app.handlers.save_post, on_save), (bpy.app.handlers.load_post, on_load)):
        if f in h:
            h.remove(f)
    del bpy.types.Scene.studio_rec
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


def _harvest_cli():
    """blender -b shot.blend --python studio_recorder.py -- --harvest OUT_DIR"""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--harvest" not in argv:
        return False
    out = argv[argv.index("--harvest") + 1]
    n = 0
    for sc in bpy.data.scenes:
        snap = snapshot("harvest", sc)
        if any("anim" in o or "shape_anim" in o for o in snap["objects"]):
            _write(snap, out)
            n += 1
    print(f"Studio Recorder: harvested {n} animated scene(s) from {bpy.data.filepath}")
    return True


if __name__ == "__main__":
    if not _harvest_cli():
        register()
