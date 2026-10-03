"""Runtime recorder: Blender handlers, timers and the input watcher.

What gets recorded (see README "Event schema"):
  * every operator: built-in ones (``window_manager.operators``) and every
    Python operator of every add-on, with its main settings
  * every hotkey and mouse click, the editor it happened in, and the
    operator / menu the hotkey maps to (add-on hotkeys included)
  * mouse travel per editor, summarised every second
  * scrubbing / playback, mode and active bone changes, undo / redo
  * which datablocks changed (catches add-on panels and sliders too)
  * every keyframe/F-curve change, as the full new state of changed curves
  * rig layouts, enabled add-ons, saves and file loads
  * optional: low-fps screen video while Blender is the active window

Typed text in text editors / the Python console is never recorded.
"""

import os
import sys
import time
import uuid

import bpy

from . import capture, hooks
from .screen import ScreenRecorder
from .storage import SessionLog, Uploader

SCHEMA_VERSION = 1
POLL_INTERVAL = 1.0
KEY_DIFF_INTERVAL = 2.0
HOOK_SCAN_INTERVAL = 10.0
FRAME_THROTTLE = 0.25
IDLE_PAUSE_SECONDS = 120.0
# Blender skips autosave while a modal operator is running, so the input
# watcher steps aside briefly every minute to let autosave through.
WATCHER_LIFETIME = 60.0
WATCHER_GAP = 0.5
TEXT_AREAS = {"TEXT_EDITOR", "CONSOLE", "INFO"}
MOUSE_BUTTONS = {"LEFTMOUSE", "RIGHTMOUSE", "MIDDLEMOUSE", "BUTTON4MOUSE", "BUTTON5MOUSE"}
MODIFIER_KEYS = {"LEFT_CTRL", "RIGHT_CTRL", "LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_ALT", "RIGHT_ALT", "OSKEY"}
IGNORED_EVENTS = {"TIMER", "TIMER_REPORT", "TIMERREGION", "TIMER0", "TIMER1", "TIMER2", "TIMER_JOBS",
                  "TIMER_AUTOSAVE", "NONE", "INBETWEEN_MOUSEMOVE", "TRACKPADPAN", "TRACKPADZOOM",
                  "MOUSEROTATE", "NDOF_MOTION", "XR_ACTION"}


class State:
    def __init__(self):
        self.active = False
        self.log = None
        self.uploader = None
        self.screen = None
        self.session_id = ""
        self.animator = ""
        self.prefs = None
        self.last_op_ptr = 0
        self.last_area = ""
        self.mode = ""
        self.mouse = {}
        self.last_mouse = None
        self.last_frame = None
        self.last_frame_t = 0.0
        self.last_input_t = time.time()
        self.playing = False
        self.context_key = None
        self.digests = {}
        self.dirty_actions = set()
        self.updates = {}
        self.rigs_sent = set()
        self.recent_py_ops = {}
        self.watcher_gen = 0


STATE = State()


def emit(kind, **fields):
    if not STATE.active:
        return
    event = {"v": SCHEMA_VERSION, "t": round(time.time(), 3), "sid": STATE.session_id,
             "a": STATE.animator, "type": kind}
    event.update(fields)
    STATE.log.write(event)


def _file():
    return bpy.data.filepath or "<unsaved>"


def _enabled_addons():
    result = []
    for addon in bpy.context.preferences.addons:
        module = sys.modules.get(addon.module)
        info = getattr(module, "bl_info", {}) if module else {}
        version = info.get("version", "")
        result.append({"module": addon.module, "name": info.get("name", addon.module),
                       "version": ".".join(str(v) for v in version) if isinstance(version, tuple) else str(version)})
    return result


# ---------------------------------------------------------------- animation
def _snapshot_rigs():
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        sig = capture.rig_signature(obj.data)
        key = (obj.name, sig)
        if key in STATE.rigs_sent:
            continue
        STATE.rigs_sent.add(key)
        description = capture.describe_rig(obj)
        for bone in description["bones"]:
            pose_bone = obj.pose.bones.get(bone["name"]) if obj.pose else None
            if pose_bone is not None:
                bone["rotation_mode"] = pose_bone.rotation_mode
        emit("rig", file=_file(), **description)


def _diff_actions(names=None):
    owners = capture.action_owners(bpy.data.objects)
    actions = bpy.data.actions if names is None else [bpy.data.actions.get(n) for n in names]
    for action in actions:
        if action is None:
            continue
        previous = STATE.digests.get(action.name, {})
        changed, removed, current = capture.diff_action(action, previous)
        STATE.digests[action.name] = current
        if changed or removed:
            scene = bpy.context.scene
            emit("keys", file=_file(), action=action.name, owners=owners.get(action.name, []),
                 fps=scene.render.fps / scene.render.fps_base if scene else 24.0,
                 fcurves=changed, removed=removed)
    if names is not None:
        for name in names:
            if name not in bpy.data.actions and STATE.digests.pop(name, None) is not None:
                emit("action_removed", file=_file(), action=name)


def baseline_snapshot():
    """Full state of the file: scene settings, rigs and all animation."""
    scene = bpy.context.scene
    emit("file", file=_file(), scene=scene.name if scene else "",
         fps=(scene.render.fps / scene.render.fps_base) if scene else 24.0,
         frame_start=scene.frame_start if scene else 1, frame_end=scene.frame_end if scene else 250,
         objects=len(bpy.data.objects), actions=len(bpy.data.actions))
    STATE.digests = {}
    STATE.rigs_sent = set()
    _snapshot_rigs()
    _diff_actions()


# ---------------------------------------------------------------- handlers
@bpy.app.handlers.persistent
def on_depsgraph_update(scene, depsgraph):
    if not STATE.active or STATE.playing:
        return
    for update in depsgraph.updates:
        data = update.id.original if hasattr(update.id, "original") else update.id
        if isinstance(data, bpy.types.Action):
            STATE.dirty_actions.add(data.name)
        else:
            anim = getattr(data, "animation_data", None)
            if anim is not None and anim.action is not None:
                STATE.dirty_actions.add(anim.action.name)
        kind = type(data).__name__
        if kind in ("Scene", "ViewLayer", "Depsgraph"):
            continue
        flags = STATE.updates.setdefault(kind + ":" + data.name, set())
        if update.is_updated_transform:
            flags.add("transform")
        if update.is_updated_geometry:
            flags.add("geometry")
        if getattr(update, "is_updated_shading", False):
            flags.add("shading")


@bpy.app.handlers.persistent
def on_frame_change(scene, depsgraph=None):
    if not STATE.active:
        return
    screen = getattr(bpy.context, "screen", None)
    playing = bool(screen and screen.is_animation_playing)
    if playing != STATE.playing:
        STATE.playing = playing
        emit("play", on=playing, f=scene.frame_current)
    if playing:
        return
    now = time.time()
    if scene.frame_current != STATE.last_frame and now - STATE.last_frame_t >= FRAME_THROTTLE:
        STATE.last_frame = scene.frame_current
        STATE.last_frame_t = now
        emit("frame", f=scene.frame_current, area=STATE.last_area)


@bpy.app.handlers.persistent
def on_save_post(*_args):
    if STATE.active:
        _diff_actions()
        emit("save", file=_file())
        STATE.log.flush()


@bpy.app.handlers.persistent
def on_load_post(*_args):
    if STATE.active:
        STATE.last_op_ptr = _latest_op_ptr()
        baseline_snapshot()
        bpy.app.timers.register(start_input_watchers, first_interval=0.5)


@bpy.app.handlers.persistent
def on_undo_post(*_args):
    if STATE.active:
        emit("undo", area=STATE.last_area)
        _diff_actions()


@bpy.app.handlers.persistent
def on_redo_post(*_args):
    if STATE.active:
        emit("redo", area=STATE.last_area)
        _diff_actions()


def on_python_operator(idname, label, module, method, props):
    op_id = capture.op_idname(idname)
    STATE.recent_py_ops[op_id] = time.time()
    emit("op", id=op_id, name=label, area=STATE.last_area, src="py", addon=module, via=method, props=props)


# ---------------------------------------------------------------- polling
def _latest_op_ptr():
    ops = bpy.context.window_manager.operators
    return ops[-1].as_pointer() if len(ops) else 0


def _poll_operators():
    ops = list(bpy.context.window_manager.operators)
    if not ops:
        return
    pointers = [op.as_pointer() for op in ops]
    start = pointers.index(STATE.last_op_ptr) + 1 if STATE.last_op_ptr in pointers else 0
    now = time.time()
    for op in ops[start:]:
        if now - STATE.recent_py_ops.get(op.bl_idname, 0) < 3 * POLL_INTERVAL:
            continue  # already logged by the Python operator hook
        emit("op", id=op.bl_idname, name=op.name, area=STATE.last_area, props=capture.op_props(op))
    STATE.last_op_ptr = pointers[-1]


def _poll_context():
    obj = bpy.context.active_object
    bone = obj.data.bones.active if obj is not None and obj.type == "ARMATURE" and obj.mode == "POSE" else None
    key = (obj.mode if obj else "NONE", obj.name if obj else "", bone.name if bone else "")
    if key != STATE.context_key:
        STATE.context_key = key
        STATE.mode = key[0]
        emit("mode", mode=key[0], active=key[1], bone=key[2])


def _flush_mouse():
    if STATE.mouse:
        emit("mouse", areas={a: {"dist": round(v[0]), "n": v[1]} for a, v in STATE.mouse.items()})
        STATE.mouse = {}


def _flush_updates():
    if STATE.updates:
        items = sorted(STATE.updates.items())[:40]
        emit("upd", ids={name: sorted(flags) for name, flags in items}, more=max(0, len(STATE.updates) - 40))
        STATE.updates = {}


def poll_timer():
    if not STATE.active:
        return None
    try:
        _poll_operators()
        _poll_context()
        _flush_mouse()
        _flush_updates()
        if STATE.screen is not None:
            if time.time() - STATE.last_input_t > IDLE_PAUSE_SECONDS and STATE.screen.recording:
                STATE.screen.pause("idle")
            STATE.screen.tick()
        STATE.log.flush()
    except Exception as exc:  # never break the animator's session
        print("[mouseadon] poll error:", exc)
    return POLL_INTERVAL


def key_diff_timer():
    if not STATE.active:
        return None
    if STATE.dirty_actions:
        names, STATE.dirty_actions = STATE.dirty_actions, set()
        try:
            _diff_actions(names)
            _snapshot_rigs()
        except Exception as exc:
            print("[mouseadon] key diff error:", exc)
    return KEY_DIFF_INTERVAL


def hook_scan_timer():
    if not STATE.active:
        return None
    try:
        hooks.wrap_new_operators()
    except Exception as exc:
        print("[mouseadon] operator hook error:", exc)
    return HOOK_SCAN_INTERVAL


# ---------------------------------------------------------------- input
def _mods(event):
    return [m for m in ("ctrl", "shift", "alt", "oskey") if getattr(event, m, False)]


def on_input(context, event):
    if event.type in IGNORED_EVENTS:
        return
    if event.type == "WINDOW_DEACTIVATE":
        emit("focus", on=False)
        if STATE.screen is not None:
            STATE.screen.pause("unfocused")
        return
    STATE.last_input_t = time.time()
    if STATE.screen is not None and STATE.screen.paused_reason:
        if STATE.screen.paused_reason in ("unfocused", "idle"):
            emit("focus", on=True)
        STATE.screen.resume()
    area, rx, ry = capture.find_area(context.window.screen if context.window else None,
                                     event.mouse_x, event.mouse_y)
    area_type = area.type if area else ""
    if area_type:
        STATE.last_area = area_type
    if event.type == "MOUSEMOVE":
        if STATE.last_mouse is not None:
            dx = event.mouse_x - STATE.last_mouse[0]
            dy = event.mouse_y - STATE.last_mouse[1]
            slot = STATE.mouse.setdefault(area_type or "?", [0.0, 0])
            slot[0] += (dx * dx + dy * dy) ** 0.5
            slot[1] += 1
        STATE.last_mouse = (event.mouse_x, event.mouse_y)
        return
    if event.type in MOUSE_BUTTONS or event.type.startswith("WHEEL"):
        if event.value in {"PRESS", "RELEASE", "CLICK_DRAG", "DOUBLE_CLICK"} or event.type.startswith("WHEEL"):
            region = ""
            if area is not None:
                for reg in area.regions:
                    if reg.x <= event.mouse_x < reg.x + reg.width and reg.y <= event.mouse_y < reg.y + reg.height:
                        region = reg.type
                        break
            emit("click", btn=event.type, value=event.value, area=area_type, region=region,
                 rx=round(rx, 3), ry=round(ry, 3), mods=_mods(event))
        return
    if event.value != "PRESS" or event.type in MODIFIER_KEYS:
        return
    if area_type in TEXT_AREAS and not _mods(event):
        return  # privacy: never log typing
    fields = {"key": event.type, "mods": _mods(event), "area": area_type, "repeat": bool(event.is_repeat)}
    try:
        hit = capture.resolve_hotkey(context.window_manager.keyconfigs.user, area_type, STATE.mode, event.type,
                                     event.value, event.ctrl, event.shift, event.alt, event.oskey)
    except Exception:
        hit = None
    if hit:
        fields.update(hit)
    emit("key", **fields)


class MOUSEADON_OT_input_watch(bpy.types.Operator):
    """Passive watcher: sees every input event and passes it through untouched"""

    bl_idname = "mouseadon.input_watch"
    bl_label = "Mouseadon Input Watcher"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        self._born = time.time()
        self._gen = STATE.watcher_gen
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if not STATE.active or self._gen != STATE.watcher_gen:
            return {"FINISHED", "PASS_THROUGH"}  # replaced by a newer watcher
        if time.time() - self._born > WATCHER_LIFETIME:
            if not bpy.app.timers.is_registered(start_input_watchers):
                bpy.app.timers.register(start_input_watchers, first_interval=WATCHER_GAP)
            return {"FINISHED", "PASS_THROUGH"}
        try:
            on_input(context, event)
        except Exception as exc:
            print("[mouseadon] input error:", exc)
        return {"PASS_THROUGH"}


def start_input_watchers():
    if not STATE.active or not STATE.prefs.get("record_input", True):
        return None
    wm = bpy.context.window_manager
    if wm is None:
        return None
    STATE.watcher_gen += 1
    for window in wm.windows:
        try:
            with bpy.context.temp_override(window=window):
                bpy.ops.mouseadon.input_watch("INVOKE_DEFAULT")
        except Exception as exc:
            print("[mouseadon] could not start input watcher:", exc)
    return None


# ---------------------------------------------------------------- lifecycle
def start(prefs):
    """Start recording. ``prefs`` is a plain dict (see ``__init__.py``)."""
    if STATE.active:
        return
    STATE.__init__()
    STATE.prefs = prefs
    STATE.animator = prefs["animator_id"]
    STATE.session_id = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:8]
    root = os.path.expanduser(prefs["log_dir"])
    STATE.log = SessionLog(root, STATE.animator, STATE.session_id)
    STATE.active = True
    STATE.last_op_ptr = _latest_op_ptr()
    emit("session_start", blender=bpy.app.version_string, addon=prefs.get("addon_version", ""),
         host=os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", ""),
         platform=sys.platform, addons=_enabled_addons())
    baseline_snapshot()
    STATE.log.flush()

    hooks.EMIT = on_python_operator
    hooks.wrap_new_operators()

    if prefs.get("record_screen") and not bpy.app.background:
        STATE.screen = ScreenRecorder(root, STATE.log.animator, STATE.log.session_id,
                                      ffmpeg=prefs.get("ffmpeg_path") or "ffmpeg",
                                      fps=prefs.get("screen_fps", 2), max_width=prefs.get("screen_width", 1600),
                                      on_event=emit)
        STATE.screen.start_segment()

    if prefs.get("server_url") and prefs.get("token"):
        STATE.uploader = Uploader(root, STATE.animator, prefs["server_url"], prefs["token"],
                                  upload_screen=prefs.get("upload_screen", True))
        STATE.uploader.start()

    for handlers, func in _HANDLERS:
        if func not in handlers:
            handlers.append(func)
    bpy.app.timers.register(poll_timer, first_interval=POLL_INTERVAL, persistent=True)
    bpy.app.timers.register(key_diff_timer, first_interval=KEY_DIFF_INTERVAL, persistent=True)
    bpy.app.timers.register(hook_scan_timer, first_interval=HOOK_SCAN_INTERVAL, persistent=True)
    if not bpy.app.background:
        bpy.app.timers.register(start_input_watchers, first_interval=0.1)


def stop():
    if not STATE.active:
        return
    try:
        _poll_operators()
        _diff_actions()
        _flush_mouse()
        _flush_updates()
    except Exception as exc:
        print("[mouseadon] final snapshot error:", exc)
    if STATE.screen is not None:
        STATE.screen.stop_segment("session end")
    emit("session_end")
    STATE.log.flush()
    STATE.active = False
    hooks.unwrap_all()
    hooks.EMIT = None
    for handlers, func in _HANDLERS:
        if func in handlers:
            handlers.remove(func)
    for timer in (poll_timer, key_diff_timer, hook_scan_timer, start_input_watchers):
        if bpy.app.timers.is_registered(timer):
            bpy.app.timers.unregister(timer)
    if STATE.uploader:
        STATE.uploader.stop(final_sync=True)


_HANDLERS = (
    (bpy.app.handlers.depsgraph_update_post, on_depsgraph_update),
    (bpy.app.handlers.frame_change_post, on_frame_change),
    (bpy.app.handlers.save_post, on_save_post),
    (bpy.app.handlers.load_post, on_load_post),
    (bpy.app.handlers.undo_post, on_undo_post),
    (bpy.app.handlers.redo_post, on_redo_post),
)
