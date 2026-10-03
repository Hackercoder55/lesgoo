"""Mouseadon Recorder - learns how your animators animate.

Install on every animator's Blender (Edit > Preferences > Add-ons > Install),
set the animator ID / server, and have the animator tick the consent box once.
From then on every Blender session is recorded automatically and synced to the
Mouseadon collector, where the daily pipeline learns from it.
"""

bl_info = {
    "name": "Mouseadon Recorder",
    "author": "Mouseadon",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Mouseadon",
    "description": "Records animation workflow (operators, input, keyframes) for training",
    "category": "Animation",
}

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty

from . import recorder

ADDON_VERSION = ".".join(str(v) for v in bl_info["version"])


def get_prefs(context=None):
    context = context or bpy.context
    addon = context.preferences.addons.get(__package__)
    return addon.preferences if addon else None


def prefs_dict(prefs):
    return {
        "animator_id": prefs.animator_id.strip(),
        "log_dir": prefs.log_dir,
        "server_url": prefs.server_url.strip(),
        "token": prefs.token.strip(),
        "record_input": prefs.record_input,
        "record_screen": prefs.record_screen,
        "upload_screen": prefs.upload_screen,
        "ffmpeg_path": prefs.ffmpeg_path.strip(),
        "screen_fps": prefs.screen_fps,
        "screen_width": prefs.screen_width,
        "addon_version": ADDON_VERSION,
    }


def can_record(prefs):
    return bool(prefs and prefs.consent and prefs.enabled and prefs.animator_id.strip())


class MouseadonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    animator_id: StringProperty(name="Animator ID", description="Unique ID, e.g. anim_042")
    log_dir: StringProperty(name="Local log folder", default="~/mouseadon_logs", subtype="DIR_PATH")
    server_url: StringProperty(name="Collector URL", description="e.g. http://10.0.0.5:8765 (empty = local only)")
    token: StringProperty(name="Token", subtype="PASSWORD")
    record_input: BoolProperty(name="Record hotkeys, clicks and mouse travel", default=True)
    record_screen: BoolProperty(
        name="Record screen video",
        description="Low-fps screen video via ffmpeg, only while Blender is the active window",
        default=True,
    )
    upload_screen: BoolProperty(name="Upload screen video to the collector", default=True)
    ffmpeg_path: StringProperty(name="ffmpeg", default="ffmpeg", subtype="FILE_PATH",
                                description="ffmpeg executable (name on PATH or full path)")
    screen_fps: IntProperty(name="Screen fps", default=2, min=1, max=15)
    screen_width: IntProperty(name="Max video width", default=1600, min=640, max=3840)
    enabled: BoolProperty(name="Record automatically", default=True)
    consent: BoolProperty(
        name="I agree that my Blender activity is recorded for training",
        description="Recording never starts without this",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "animator_id")
        col.prop(self, "log_dir")
        col.prop(self, "server_url")
        col.prop(self, "token")
        col.prop(self, "record_input")
        col.prop(self, "enabled")
        screen = layout.box()
        screen.prop(self, "record_screen")
        sub = screen.column()
        sub.active = self.record_screen
        sub.prop(self, "ffmpeg_path")
        row = sub.row()
        row.prop(self, "screen_fps")
        row.prop(self, "screen_width")
        sub.prop(self, "upload_screen")
        box = layout.box()
        box.label(text="Records every operator (all add-ons), hotkeys, clicks, mouse travel, keyframe edits")
        box.label(text="and, if enabled, screen video while Blender is the active window.")
        box.label(text="Never records typed text; video pauses when you switch away from Blender or go idle.")
        box.prop(self, "consent")


class MOUSEADON_OT_start(bpy.types.Operator):
    """Start recording this session"""

    bl_idname = "mouseadon.start"
    bl_label = "Start Recording"

    def execute(self, context):
        prefs = get_prefs(context)
        if not prefs or not prefs.consent or not prefs.animator_id.strip():
            self.report({"ERROR"}, "Set Animator ID and give consent in the add-on preferences")
            return {"CANCELLED"}
        recorder.start(prefs_dict(prefs))
        return {"FINISHED"}


class MOUSEADON_OT_stop(bpy.types.Operator):
    """Stop recording this session"""

    bl_idname = "mouseadon.stop"
    bl_label = "Stop Recording"

    def execute(self, context):
        recorder.stop()
        return {"FINISHED"}


class MOUSEADON_PT_panel(bpy.types.Panel):
    bl_label = "Mouseadon"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mouseadon"

    def draw(self, context):
        layout = self.layout
        state = recorder.STATE
        if state.active:
            layout.label(text="Recording: " + state.animator, icon="REC")
            layout.label(text="Events saved: %d" % state.log.events_written)
            if state.screen is not None:
                scr = state.screen
                if scr.recording:
                    layout.label(text="Screen: recording", icon="RENDER_ANIMATION")
                elif scr.last_error:
                    layout.label(text="Screen: " + scr.last_error[:40], icon="ERROR")
                else:
                    layout.label(text="Screen: paused (%s)" % (scr.paused_reason or "-"), icon="PAUSE")
            if state.uploader:
                err = state.uploader.last_error
                layout.label(text=("Sync error: " + err[:40]) if err else "Sync: OK",
                             icon="ERROR" if err else "CHECKMARK")
            layout.operator("mouseadon.stop", icon="PAUSE")
        else:
            layout.label(text="Not recording", icon="RADIOBUT_OFF")
            layout.operator("mouseadon.start", icon="REC")


def draw_status(self, context):
    if recorder.STATE.active:
        self.layout.label(text="Mouseadon REC", icon="REC")


def _autostart():
    prefs = get_prefs()
    if can_record(prefs) and not recorder.STATE.active:
        recorder.start(prefs_dict(prefs))
    return None


CLASSES = (MouseadonPreferences, recorder.MOUSEADON_OT_input_watch, MOUSEADON_OT_start,
           MOUSEADON_OT_stop, MOUSEADON_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.STATUSBAR_HT_header.append(draw_status)
    bpy.app.timers.register(_autostart, first_interval=1.0)


def unregister():
    recorder.stop()
    if bpy.app.timers.is_registered(_autostart):
        bpy.app.timers.unregister(_autostart)
    bpy.types.STATUSBAR_HT_header.remove(draw_status)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
