# Lip-Sync Rig - audio to mouth-shape keyframes on your own character rig.
#
# Install: Blender > Edit > Preferences > Add-ons > Install... > pick this .py
# Use:     3D View > sidebar (N) > "Lip Sync" tab
#
# The mouth shapes come from Rhubarb Lip Sync (MIT, github.com/DanielSWolf/
# rhubarb-lip-sync), which the add-on can download for you. Nothing is
# generated as pixels: the character's own shape keys are animated, so the
# face never warps, whatever the camera or the head does.

bl_info = {
    "name": "Lip-Sync Rig (1.3.1)",
    "author": "Lip-Sync Studio",
    "version": (1, 3, 1),
    "blender": (3, 6, 0),
    "location": "3D View > Sidebar > Lip Sync",
    "description": "Keyframe mouth shape keys from dialogue audio (Rhubarb Lip Sync), "
                   "for several characters at once",
    "category": "Animation",
}

import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import urllib.request
import zipfile

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)

RHUBARB_VERSION = "1.14.0"
RHUBARB_URL = ("https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/"
               "v{v}/Rhubarb-Lip-Sync-{v}-{os}.zip")

# Rhubarb's mouth shapes (its README):
#   A closed (P B M)          B slightly open, teeth together (K S T, EE)
#   C open (EH, AE)           D wide open (AA)
#   E slightly rounded (AO, ER)   F puckered (UW, OW, W)
#   G teeth on lower lip (F V)    H tongue up (long L)     X rest / pause
SHAPES = "ABCDEFGHX"
SHAPE_HELP = {
    "A": "Closed - P, B, M", "B": "Slightly open, teeth together - K, S, T, EE",
    "C": "Open - EH, AE", "D": "Wide open - AA", "E": "Slightly rounded - AO, ER",
    "F": "Puckered - OO, OW, W", "G": "Teeth on lower lip - F, V",
    "H": "Tongue up - long L", "X": "Rest / pause",
}

# Recipes for common shape-key sets, tried in order. Each maps a Rhubarb
# shape to {shape key name: value}; names are matched case-insensitively and
# ignoring "_", "." and spaces, so "Jaw_Open" finds "jawOpen".
PRESETS = [
    ("Character Creator 4 (V_Open, V_Explosive...)", {
        "A": {"V_Explosive": 1}, "B": {"V_Wide": .45, "V_Lip_Open": .35},
        "C": {"V_Open": .45, "V_Wide": .35}, "D": {"V_Open": 1},
        "E": {"V_Tight_O": .6, "V_Open": .3}, "F": {"V_Tight_O": 1},
        "G": {"V_Dental_Lip": 1}, "H": {"V_Open": .5, "V_Lip_Open": .3}, "X": {},
    }),
    ("Character Creator 4 direct (AE, AH, B_M_P...)", {
        "A": {"B_M_P": 1}, "B": {"S_Z": .7, "EE": .3}, "C": {"AE": 1}, "D": {"AH": 1},
        "E": {"Oh": .7, "Er": .3}, "F": {"W_OO": 1}, "G": {"F_V": 1}, "H": {"T_L_D_N": 1},
        "X": {},
    }),
    ("Character Creator 3 / iClone (Open, Explosive, Tight-O...)", {
        "A": {"Explosive": 1}, "B": {"Wide": .45, "Lip_Open": .35},
        "C": {"Open": .45, "Wide": .35}, "D": {"Open": 1},
        "E": {"Tight-O": .6, "Open": .3}, "F": {"Tight-O": 1},
        "G": {"Dental_Lip": 1}, "H": {"Open": .5, "Lip_Open": .3}, "X": {},
    }),
    ("ARKit / Apple (jawOpen, mouthFunnel...)", {
        "A": {"mouthClose": .6, "mouthPressLeft": .5, "mouthPressRight": .5},
        "B": {"jawOpen": .12, "mouthStretchLeft": .25, "mouthStretchRight": .25},
        "C": {"jawOpen": .32, "mouthStretchLeft": .2, "mouthStretchRight": .2},
        "D": {"jawOpen": .6},
        "E": {"jawOpen": .3, "mouthFunnel": .45},
        "F": {"jawOpen": .12, "mouthPucker": .85, "mouthFunnel": .25},
        "G": {"jawOpen": .08, "mouthRollLower": .6, "mouthUpperUpLeft": .3, "mouthUpperUpRight": .3},
        "H": {"jawOpen": .38, "mouthStretchLeft": .1, "mouthStretchRight": .1},
        "X": {},
    }),
    ("Oculus / ReadyPlayerMe / CC (viseme_aa...)", {
        "A": {"viseme_PP": 1}, "B": {"viseme_SS": .8, "viseme_kk": .3},
        "C": {"viseme_E": 1}, "D": {"viseme_aa": 1}, "E": {"viseme_O": .8},
        "F": {"viseme_U": 1}, "G": {"viseme_FF": 1}, "H": {"viseme_nn": .6, "viseme_aa": .3},
        "X": {"viseme_sil": 1},
    }),
    ("VRChat (vrc.v_aa...)", {
        "A": {"vrc.v_pp": 1}, "B": {"vrc.v_ss": .8, "vrc.v_kk": .3},
        "C": {"vrc.v_e": 1}, "D": {"vrc.v_aa": 1}, "E": {"vrc.v_oh": .8},
        "F": {"vrc.v_ou": 1}, "G": {"vrc.v_ff": 1}, "H": {"vrc.v_nn": .6, "vrc.v_aa": .3},
        "X": {"vrc.v_sil": 1},
    }),
    ("Preston Blair / Papagayo (AI, E, O, MBP...)", {
        "A": {"MBP": 1}, "B": {"etc": 1}, "C": {"E": 1}, "D": {"AI": 1}, "E": {"O": 1},
        "F": {"U": .6, "WQ": .6}, "G": {"FV": 1}, "H": {"L": 1}, "X": {"rest": 1},
    }),
    ("Rhubarb letters (A, B, C ... X)", {s: {s: 1} for s in SHAPES}),
    ("Single 'mouth open' key", {
        "A": {}, "B": {"mouthOpen": .15}, "C": {"mouthOpen": .45}, "D": {"mouthOpen": .85},
        "E": {"mouthOpen": .4}, "F": {"mouthOpen": .2}, "G": {"mouthOpen": .1},
        "H": {"mouthOpen": .5}, "X": {},
    }),
]


def _norm(name):
    return "".join(c for c in name.lower() if c.isalnum())


def _keys_of(obj):
    sk = getattr(getattr(obj, "data", None), "shape_keys", None)
    return {kb.name: kb for kb in sk.key_blocks} if sk else {}


# Expression keys by meaning, for rigs whose names match no preset exactly
# (Character Creator "Extended" Mouth_Funnel_Up_L, Daz, Mixamo, custom...).
# A key belongs to a concept when its name contains all of the words.
CONCEPTS = {
    "jaw":     [["jaw", "open"], ["mouth", "open"]],
    "close":   [["mouth", "close"]],
    "funnel":  [["funnel"]],
    "pucker":  [["pucker"], ["kiss"]],
    "press":   [["mouth", "press"], ["lips", "press"]],
    "stretch": [["mouth", "stretch"], ["mouth", "wide"]],
    "rolllow": [["roll", "lower"], ["roll", "in", "lower"], ["lower", "lip", "in"]],
    "upperup": [["upper", "up"], ["up", "upper"], ["upper", "lip", "raise"]],
}
CONCEPT_RECIPE = {
    "A": {"close": .7, "press": .5}, "B": {"jaw": .12, "stretch": .3},
    "C": {"jaw": .35, "stretch": .25}, "D": {"jaw": .65},
    "E": {"jaw": .3, "funnel": .5}, "F": {"jaw": .1, "pucker": .8, "funnel": .3},
    "G": {"jaw": .08, "rolllow": .6, "upperup": .3}, "H": {"jaw": .4, "stretch": .1},
    "X": {},
}


def concept_mapping(obj):
    """(concepts found, mapping) built from what the shape keys are called."""
    keys = list(_keys_of(obj))
    found = {}
    for c, alts in CONCEPTS.items():
        hits = []
        for k in keys:
            n = _norm(k)
            if any(all(w in n for w in alt) for alt in alts):
                hits.append(k)
        if c == "jaw":                  # one jaw key is enough; prefer jaw_open
            hits = sorted(hits, key=lambda k: "jaw" not in _norm(k))[:1]
        if hits:
            found[c] = hits
    mapping = {s: {k: v for c, v in d.items() for k in found.get(c, [])}
               for s, d in CONCEPT_RECIPE.items()}
    return len(found), mapping


def auto_mapping(obj, with_score=False):
    """(preset name, {shape: {real key: value}}) for the preset that finds
    the most of its keys on this object, or (None, {})."""
    keys = _keys_of(obj)
    by_norm = {_norm(k): k for k in keys}
    alias = {"mouthopen": ["mouthopen", "openmouth", "mouth", "open", "jawopen"]}
    best = (0, None, {})
    for name, recipe in PRESETS:
        wanted = {k for d in recipe.values() for k in d}
        if not wanted:
            continue
        found = {}
        for w in wanted:
            for cand in alias.get(_norm(w), [_norm(w)]):
                if cand in by_norm:
                    found[w] = by_norm[cand]
                    break
        score = len(found) / len(wanted)
        if score > best[0] and found:
            mapping = {s: {found[k]: v for k, v in d.items() if k in found}
                       for s, d in recipe.items()}
            best = (score, name, mapping)
    # a full viseme set wins; otherwise, when the rig has real lip shapes
    # (funnel, pucker, press...) use them rather than only opening the jaw
    if best[0] < 0.75 or (best[1] or "").startswith("Single"):
        nc, cm = concept_mapping(obj)
        if nc >= 3:
            best = (0.74 + nc / 100, f"Expression keys ({nc} lip shapes found)", cm)
    return best if with_score else (best[1], best[2])


def _rig_of(obj):
    for m in getattr(obj, "modifiers", []):
        if m.type == "ARMATURE" and m.object:
            return m.object
    return obj.parent


def followers(obj):
    """Other meshes of the same character - beard, moustache, teeth, tongue,
    eyelashes - that have shape keys with the same names. Keying them in step
    keeps a beard on the lip instead of floating while the mouth moves."""
    rig = _rig_of(obj)
    if rig is None:
        return []
    return [o for o in bpy.data.objects
            if o is not obj and o.type == "MESH" and _rig_of(o) is rig and _keys_of(o)]


def _mouth_key_count(o):
    """How many of this mesh's shape keys drive the mouth - the head has
    dozens, a beard or moustache only a few."""
    _, mapping = auto_mapping(o)
    used = {k for d in mapping.values() for k in d}
    return max(len(used), sum(len(v) for v in arkit_targets(o).values()))


def face_mesh(obj):
    """The mesh of this character that carries the mouth shapes - so picking
    the moustache or a shirt still finds the head (CC_Base_Body...)."""
    cands = [obj] + followers(obj)
    return max(cands, key=lambda o: (_mouth_key_count(o), o is obj))


def mapping_to_text(d):
    return ", ".join(f"{k}:{v:g}" for k, v in d.items())


def text_to_mapping(text):
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, val = part.rpartition(":")
        if not name:
            name, val = val, "1"
        try:
            out[name.strip()] = float(val)
        except ValueError:
            out[part] = 1.0
    return out


# ------------------------------------------------------------------ rhubarb

def _bin_dir():
    return bpy.utils.user_resource("DATAFILES", path="lipsync_rig", create=True)


def find_rhubarb(context):
    prefs = context.preferences.addons[__name__].preferences if __name__ in \
        context.preferences.addons else None
    if prefs and prefs.rhubarb_path and os.path.isfile(bpy.path.abspath(prefs.rhubarb_path)):
        return bpy.path.abspath(prefs.rhubarb_path)
    exe = "rhubarb.exe" if platform.system() == "Windows" else "rhubarb"
    for root, _, files in os.walk(_bin_dir()):
        if exe in files:
            return os.path.join(root, exe)
    return shutil.which("rhubarb")


def download_rhubarb():
    osname = {"Windows": "Windows", "Darwin": "macOS"}.get(platform.system(), "Linux")
    url = RHUBARB_URL.format(v=RHUBARB_VERSION, os=osname)
    dst = _bin_dir()
    zpath = os.path.join(dst, "rhubarb.zip")
    with urllib.request.urlopen(url, timeout=120) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dst)
    os.remove(zpath)
    exe = "rhubarb.exe" if osname == "Windows" else "rhubarb"
    for root, _, files in os.walk(dst):
        if exe in files:
            p = os.path.join(root, exe)
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return p
    raise RuntimeError("downloaded Rhubarb but found no executable in it")


def wav_for_rhubarb(path):
    """Rhubarb reads WAV and OGG only. Anything else is converted with
    Blender's audio library, or ffmpeg when that is on PATH."""
    if os.path.splitext(path)[1].lower() in (".wav", ".ogg"):
        return path, None
    tmp = os.path.join(tempfile.mkdtemp(prefix="lipsync_rig_"), "audio.wav")
    try:
        import aud
        aud.Sound(path).write(tmp, rate=16000, channels=1, format=aud.FORMAT_S16,
                              container=aud.CONTAINER_WAV, codec=aud.CODEC_PCM)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"{os.path.basename(path)}: use a WAV or OGG file")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000",
                        tmp], check=True)
    return tmp, tmp


def rhubarb_cmd(exe, wav, out_json, recognizer, dialog_file):
    cmd = [exe, "-f", "json", "-r", recognizer, "--extendedShapes", "GHX",
           "--machineReadable", "-o", out_json]
    if dialog_file:
        cmd += ["-d", dialog_file]
    return cmd + [wav]


# --------------------------------------------------------------- animation

def _fcurves(action):
    """F-curves of an action, for both the classic and the layered (4.4+) API."""
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


def apply_cues(obj, cues, mapping, start_frame, fps, intensity=1.0, rng=None,
               min_hold=1.0):
    """Keyframe the mapped shape keys for Rhubarb cues [(start_s, end_s, shape)].
    Existing keys of those shape keys inside the span are replaced, so running
    again redoes the line instead of piling keys on top. Returns key count."""
    keys = _keys_of(obj)
    targets = sorted({k for d in mapping.values() for k in d if k in keys})
    if not targets:
        raise RuntimeError(f"{obj.name}: none of the mapped shape keys exist on it")
    pts = []                        # (frame, shape)
    for s, e, shape in cues:
        f = start_frame + s * fps
        if rng and not (rng[0] <= f <= rng[1]):
            continue
        if pts and f - pts[-1][0] < min_hold:
            pts[-1] = (pts[-1][0], shape)       # cues shorter than a frame: keep the later
        else:
            pts.append((f, shape))
    if not pts:
        return 0
    if cues and (not rng or start_frame + cues[-1][1] * fps <= rng[1]):
        pts.append((start_frame + cues[-1][1] * fps, "X"))
    f0, f1 = pts[0][0] - 1, pts[-1][0] + 1

    sk = obj.data.shape_keys
    if sk.animation_data and sk.animation_data.action:
        for fc in _fcurves(sk.animation_data.action):
            name = fc.data_path.split('"')[1] if '"' in fc.data_path else ""
            if name in targets:
                for kp in reversed(list(fc.keyframe_points)):
                    if f0 <= kp.co[0] <= f1:
                        fc.keyframe_points.remove(kp, fast=True)
                fc.update()
    n = 0
    for f, shape in pts:
        want = mapping.get(shape, {})
        for t in targets:
            kb = keys[t]
            kb.value = max(kb.slider_min, min(kb.slider_max, want.get(t, 0.0) * intensity))
            kb.keyframe_insert("value", frame=f)
            n += 1
    if sk.animation_data and sk.animation_data.action:
        for fc in _fcurves(sk.animation_data.action):
            for kp in fc.keyframe_points:
                if f0 <= kp.co[0] <= f1:
                    kp.interpolation = "BEZIER"
                    kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
            fc.update()
    return n


# ------------------------------------------------------ studio AI model

MOUTH_ARKIT = ("jaw", "mouth", "cheekPuff")


def _words(arkit):
    w = re.findall(r"[A-Z]?[a-z]+", arkit)
    return [x.lower() for x in w]


def arkit_targets(obj):
    """{ARKit mouth channel: [shape keys on this mesh]} - exact ARKit names, or
    the same words in another order / with _L _R (Character Creator's
    Mouth_Press_L, Mouth_Roll_In_Lower_R, Mouth_Funnel_Up_L...)."""
    keys = list(_keys_of(obj))
    norm = {k: _norm(k) for k in keys}
    out = {}
    for a in _studio_names:
        if not a.startswith(MOUTH_ARKIT):
            continue
        words = _words(a)
        side = words[-1] if words[-1] in ("left", "right") else None
        core = [w for w in words if w not in ("left", "right")]
        hits = []
        for k, n in norm.items():
            if n == _norm(a):
                hits = [k]
                break
            if not all(w in n for w in core):
                continue
            if side and not (n.endswith(side[0]) or side in n):
                continue
            hits.append(k)
        if hits:
            out[a] = hits
    return out


_studio_names = [
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "cheekPuff",
]


def apply_curves(obj, names, frames, targets, start_frame, intensity=1.0, rng=None):
    """Key per-frame model values (names x frames) onto the mapped shape keys.
    Earlier keys of those shape keys in the span are replaced. Returns key count."""
    keys = _keys_of(obj)
    col = {n: i for i, n in enumerate(names)}
    per_key = {}                       # shape key -> values over frames (max of channels)
    for a, ks in targets.items():
        if a not in col:
            continue
        vals = [f[col[a]] for f in frames]
        for k in ks:
            prev = per_key.get(k)
            per_key[k] = vals if prev is None else [max(x, y) for x, y in zip(prev, vals)]
    if not per_key:
        return 0
    fr = [start_frame + i for i in range(len(frames))]
    keep = [i for i, f in enumerate(fr) if not rng or rng[0] <= f <= rng[1]]
    if not keep:
        return 0
    f0, f1 = fr[keep[0]], fr[keep[-1]]
    sk = obj.data.shape_keys
    if sk.animation_data and sk.animation_data.action:
        for fc in _fcurves(sk.animation_data.action):
            name = fc.data_path.split('"')[1] if '"' in fc.data_path else ""
            if name in per_key:
                for kp in reversed(list(fc.keyframe_points)):
                    if f0 - 0.5 <= kp.co[0] <= f1 + 0.5:
                        fc.keyframe_points.remove(kp, fast=True)
                fc.update()
    n = 0
    for k, vals in per_key.items():
        kb = keys[k]
        lo, hi = kb.slider_min, kb.slider_max
        first = True
        fc = None
        for i in keep:
            v = max(lo, min(hi, vals[i] * intensity))
            if first:
                kb.value = v
                kb.keyframe_insert("value", frame=fr[i])
                path = kb.path_from_id("value")
                fc = next(f for f in _fcurves(sk.animation_data.action) if f.data_path == path)
                start = len(fc.keyframe_points)
                fc.keyframe_points.add(len(keep) - 1)
                first = False
                j = start
                continue
            kp = fc.keyframe_points[j]
            kp.co = (fr[i], v)
            kp.interpolation = "BEZIER"
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"
            j += 1
        fc.update()
        n += len(keep)
    return n


def studio_paths(context):
    prefs = context.preferences.addons[__name__].preferences if __name__ in \
        context.preferences.addons else None
    if not prefs:
        return None, None, None
    return (bpy.path.abspath(prefs.studio_python), bpy.path.abspath(prefs.studio_dir),
            bpy.path.abspath(prefs.studio_model))


# -------------------------------------------------------------- properties

def _sound_strips(scene):
    se = scene.sequence_editor
    if not se:
        return []
    allx = getattr(se, "strips_all", None) or getattr(se, "sequences_all", [])
    return [s for s in allx if s.type == "SOUND"]


def _strip_items(self, context):
    items = [("", "(choose a sound strip)", "")]
    for s in _sound_strips(context.scene):
        items.append((s.name, s.name, f"starts at frame {s.frame_start:g}"))
    return items


def _obj_poll(self, obj):
    return obj.type == "MESH" and bool(_keys_of(obj))


class LSR_Character(bpy.types.PropertyGroup):
    obj: PointerProperty(name="Character", type=bpy.types.Object, poll=_obj_poll,
                         description="Mesh whose shape keys move the mouth")
    source: EnumProperty(name="Audio", items=[
        ("FILE", "File", "A dialogue audio file"),
        ("STRIP", "Sound strip", "A sound strip in the Video Sequencer; its position on "
                                 "the timeline is used as the start frame")], default="FILE")
    audio: StringProperty(name="Audio file", subtype="FILE_PATH")
    strip: EnumProperty(name="Strip", items=_strip_items)
    start_frame: IntProperty(name="Start frame", default=1,
                             description="Frame where the audio file starts")
    engine: EnumProperty(name="Engine", items=[
        ("RHUBARB", "Mouth shapes (Rhubarb)", "Rule-based mouth shapes, works out of the box"),
        ("STUDIO", "Studio AI model", "Your own model trained on the studio's lip-synced "
                                      "videos (set it up in the add-on preferences)")],
        default="RHUBARB")
    language: EnumProperty(name="Language", items=[
        ("phonetic", "Any language (Hindi...)", "Language-independent recognizer"),
        ("pocketSphinx", "English", "More precise for English dialogue")], default="phonetic")
    dialog: StringProperty(name="Dialogue text",
                           description="Optional: the words spoken (English) - improves accuracy")
    intensity: FloatProperty(name="Strength", default=1.0, min=0.0, max=2.0,
                             description="Scales every mouth shape")
    use_range: BoolProperty(name="Only in frame range", default=False,
                            description="Keyframe only where this character talks, when "
                                        "several characters share one audio track")
    range_start: IntProperty(name="From", default=1)
    range_end: IntProperty(name="To", default=250)
    preset: StringProperty(name="Detected rig")
    follow: BoolProperty(name="Move beard / teeth / tongue too", default=True,
                         description="Also key the same shape keys on the character's other "
                                     "meshes (beard, moustache, teeth, tongue, eyelashes)")
    show_map: BoolProperty(name="Mouth shapes", default=False)
    status: StringProperty()


for _s in SHAPES:
    LSR_Character.__annotations__[f"map_{_s}"] = StringProperty(
        name=_s, description=f"{SHAPE_HELP[_s]}. Shape keys and values, e.g. jawOpen:0.6, "
                             f"mouthFunnel:0.3")


def char_mapping(c):
    return {s: text_to_mapping(getattr(c, f"map_{s}")) for s in SHAPES}


def char_audio(c, scene):
    """(audio path, frame where its second 0 sits)."""
    if c.source == "STRIP":
        st = next((s for s in _sound_strips(scene) if s.name == c.strip), None)
        if not st or not st.sound:
            raise RuntimeError("choose a sound strip")
        return bpy.path.abspath(st.sound.filepath), st.frame_start
    if not c.audio:
        raise RuntimeError("choose an audio file")
    return bpy.path.abspath(c.audio), c.start_frame


# --------------------------------------------------------------- operators

class LSR_OT_add(bpy.types.Operator):
    bl_idname = "lipsync_rig.add"
    bl_label = "Add character"
    bl_description = "Add a character (the selected mesh, if it has shape keys)"

    def execute(self, context):
        sc = context.scene
        c = sc.lsr_chars.add()
        c.start_frame = sc.frame_start
        c.range_start, c.range_end = sc.frame_start, sc.frame_end
        ob = context.active_object
        if ob and ob.type == "MESH":
            ob = face_mesh(ob)          # the moustache was selected -> use the head
        if ob and _obj_poll(None, ob) and ob not in [x.obj for x in sc.lsr_chars]:
            c.obj = ob
            bpy.ops.lipsync_rig.auto_map(index=len(sc.lsr_chars) - 1)
        sc.lsr_index = len(sc.lsr_chars) - 1
        return {"FINISHED"}


class LSR_OT_remove(bpy.types.Operator):
    bl_idname = "lipsync_rig.remove"
    bl_label = "Remove character"

    def execute(self, context):
        sc = context.scene
        if 0 <= sc.lsr_index < len(sc.lsr_chars):
            sc.lsr_chars.remove(sc.lsr_index)
            sc.lsr_index = max(0, sc.lsr_index - 1)
        return {"FINISHED"}


class LSR_OT_auto_map(bpy.types.Operator):
    bl_idname = "lipsync_rig.auto_map"
    bl_label = "Detect mouth shape keys"
    bl_description = "Fill the mouth-shape table from the character's shape key names " \
                     "(ARKit, Oculus/ReadyPlayerMe, VRChat, Preston Blair, ...)"
    index: IntProperty(default=-1)

    def execute(self, context):
        sc = context.scene
        c = sc.lsr_chars[self.index if self.index >= 0 else sc.lsr_index]
        if not c.obj:
            self.report({"ERROR"}, "Choose the character's mesh first")
            return {"CANCELLED"}
        name, mapping = auto_mapping(c.obj)
        if not name:
            c.preset = "not recognised - type the shape key names below"
            self.report({"WARNING"}, "No known mouth shape keys found; fill the table by hand")
            return {"FINISHED"}
        c.preset = name
        for s in SHAPES:
            setattr(c, f"map_{s}", mapping_to_text(mapping.get(s, {})))
        c.show_map = True
        return {"FINISHED"}


class LSR_OT_download(bpy.types.Operator):
    bl_idname = "lipsync_rig.download"
    bl_label = "Download Rhubarb Lip Sync"
    bl_description = f"Download Rhubarb Lip Sync {RHUBARB_VERSION} (MIT, ~40 MB) once"

    def execute(self, context):
        try:
            p = download_rhubarb()
        except Exception as e:
            self.report({"ERROR"}, f"Download failed: {e}. Download it from github.com/"
                                   f"DanielSWolf/rhubarb-lip-sync and set its path in the "
                                   f"add-on preferences.")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Rhubarb installed: {p}")
        return {"FINISHED"}


class LSR_OT_generate(bpy.types.Operator):
    bl_idname = "lipsync_rig.generate"
    bl_label = "Generate lip sync"
    bl_description = "Listen to each character's dialogue and keyframe its mouth"
    only_active: BoolProperty(default=False)

    _timer = None

    def _jobs(self, context):
        sc = context.scene
        idx = [sc.lsr_index] if self.only_active else range(len(sc.lsr_chars))
        return [i for i in idx if 0 <= i < len(sc.lsr_chars) and sc.lsr_chars[i].obj]

    def _start(self, context, i):
        c = context.scene.lsr_chars[i]
        path, frame0 = char_audio(c, context.scene)
        if not os.path.isfile(path):
            raise RuntimeError(f"audio file not found: {path}")
        if c.engine == "STUDIO":
            return self._start_studio(context, i, c, path, frame0)
        if not self.exe:
            raise RuntimeError("Rhubarb Lip Sync not found - click 'Download Rhubarb'")
        wav, tmp = wav_for_rhubarb(path)
        work = tempfile.mkdtemp(prefix="lipsync_rig_")
        dialog = None
        if c.dialog.strip():
            dialog = os.path.join(work, "dialog.txt")
            with open(dialog, "w", encoding="utf-8") as f:
                f.write(c.dialog)
        out = os.path.join(work, "cues.json")
        cmd = rhubarb_cmd(self.exe, wav, out, c.language, dialog)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                             text=True, creationflags=flags)
        st = {"i": i, "p": p, "out": out, "frame0": frame0, "tmp": tmp, "progress": 0.0,
              "err": ""}

        def read():
            for line in p.stderr:
                try:
                    m = json.loads(line)
                except ValueError:
                    continue
                if m.get("type") == "progress":
                    st["progress"] = float(m.get("value", 0))
                elif m.get("type") == "failure":
                    st["err"] = m.get("reason", "")
        threading.Thread(target=read, daemon=True).start()
        c.status = "listening..."
        return st

    def _start_studio(self, context, i, c, path, frame0):
        py, d, model = studio_paths(context)
        script = os.path.join(d or "", "predict.py")
        for what, f in (("Python with torch", py), ("face_model folder/predict.py", script),
                        ("trained model", model)):
            if not f or not os.path.isfile(f):
                raise RuntimeError(f"Studio AI model: set '{what}' in the add-on preferences")
        sc = context.scene
        fps = sc.render.fps / sc.render.fps_base
        work = tempfile.mkdtemp(prefix="lipsync_rig_")
        out = os.path.join(work, "curves.json")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.Popen([py, script, model, path, out, "--fps", str(fps)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                             creationflags=flags)
        st = {"i": i, "p": p, "out": out, "frame0": frame0, "tmp": None, "progress": 0.5,
              "err": "", "kind": "studio"}

        def read():
            st["err"] = (p.stderr.read() or "").strip()[-400:]
        threading.Thread(target=read, daemon=True).start()
        c.status = "AI model listening..."
        return st

    def _finish_studio(self, context, st):
        c = context.scene.lsr_chars[st["i"]]
        if st["p"].returncode != 0 or not os.path.isfile(st["out"]):
            raise RuntimeError("Studio AI model failed: " + (st["err"] or
                               f"exit {st['p'].returncode}"))
        with open(st["out"], encoding="utf-8") as f:
            data = json.load(f)
        rng = (c.range_start, c.range_end) if c.use_range else None
        tg = arkit_targets(c.obj)
        if not tg:
            raise RuntimeError(f"{c.obj.name} has no ARKit-style mouth shape keys "
                               f"(jawOpen, mouthFunnel... or Jaw_Open, Mouth_Funnel_Up_L...)")
        n = apply_curves(c.obj, data["names"], data["frames"], tg, st["frame0"],
                         c.intensity, rng)
        moved = []
        if c.follow:
            for o in followers(c.obj):
                t2 = arkit_targets(o)
                if t2:
                    n += apply_curves(o, data["names"], data["frames"], t2, st["frame0"],
                                      c.intensity, rng)
                    moved.append(o.name)
        c.status = f"AI model: {len(data['frames'])} frames, {len(tg)} mouth channels, " \
                   f"{n} keys" + (f", +{len(moved)} meshes" if moved else "")
        return c.status

    def _finish(self, context, st):
        if st.get("kind") == "studio":
            return self._finish_studio(context, st)
        c = context.scene.lsr_chars[st["i"]]
        if st["tmp"]:
            shutil.rmtree(os.path.dirname(st["tmp"]), ignore_errors=True)
        if st["p"].returncode != 0 or not os.path.isfile(st["out"]):
            raise RuntimeError(st["err"] or f"Rhubarb exited with {st['p'].returncode}")
        with open(st["out"], encoding="utf-8") as f:
            cues = [(q["start"], q["end"], q["value"]) for q in json.load(f)["mouthCues"]]
        sc = context.scene
        fps = sc.render.fps / sc.render.fps_base
        rng = (c.range_start, c.range_end) if c.use_range else None
        mapping = char_mapping(c)
        n = apply_cues(c.obj, cues, mapping, st["frame0"], fps, c.intensity, rng)
        moved = []
        if c.follow:
            for o in followers(c.obj):
                have = _keys_of(o)
                sub = {s: {k: v for k, v in d.items() if k in have} for s, d in mapping.items()}
                if any(sub.values()):
                    n += apply_cues(o, cues, sub, st["frame0"], fps, c.intensity, rng)
                    moved.append(o.name)
        talk = sum(1 for q in cues if q[2] not in "X")
        c.status = f"done: {talk} mouth shapes, {n} keys" + \
            (f", +{len(moved)} meshes (beard/teeth...)" if moved else "")
        if moved:
            print("Lip-Sync Rig: also keyed", ", ".join(moved))
        return c.status

    def execute(self, context):
        """Blocking run - used from scripts and in background mode."""
        self.exe = find_rhubarb(context)
        if not self.exe and any(context.scene.lsr_chars[i].engine == "RHUBARB"
                                for i in self._jobs(context)):
            self.report({"ERROR"}, "Rhubarb Lip Sync not found - click 'Download Rhubarb'")
            return {"CANCELLED"}
        for i in self._jobs(context):
            try:
                st = self._start(context, i)
                st["p"].wait()
                self.report({"INFO"}, f"{context.scene.lsr_chars[i].obj.name}: "
                                      f"{self._finish(context, st)}")
            except Exception as e:
                context.scene.lsr_chars[i].status = f"error: {e}"
                self.report({"ERROR"}, str(e))
        return {"FINISHED"}

    def invoke(self, context, event):
        self.exe = find_rhubarb(context)
        if not self.exe and any(context.scene.lsr_chars[i].engine == "RHUBARB"
                                for i in self._jobs(context)):
            self.report({"ERROR"}, "Rhubarb Lip Sync not found - click 'Download Rhubarb'")
            return {"CANCELLED"}
        self.queue = self._jobs(context)
        if not self.queue:
            self.report({"ERROR"}, "Add a character and choose its mesh first")
            return {"CANCELLED"}
        self.cur = None
        self.done = 0
        self.total = len(self.queue)
        context.window_manager.progress_begin(0, 100)
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            if self.cur:
                self.cur["p"].kill()
            return self._end(context, {"CANCELLED"}, "cancelled")
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            if self.cur is None:
                if not self.queue:
                    return self._end(context, {"FINISHED"}, f"lip sync done for "
                                                           f"{self.total} character(s)")
                self.cur = self._start(context, self.queue.pop(0))
            elif self.cur["p"].poll() is not None:
                self._finish(context, self.cur)
                self.cur = None
                self.done += 1
            else:
                pct = 100 * (self.done + self.cur["progress"]) / self.total
                context.window_manager.progress_update(pct)
                context.scene.lsr_chars[self.cur["i"]].status = \
                    f"listening... {int(self.cur['progress'] * 100)}%"
        except Exception as e:
            if self.cur:
                context.scene.lsr_chars[self.cur["i"]].status = f"error: {e}"
            self.report({"ERROR"}, str(e))
            self.cur = None
        for area in context.screen.areas:
            area.tag_redraw()
        return {"RUNNING_MODAL"}

    def _end(self, context, result, msg):
        context.window_manager.event_timer_remove(self._timer)
        context.window_manager.progress_end()
        self.report({"INFO"}, msg)
        return result


# ---------------------------------------------------------------------- ui

class LSR_UL_chars(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop):
        row = layout.row(align=True)
        row.label(text=item.obj.name if item.obj else "(choose mesh)", icon="USER")
        row.label(text=item.status)


class LSR_PT_panel(bpy.types.Panel):
    bl_label = "Lip Sync"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lip Sync"

    def draw(self, context):
        sc = context.scene
        lay = self.layout
        if not find_rhubarb(context):
            box = lay.box()
            box.label(text="Needs Rhubarb Lip Sync (free)", icon="INFO")
            box.operator("lipsync_rig.download", icon="IMPORT")
        row = lay.row()
        row.template_list("LSR_UL_chars", "", sc, "lsr_chars", sc, "lsr_index", rows=3)
        col = row.column(align=True)
        col.operator("lipsync_rig.add", icon="ADD", text="")
        col.operator("lipsync_rig.remove", icon="REMOVE", text="")
        if 0 <= sc.lsr_index < len(sc.lsr_chars):
            c = sc.lsr_chars[sc.lsr_index]
            box = lay.box()
            box.prop(c, "obj")
            box.prop(c, "source", expand=True)
            if c.source == "FILE":
                box.prop(c, "audio")
                box.prop(c, "start_frame")
            else:
                box.prop(c, "strip")
            box.prop(c, "engine")
            if c.engine == "RHUBARB":
                box.prop(c, "language")
            box.prop(c, "dialog")
            box.prop(c, "intensity", slider=True)
            box.prop(c, "follow")
            r = box.row(align=True)
            r.prop(c, "use_range")
            if c.use_range:
                r = box.row(align=True)
                r.prop(c, "range_start")
                r.prop(c, "range_end")
            r = box.row()
            r.prop(c, "show_map", icon="TRIA_DOWN" if c.show_map else "TRIA_RIGHT",
                   emboss=False)
            r.operator("lipsync_rig.auto_map", text="Detect", icon="VIEWZOOM").index = -1
            if c.preset:
                box.label(text=f"Rig: {c.preset}")
            if c.show_map:
                for s in SHAPES:
                    rr = box.row()
                    rr.label(text=f"{s}  {SHAPE_HELP[s]}")
                    box.prop(c, f"map_{s}", text="")
            box.operator("lipsync_rig.generate", text="Lip sync this character",
                         icon="PLAY_SOUND").only_active = True
        lay.separator()
        lay.operator("lipsync_rig.generate", text="Lip sync ALL characters",
                     icon="OUTLINER_OB_SPEAKER").only_active = False


class LSR_Prefs(bpy.types.AddonPreferences):
    bl_idname = __name__
    rhubarb_path: StringProperty(name="Rhubarb executable", subtype="FILE_PATH",
                                 description="Leave empty to use the downloaded copy")
    studio_python: StringProperty(name="Python with torch", subtype="FILE_PATH",
                                  description="e.g. C:\\Projects\\3dLipsync\\lesgoo\\.venv\\Scripts\\python.exe")
    studio_dir: StringProperty(name="face_model folder", subtype="DIR_PATH",
                               description="The repository's lipsync-3d\\face_model folder")
    studio_model: StringProperty(name="Trained model (.pt)", subtype="FILE_PATH")

    def draw(self, context):
        self.layout.prop(self, "rhubarb_path")
        self.layout.operator("lipsync_rig.download", icon="IMPORT")
        box = self.layout.box()
        box.label(text="Studio AI model (engine 'Studio AI model')")
        box.prop(self, "studio_python")
        box.prop(self, "studio_dir")
        box.prop(self, "studio_model")


CLASSES = (LSR_Character, LSR_OT_add, LSR_OT_remove, LSR_OT_auto_map, LSR_OT_download,
           LSR_OT_generate, LSR_UL_chars, LSR_PT_panel, LSR_Prefs)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.lsr_chars = CollectionProperty(type=LSR_Character)
    bpy.types.Scene.lsr_index = IntProperty()


def unregister():
    del bpy.types.Scene.lsr_index
    del bpy.types.Scene.lsr_chars
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
