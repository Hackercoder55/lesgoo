"""Build and render a shot from a Mouseadon plan, inside Blender.

    blender [template.blend] --background --python generate.py -- \
        --plan plan.json --out shot.mp4 [--save-blend shot.blend] [--engine workbench]

Standalone on purpose: it runs with Blender's own Python and imports nothing
from the ``mouseadon`` package. It executes the plan's steps in order:

  layout    - find the plan's rig in the template, or build a proxy character
              from the recorded bone layout; ground, light, camera
  blocking  - key the main poses (KEYFRAME/EXTREME keys) stepped/CONSTANT
  splining  - add breakdowns and switch to the animators' interpolation
  polish    - restore the animators' exact handles, smooth the beat joins
  review    - frame the camera on the action and render
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys

import bpy
from mathutils import Matrix, Vector

BONE_PATH = re.compile(r'pose\.bones\["((?:[^"\\]|\\.)*)"\]\.(\w+)')
MAIN_KEY_TYPES = {"KEYFRAME", "EXTREME"}


def rig_signature(armature):
    names = sorted(bone.name for bone in armature.bones)
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


def rest_value(prop, index):
    if prop == "scale" or (prop == "rotation_quaternion" and index == 0):
        return 1.0
    return 0.0


def ensure_fcurve(obj, action, path, index, group):
    if hasattr(action, "fcurve_ensure_for_datablock"):  # Blender 4.4+ slotted actions
        return action.fcurve_ensure_for_datablock(obj, path, index=index, group_name=group)
    fcurve = action.fcurves.find(path, index=index)
    return fcurve or action.fcurves.new(path, index=index, action_group=group)


# ------------------------------------------------------------------ layout
def find_rig(sig):
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE" and rig_signature(obj.data) == sig:
            return obj
    return None


def _box_mesh(name, length, width):
    w = width / 2.0
    verts = [(x, y, z) for y in (0.0, length) for x in (-w, w) for z in (-w, w)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def build_proxy_rig(rig, collection):
    """Armature + one box per deform bone, from the recorded rest layout."""
    bones = rig.get("bones") or []
    if not bones:
        raise SystemExit("Plan has no rig layout and the template has no matching rig.")
    data = bpy.data.armatures.new("MouseadonRig")
    obj = bpy.data.objects.new("MouseadonRig", data)
    collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    edit = {}
    for bone in bones:
        eb = data.edit_bones.new(bone["name"])
        eb.head, eb.tail = Vector(bone["head"]), Vector(bone["tail"])
        if eb.length < 1e-4:
            eb.tail = eb.head + Vector((0, 0.05, 0))
        if bone.get("roll_z"):
            eb.align_roll(Vector(bone["roll_z"]))
        eb.use_deform = bone.get("deform", True)
        edit[bone["name"]] = eb
    for bone in bones:
        if bone.get("parent") in edit:
            edit[bone["name"]].parent = edit[bone["parent"]]
            edit[bone["name"]].use_connect = bone.get("connect", False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for bone in bones:
        if bone.get("rotation_mode"):
            obj.pose.bones[bone["name"]].rotation_mode = bone["rotation_mode"]

    material = bpy.data.materials.new("MouseadonProxy")
    material.diffuse_color = (0.85, 0.55, 0.3, 1.0)
    deform = [b for b in data.bones if b.use_deform] or list(data.bones)
    bpy.context.view_layer.update()
    for bone in deform:
        mesh = _box_mesh("proxy_" + bone.name, bone.length, max(bone.length * 0.3, 0.02))
        mesh.materials.append(material)
        part = bpy.data.objects.new("proxy_" + bone.name, mesh)
        collection.objects.link(part)
        part.parent = obj
        part.parent_type = "BONE"
        part.parent_bone = bone.name
        bpy.context.view_layer.update()
        part.matrix_world = obj.matrix_world @ bone.matrix_local
    return obj


def step_layout(ctx):
    scene = ctx["scene"]
    plan = ctx["plan"]
    collection = scene.collection
    arm = find_rig(plan["rig"]["sig"])
    ctx["built_proxy"] = arm is None
    if arm is None:
        arm = build_proxy_rig(plan["rig"], collection)
    ctx["arm"] = arm

    rest = [arm.matrix_world @ b.head_local for b in arm.data.bones] + \
           [arm.matrix_world @ b.tail_local for b in arm.data.bones]
    low = Vector((min(v.x for v in rest), min(v.y for v in rest), min(v.z for v in rest)))
    high = Vector((max(v.x for v in rest), max(v.y for v in rest), max(v.z for v in rest)))
    ctx["size"] = max((high - low).length, 0.1)

    if not any(o.type == "LIGHT" for o in scene.objects):
        sun = bpy.data.objects.new("MouseadonSun", bpy.data.lights.new("MouseadonSun", "SUN"))
        sun.data.energy = 3.0
        sun.rotation_euler = (math.radians(50), 0, math.radians(30))
        collection.objects.link(sun)
    if ctx["built_proxy"]:
        ground = bpy.data.objects.new("MouseadonGround", _ground_mesh(ctx["size"] * 40))
        ground.location.z = low.z
        collection.objects.link(ground)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("MouseadonWorld")
    scene.world.color = (0.05, 0.06, 0.08)
    if scene.camera is None:
        cam = bpy.data.objects.new("MouseadonCamera", bpy.data.cameras.new("MouseadonCamera"))
        collection.objects.link(cam)
        scene.camera = cam
        ctx["own_camera"] = True


def _ground_mesh(size):
    s = size / 2.0
    mesh = bpy.data.meshes.new("MouseadonGround")
    mesh.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    return mesh


# ---------------------------------------------------------------- keys
def plan_channels(ctx):
    """All planned keys per channel, with root-motion continuity between beats."""
    arm = ctx["arm"]
    bones = arm.data.bones
    channels = {}
    last_value = {}
    for b_index, beat in enumerate(ctx["plan"]["beats"]):
        for fc in beat["fcurves"]:
            match = BONE_PATH.match(fc["path"])
            bone = match.group(1) if match else ""
            if bone and bone not in bones:
                continue
            if not bone and not hasattr(arm, fc["path"].split(".")[0].split("[")[0]):
                continue
            prop = match.group(2) if match else fc["path"]
            key_id = (fc["path"], fc["idx"])
            offset = 0.0
            is_root = (not bone) or bones[bone].parent is None
            if prop == "location" and is_root and key_id in last_value and fc["keys"]:
                offset = last_value[key_id] - fc["keys"][0][1]
            entry = channels.setdefault(key_id, {"group": fc.get("group") or bone, "prop": prop, "keys": []})
            for i, key in enumerate(fc["keys"]):
                entry["keys"].append({
                    "frame": round(beat["start"] + key[0], 3), "value": key[1] + offset,
                    "interp": key[2], "type": key[3],
                    "hl": (beat["start"] + key[4], key[5] + offset), "hr": (beat["start"] + key[6], key[7] + offset),
                    "hlt": key[8], "hrt": key[9], "easing": key[10],
                    "beat": b_index, "join": i == 0 and b_index > 0,
                })
            if fc["keys"]:
                last_value[key_id] = fc["keys"][-1][1] + offset
        # channels animated earlier but not in this beat go back to rest
        present = {(fc["path"], fc["idx"]) for fc in beat["fcurves"]}
        for key_id, entry in channels.items():
            if key_id not in present and entry["keys"] and entry["keys"][-1]["beat"] < b_index:
                if entry["prop"] == "location" and key_id in last_value:
                    value = last_value[key_id]
                else:
                    value = rest_value(entry["prop"], key_id[1])
                f = beat["start"]
                entry["keys"].append({"frame": f, "value": value, "interp": "BEZIER", "type": "KEYFRAME",
                                      "hl": (f - 3, value), "hr": (f + 3, value), "hlt": "AUTO_CLAMPED",
                                      "hrt": "AUTO_CLAMPED", "easing": "AUTO", "beat": b_index, "join": True})
    return channels


def _insert(fcurve, key, interp):
    point = fcurve.keyframe_points.insert(key["frame"], key["value"], options={"FAST"})
    point.interpolation = interp
    point.type = key["type"]
    return point


def step_blocking(ctx):
    arm = ctx["arm"]
    if arm.animation_data is None:
        arm.animation_data_create()
    action = bpy.data.actions.new("mouseadon_shot")
    arm.animation_data.action = action
    ctx["action"] = action
    ctx["channels"] = plan_channels(ctx)
    ctx["fcurves"] = {}
    for key_id, entry in ctx["channels"].items():
        fcurve = ensure_fcurve(arm, action, key_id[0], key_id[1], entry["group"])
        ctx["fcurves"][key_id] = fcurve
        main = [k for k in entry["keys"] if k["type"] in MAIN_KEY_TYPES] or entry["keys"]
        for key in main:
            _insert(fcurve, key, "CONSTANT")
        fcurve.update()


def step_splining(ctx):
    for key_id, entry in ctx["channels"].items():
        fcurve = ctx["fcurves"][key_id]
        for key in entry["keys"]:
            if key["type"] not in MAIN_KEY_TYPES:
                _insert(fcurve, key, key["interp"])
        by_frame = {k["frame"]: k for k in entry["keys"]}
        for point in fcurve.keyframe_points:
            key = by_frame.get(round(point.co[0], 3))
            point.interpolation = key["interp"] if key else "BEZIER"
            point.handle_left_type = point.handle_right_type = "AUTO_CLAMPED"
        fcurve.update()


def step_polish(ctx):
    for key_id, entry in ctx["channels"].items():
        fcurve = ctx["fcurves"][key_id]
        by_frame = {k["frame"]: k for k in entry["keys"]}
        for point in fcurve.keyframe_points:
            key = by_frame.get(round(point.co[0], 3))
            if key is None or key["join"]:
                continue  # beat joins keep automatic handles for a smooth blend
            point.easing = key["easing"]
            point.handle_left_type = "FREE"
            point.handle_right_type = "FREE"
            point.handle_left = key["hl"]
            point.handle_right = key["hr"]
            point.handle_left_type = key["hlt"]
            point.handle_right_type = key["hrt"]
        fcurve.update()


# ---------------------------------------------------------------- review
def _engine_id(name):
    items = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    wanted = {"workbench": ["BLENDER_WORKBENCH"], "eevee": ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
              "cycles": ["CYCLES"]}.get(name, [name])
    for engine in wanted:
        if engine in items:
            return engine
    raise SystemExit("Render engine %r not available (have %s)" % (name, ", ".join(sorted(items))))


def frame_camera(ctx):
    scene, arm = ctx["scene"], ctx["arm"]
    points = []
    step = max(1, (scene.frame_end - scene.frame_start) // 24)
    for frame in range(scene.frame_start, scene.frame_end + 1, step):
        scene.frame_set(frame)
        points.extend(arm.matrix_world @ pb.head for pb in arm.pose.bones)
    scene.frame_set(scene.frame_start)
    low = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    high = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (low + high) / 2.0
    size = ctx["size"]
    travel = (high - low).length
    cam = scene.camera
    direction = Vector((0.55, -1.0, 0.3)).normalized()
    fov = cam.data.angle
    follow = travel > 2.5 * size
    extent = size if follow else max(travel, size)
    distance = extent * 0.75 / math.tan(fov / 2.0) + size
    if follow:
        root = next((b for b in arm.pose.bones if b.parent is None), None)
        target = bpy.data.objects.new("MouseadonFollow", None)
        scene.collection.objects.link(target)
        constraint = target.constraints.new("COPY_LOCATION")
        constraint.target = arm
        if root is not None:
            constraint.subtarget = root.name
        constraint.use_z = False
        target.location.z = center.z
        cam.parent = target
        cam.location = direction * distance
    else:
        cam.location = center + direction * distance
        target = bpy.data.objects.new("MouseadonLookAt", None)
        target.location = center
        scene.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"


def step_review(ctx):
    if ctx.get("own_camera"):
        frame_camera(ctx)
    args = ctx["args"]
    if not args.out:
        return
    scene = ctx["scene"]
    render = scene.render
    render.engine = _engine_id(args.engine)
    if render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.device = "CPU"
    elif render.engine.startswith("BLENDER_EEVEE") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = args.samples
    elif render.engine == "BLENDER_WORKBENCH":
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
    render.resolution_percentage = args.percent
    out = os.path.abspath(args.out)
    if out.lower().endswith((".mp4", ".mov", ".mkv")):
        render.image_settings.file_format = "FFMPEG"
        render.ffmpeg.format = "MPEG4" if out.lower().endswith(".mp4") else ("QUICKTIME" if out.lower().endswith(".mov") else "MKV")
        render.ffmpeg.codec = "H264"
        render.ffmpeg.constant_rate_factor = "MEDIUM"
        render.filepath = out
    else:
        render.image_settings.file_format = "PNG"
        render.filepath = os.path.join(out, "frame_")
    os.makedirs(os.path.dirname(render.filepath) or ".", exist_ok=True)
    bpy.ops.render.render(animation=True)


STEPS = {"layout": step_layout, "blocking": step_blocking, "splining": step_splining,
         "polish": step_polish, "review": step_review}


def run(plan, args):
    scene = bpy.context.scene
    scene.render.fps = int(round(plan["fps"]))
    scene.render.fps_base = scene.render.fps / plan["fps"]
    scene.frame_start = plan["frame_start"]
    scene.frame_end = min(plan["frame_end"], plan["frame_start"] + args.frames - 1) if args.frames else plan["frame_end"]
    scene.render.resolution_x, scene.render.resolution_y = plan["resolution"]
    ctx = {"scene": scene, "plan": plan, "args": args}
    steps = plan.get("steps") or list(STEPS)
    if "layout" not in steps:
        steps = ["layout"] + steps
    for name in steps:
        print("[mouseadon] step:", name)
        STEPS[name](ctx)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    return ctx


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="generate.py")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", help="video file (.mp4/.mov/.mkv) or folder for PNG frames")
    parser.add_argument("--save-blend", help="also save the generated scene for animators to review")
    parser.add_argument("--engine", default="eevee", help="workbench | eevee | cycles")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--percent", type=int, default=100, help="resolution percentage")
    parser.add_argument("--frames", type=int, default=0, help="render only the first N frames")
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parse_args(argv)
    with open(args.plan, encoding="utf-8") as handle:
        plan = json.load(handle)
    return run(plan, args)


if __name__ == "__main__":
    main()
