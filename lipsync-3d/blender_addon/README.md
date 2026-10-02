# Lip-Sync Rig — Blender add-on

Turns dialogue audio into mouth keyframes on your character's **own shape
keys**. Nothing is generated as pixels, so the face never warps, whatever
the head or camera does, and every key can be tweaked afterwards in the
Graph Editor.

It works for several characters in one scene. Each character gets its own
audio, or shares one track and is limited to a frame range.

## Install

1. Blender (3.6 or newer): **Edit → Preferences → Add-ons → Install…**
   (in 4.2+, use the **⌄** menu → *Install from Disk…*). Pick
   `lipsync_rig.py`, then tick **Lip-Sync Rig**.
2. In the 3D View press **N** and open the **Lip Sync** tab. Click
   **Download Rhubarb** once. Rhubarb Lip Sync 1.14 is MIT-licensed and
   about 40 MB.

## Use

1. Select a character mesh that has mouth shape keys, then click **+**.
2. The add-on detects the shape keys by name: ARKit (`jawOpen`,
   `mouthFunnel`…), Oculus / ReadyPlayerMe / Character Creator
   (`viseme_aa`…), **Character Creator 4 / 3 / iClone** (`V_Open`, `V_Explosive`…, `AE`, `B_M_P`…), VRChat (`vrc.v_aa`…), Preston Blair (`AI`, `E`, `O`,
   `MBP`…), or keys named `A`…`X`. If your names differ, open **Mouth
   shapes** and type them in, e.g. `JawOpen:0.6, LipsFunnel:0.3`.
3. Pick the audio: a file plus its start frame, or a **sound strip** from
   the Video Sequencer, which uses its position on the timeline.
4. Language: **Any language** for Hindi and others, or **English**. For
   English, pasting the dialogue text improves accuracy.
5. Click **Lip sync this character**, or **Lip sync ALL characters** for
   the whole list. Esc cancels.

Running it again replaces that character's earlier lip-sync keys in the
same frames; it does not pile new ones on top.

## Beard, moustache, teeth, tongue

Character Creator characters are split into several meshes, and the beard,
moustache, teeth, tongue and eyelashes carry the same shape-key names as the
head. With **Move beard / teeth / tongue too** ticked (the default), every
mesh bound to the same rig gets the same keys, so a moustache stays on the
lip. If you select the moustache or a shirt and click **+**, the add-on
picks the head mesh (e.g. `CC_Base_Body`) by itself.

## Several characters, one audio track

Add every character, then for each one tick **Only in frame range** and
set the frames where it talks. With a separate audio file per character,
no range is needed.

## Mouth shapes

| Shape | Mouth | Sounds |
|---|---|---|
| A | closed | P, B, M |
| B | slightly open, teeth together | K, S, T, EE |
| C | open | EH, AE |
| D | wide open | AA |
| E | slightly rounded | AO, ER |
| F | puckered | OO, OW, W |
| G | teeth on lower lip | F, V |
| H | tongue up | long L |
| X | rest | pauses |

## Limits

- The add-on drives **shape keys**. A face rig built only from bones
  (no shape keys) needs shape keys or drivers on top first.
- **Strength** scales every shape. Lower it for subtle characters and
  raise it for cartoon ones.

---

# Studio Recorder — collecting training data from your animators

`studio_recorder.py` is a second add-on. It is installed on every
animator's Blender and quietly records **how they animate**, so that an
animation model can later be trained on the studio's own work.

It records Blender data only, never the screen, camera or microphone:

| When | What |
|---|---|
| every save (only if the animation changed) | the full animation: every animated bone, shape key and property, with every keyframe, interpolation and handles |
| **Mark shot final** button | the same, labelled as the approved take |
| always, with each snapshot | shot id and script line (typed in the panel), dialogue sound strips, text strips, markers, cameras, linked character assets, fps, frame range |
| every 5 s | the Blender tools used (keyframe insert, rotate, graph-editor tools, …), in order |
| every minute | whether the file is being worked on, which gives the time spent per shot |

Data goes to one `.jsonl.gz` file per animator per day. Set the folder in
the add-on preferences; a shared network folder gathers everyone's data
in one place.

## Setup on each animator's PC

1. Install `studio_recorder.py` like any add-on and tick **Studio Recorder**.
2. In its preferences, set **Data folder** (e.g. `\\server\StudioData`) and
   **Animator name**.
3. In the 3D View, press N and open the **Recorder** tab. For each shot,
   type the **Shot** id and the **Script line**. When the shot is approved,
   click **Mark shot final**.

**Tell your animators about it.** The panel shows "Recording" or "Paused",
and they can pause it at any time.

## Old finished projects (data you already have)

```
python harvest_all.py "C:\Projects\Episodes" "D:\StudioData" --blender "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

This takes one snapshot of every animated scene in every `.blend` file
under the folder. Files already harvested are skipped, so it can be
re-run as projects grow.
