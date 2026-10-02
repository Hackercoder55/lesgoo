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
   (`viseme_aa`…), VRChat (`vrc.v_aa`…), Preston Blair (`AI`, `E`, `O`,
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
