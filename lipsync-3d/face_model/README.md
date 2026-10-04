# Studio face model — audio → face animation, trained on your own videos

This trains the studio's own model, which turns dialogue audio into mouth
animation curves. The **Lip-Sync Rig** Blender add-on then keys those
curves onto the character's shape keys (engine **Studio AI model**).

**Where the training data comes from:** the lip-synced videos you already
have, such as sync.so results and approved renders. MediaPipe reads the
face on every frame as the 52 ARKit blendshapes (how far the jaw is open,
how round the lips are, …). Each frame's values, paired with the audio,
make one training example. No hand labelling is needed.

## 1. Install (once, in the repo's `.venv`)

```
pip install torch mediapipe numpy
```

`ffmpeg` must be on PATH (SETUP.bat installs it).

## 2. Extract training data

```
python lipsync-3d\face_model\extract.py "D:\SyncSoResults" "D:\FaceData"
```

- Point it at a folder of lip-synced videos; it searches subfolders too.
- Clips where a face is visible on less than 30% of frames are skipped.
  Frames without a face are left out of training.
- Clips with one speaker, facing the camera, train best.
- Already extracted clips are skipped, so you can re-run it as videos come in.

## 3. Train

```
python lipsync-3d\face_model\train.py "D:\FaceData" "D:\studio_face.pt" --epochs 60
```

- Clips are split by file: 10% are held out and never trained on.
- **"val mouth error"** is the average mouth error on those unseen clips;
  lower is better. The best epoch is saved.
- On an RTX 4090 (Vast.ai), a few hours of video trains in under an hour.
  On a CPU it works, but slowly.
- How much data: about 1 hour of dialogue to start, 5+ hours for good
  results.

## 4. Use it in Blender

Lip-Sync Rig → **Edit → Preferences → Add-ons → Lip-Sync Rig**, then under
*Studio AI model* set:

- **Python with torch:** `C:\Projects\3dLipsync\lesgoo\.venv\Scripts\python.exe`
- **face_model folder:** `C:\Projects\3dLipsync\lesgoo\lipsync-3d\face_model`
- **Trained model:** `D:\studio_face.pt`

In the Lip Sync panel, set **Engine → Studio AI model**, then click **Lip
sync this character**. It keys only mouth and jaw channels: ARKit names,
or Character Creator's `Jaw_Open`, `Mouth_Funnel_Up_L`, … . Eyes and brows
stay with the animator. Beard, moustache, teeth and tongue follow, as
with Rhubarb.

## Testing a model without Blender

```
python lipsync-3d\face_model\predict.py D:\studio_face.pt dialogue.wav out.json --fps 24
```

This writes per-frame values for the 52 channels.
