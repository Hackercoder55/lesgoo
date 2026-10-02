# Lip-Sync Studio — web app

A self-hosted replacement for sync.so. Upload a video, choose a face (or let it
pick), and get the lip-synced video back. Uses LatentSync on your own GPU, so
there's no per-video bill. Your media stays on the machine the app runs on.

What it does:

- **Clip mode**: one shot. The face is detected automatically, or you click on it.
  You can give a new dialogue track. If audio and video lengths differ, you
  pick cut, loop or bounce.
- **Full video (auto)**: a whole episode. **Analyze** finds where people
  speak, splits those parts at every shot change, and shows each character
  on screen in each part as a thumbnail. "Talking" marks the one whose
  mouth already moves most. Click to choose one or several characters per
  part, then run. Only those mouths are re-synced, and the result goes back
  into the untouched original, with the same frame count and audio.
- **Job queue + history**: one GPU job at a time, in order. You can cancel,
  run again, download or delete. A server restart resumes the queue.
- **Accounts**: users and admins, plus API keys for scripts and tools.
- **REST API**: works like sync.so's. Interactive docs are at `/v1/docs`.

## Easiest: Windows, two double-clicks

1. `SETUP.bat` (repository root), once. It installs ffmpeg, Python, the
   model's packages (plus the C++ build tools if needed) and the weights,
   all into this folder. Takes 20-40 min the first time. Safe to run again.
2. `START.bat`, every time. The browser opens on the site. Close the
   black window to stop it.

On a GPU under 10 GB the site starts with low-VRAM settings (guidance
1.0, 512 px crop).

## Option 1: on an animator's own PC (Windows), by hand

Do the one-time setup from `lipsync-3d/README.md` first: the Python env,
`requirements.txt`, the checkpoints and `yunet.onnx`. Then:

```
pip install -r lipsync-3d/webapp/requirements.txt
lipsync-3d\webapp\start_local.bat
```

The browser opens http://127.0.0.1:8000. Local mode has no login, and it is
only reachable from that PC.

The GPU needs about 8 GB for the 256px model. A 6 GB laptop GPU may run out
of memory. If it does, lower "Max crop px" in Settings and use fewer steps.

## Option 2: on a rented GPU (Vast.ai / RunPod)

1. Rent an instance from a PyTorch / CUDA 12 template, with a GPU of 12 GB
   or more (24 GB if you want the sharper 512px model). Open port 8000
   (Vast: add `-p 8000:8000` to the template's docker options).
2. In the instance's terminal:
   ```
   export ADMIN_PASSWORD='choose-a-strong-one'
   curl -sL https://raw.githubusercontent.com/hackercoder55/lesgoo/claude/sync-tool-creation-44ag4z/lipsync-3d/webapp/deploy/vast_setup.sh | bash
   ```
   For the 512px LatentSync 1.6 model, run with `MODEL=1.6` (~18 GB VRAM).
3. Open the instance's public address for port 8000. Sign in as `admin`,
   then go to **Users** to add the team.

There's also a Docker image: see `deploy/Dockerfile`.

**Cost.** A rented GPU is billed per hour while the instance is on, whether
or not anyone is using it. Stop the instance when you're done. Before you
destroy it, download the results or keep `/workspace/lipsync_data`.

## Users and passwords from the command line

```
python lipsync-3d/webapp/server.py adduser NAME [--admin]
python lipsync-3d/webapp/server.py passwd NAME
```

## API

Create a key in the site under **API keys**, then:

```
KEY=ls_...
curl -H "x-api-key: $KEY" -F file=@clip.mp4 http://HOST:8000/v1/assets             # -> {"id": ...}
curl -H "x-api-key: $KEY" -H "content-type: application/json" \
     -d '{"kind":"clip","video":"ASSET_ID"}' http://HOST:8000/v1/jobs               # -> {"id": ..., "status": "queued"}
curl -H "x-api-key: $KEY" http://HOST:8000/v1/jobs/JOB_ID                           # poll until "done"
curl -H "x-api-key: $KEY" -o out.mp4 http://HOST:8000/v1/jobs/JOB_ID/result
```

Job fields:

| Field | What it does |
|---|---|
| `kind` | `clip` or `auto` |
| `audio` | an asset id |
| `face` | `{"t": seconds, "box": [cx, cy, w, h]}`; leave it out to pick automatically |
| `mode` | `cut`, `loop` or `bounce` |
| `steps`, `guidance`, `seed`, `crop_max`, `score` | model and detector settings |
| `auto` | speech-detection settings: `threshold_db`, `merge_gap`, `pad_before`, `pad_after`, `min_speech`, `max_segment`, `scene_cut` |
| `plan`, `picks` | auto mode: the `plan` id from `POST /v1/assets/{id}/analyze` and `{"seg001": ["f1", "f2"], ...}`; parts left out of `picks` use the suggestion, `[]` skips a part |

## Testing without a GPU

```
python lipsync-3d/webapp/server.py --local --engine preview
```

This runs every step except the model: upload, face pick, tracking, speech
detection, splicing and the queue. The mouth comes back unchanged.

## Limits

- A character is only tracked inside one shot, so the same person in two
  shots shows up as two thumbnails. Choose them in each part.
- When several faces are chosen in one part, they are all synced to the
  same dialogue. That is right when they speak at the same time or in
  quick turns, and wrong for a listener. Leave listeners unselected.
- Speech is found by loudness, so loud music or effects can be taken as
  speech. Raise "Speech threshold" in Settings if that happens.
- LatentSync's own face finder is trained on real faces, so very stylised
  characters can fail with "face not detected". In that case pick a frame
  where the face is frontal, or raise "Max crop px".
