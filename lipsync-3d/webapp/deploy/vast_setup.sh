#!/bin/bash
# Lip-Sync Studio on a rented GPU (Vast.ai, RunPod, any Linux box with CUDA).
#
#   Vast.ai: pick a "PyTorch (cuda 12.x)" template, a GPU with >= 12 GB
#   (24 GB for the 512px model), open port 8000 in the template
#   ("-p 8000:8000"), then in the instance terminal:
#
#     curl -sL https://raw.githubusercontent.com/hackercoder55/lesgoo/claude/sync-tool-creation-44ag4z/lipsync-3d/webapp/deploy/vast_setup.sh | bash
#
#   Set ADMIN_PASSWORD first to choose the admin password; otherwise one is
#   printed on the first start. The model is chosen from the GPU: 1.6 (512px)
#   on 20 GB or more, else 1.5 (256px); MODEL=1.5 or MODEL=1.6 forces one.
#   Run it again after a git pull to update and restart.
set -e
BRANCH=${BRANCH:-claude/sync-tool-creation-44ag4z}
DIR=${DIR:-/workspace/lesgoo}
# 1.6 = 512px, sharper mouths, ~18 GB VRAM; 1.5 = 256px, ~8 GB
MODEL_SET=${MODEL:+1}
MODEL=${MODEL:-1.6}
VRAM_ALL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -z "${MODEL_SET:-}" ] && [ "${VRAM_ALL:-0}" -lt 20000 ]; then MODEL=1.5; fi
PORT=${PORT:-8000}

command -v ffmpeg >/dev/null || (apt-get update && apt-get install -y ffmpeg libgl1)
[ -d "$DIR/.git" ] || git clone -b "$BRANCH" https://github.com/hackercoder55/lesgoo.git "$DIR"
cd "$DIR"
git pull --ff-only || true
pip install -q -r requirements.txt -r lipsync-3d/webapp/requirements.txt
pip install -q "huggingface_hub[cli]"

# each model version in its own folder: 1.5 weights under the 512px config
# (or the reverse) give warped, smeared faces
CKPT="checkpoints/v$MODEL/latentsync_unet.pt"
[ -f "$CKPT" ] || huggingface-cli download "ByteDance/LatentSync-$MODEL" latentsync_unet.pt \
    --local-dir "checkpoints/v$MODEL"
[ -f checkpoints/whisper/tiny.pt ] || huggingface-cli download "ByteDance/LatentSync-$MODEL" \
    whisper/tiny.pt --local-dir checkpoints
python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/sd-vae-ft-mse')"

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$MODEL" = "1.6" ] && [ "${VRAM:-0}" -lt 20000 ]; then
  echo "!! MODEL=1.6 needs about 18 GB of GPU memory; this GPU has ${VRAM} MB."
  echo "!! It may run out of memory - use MODEL=1.5 or rent a 24 GB GPU."
fi
mkdir -p lipsync-3d/models
[ -s lipsync-3d/models/yunet.onnx ] || curl -sL -o lipsync-3d/models/yunet.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

[ -s lipsync-3d/models/sface.onnx ] || curl -sL -o lipsync-3d/models/sface.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

CONFIG=configs/unet/stage2.yaml
[ "$MODEL" = "1.6" ] && CONFIG=configs/unet/stage2_512.yaml
export LIPSYNC_DATA=${LIPSYNC_DATA:-/workspace/lipsync_data}
# InfiniteTalk (if installed): keep only part of its 14B model on GPUs under 40 GB
export INFINITETALK_DIR=${INFINITETALK_DIR:-/workspace/InfiniteTalk}
if [ "${VRAM:-0}" -lt 40000 ]; then export INFINITETALK_LOWVRAM=1; else export INFINITETALK_LOWVRAM=0; fi
pkill -f "lipsync-3d/webapp/server.py" 2>/dev/null && sleep 2
echo "starting LatentSync $MODEL on port $PORT (data in $LIPSYNC_DATA)"
nohup python lipsync-3d/webapp/server.py --host 0.0.0.0 --port "$PORT" \
  --ls-config "$CONFIG" --ls-ckpt "$CKPT" > /workspace/lipsync.log 2>&1 &
sleep 8
tail -n 20 /workspace/lipsync.log
echo
echo "Open the instance's public address for port $PORT in your browser."
echo "Logs: tail -f /workspace/lipsync.log"
