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
#   printed on the first start. MODEL=1.6 uses the 512px model (needs ~18 GB).
set -e
BRANCH=${BRANCH:-claude/sync-tool-creation-44ag4z}
DIR=${DIR:-/workspace/lesgoo}
MODEL=${MODEL:-1.5}
PORT=${PORT:-8000}

command -v ffmpeg >/dev/null || (apt-get update && apt-get install -y ffmpeg libgl1)
[ -d "$DIR/.git" ] || git clone -b "$BRANCH" https://github.com/hackercoder55/lesgoo.git "$DIR"
cd "$DIR"
git pull --ff-only || true
pip install -q -r requirements.txt -r lipsync-3d/webapp/requirements.txt
pip install -q "huggingface_hub[cli]"

if [ ! -f checkpoints/latentsync_unet.pt ]; then
  huggingface-cli download "ByteDance/LatentSync-$MODEL" latentsync_unet.pt whisper/tiny.pt \
    --local-dir checkpoints
fi
mkdir -p lipsync-3d/models
[ -s lipsync-3d/models/yunet.onnx ] || curl -sL -o lipsync-3d/models/yunet.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

CONFIG=configs/unet/stage2.yaml
[ "$MODEL" = "1.6" ] && CONFIG=configs/unet/stage2_512.yaml
export LIPSYNC_DATA=${LIPSYNC_DATA:-/workspace/lipsync_data}
echo "starting on port $PORT (data in $LIPSYNC_DATA)"
nohup python lipsync-3d/webapp/server.py --host 0.0.0.0 --port "$PORT" \
  --ls-config "$CONFIG" > /workspace/lipsync.log 2>&1 &
sleep 8
tail -n 20 /workspace/lipsync.log
echo
echo "Open the instance's public address for port $PORT in your browser."
echo "Logs: tail -f /workspace/lipsync.log"
