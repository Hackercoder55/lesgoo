#!/bin/bash
# Adds the InfiniteTalk engine (MeiGen-AI/InfiniteTalk, Apache-2.0) next to
# LatentSync on a rented GPU, then restarts the site with both engines.
#
#   bash lipsync-3d/webapp/deploy/infinitetalk_setup.sh
#
# Needs: a CUDA GPU (A100/H100 80 GB is comfortable; 24 GB works with the
# low-VRAM mode the site turns on by itself, but slowly) and ~120 GB of free
# disk for the 14B base model. First run downloads ~80 GB.
set -e
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
IT=${INFINITETALK_DIR:-/workspace/InfiniteTalk}

FREE=$(df -Pm "$(dirname "$IT")" | awk 'NR==2{print $4}')
if [ "${FREE:-0}" -lt 120000 ]; then
  echo "!! only $((FREE/1024)) GB free disk; InfiniteTalk needs ~120 GB. Rent a bigger disk."
  exit 1
fi

command -v ffmpeg >/dev/null || (apt-get update && apt-get install -y ffmpeg)
[ -d "$IT/.git" ] || git clone https://github.com/MeiGen-AI/InfiniteTalk.git "$IT"
cd "$IT"
git pull --ff-only || true

# its own environment: torch 2.4.1 + flash-attn 2.7.4, unlike LatentSync's 2.5.1
pip install -q uv
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
PY="$IT/.venv/bin/python"
uv pip install --python "$PY" torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$PY" xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$PY" "misaki[en]" ninja psutil packaging wheel setuptools
# flash-attn's installer fetches a prebuilt wheel for this torch/CUDA when one exists
uv pip install --python "$PY" flash_attn==2.7.4.post1 --no-build-isolation
uv pip install --python "$PY" -r requirements.txt librosa soundfile "huggingface_hub[cli]"

HF="$IT/.venv/bin/huggingface-cli"
[ -f weights/Wan2.1-I2V-14B-480P/config.json ] || \
  "$HF" download Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P
[ -f weights/chinese-wav2vec2-base/config.json ] || \
  "$HF" download TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base
[ -f weights/chinese-wav2vec2-base/model.safetensors ] || \
  "$HF" download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 \
    --local-dir weights/chinese-wav2vec2-base
[ -f weights/InfiniteTalk/single/infinitetalk.safetensors ] || \
  "$HF" download MeiGen-AI/InfiniteTalk single/infinitetalk.safetensors --local-dir weights/InfiniteTalk
# 8-step LoRA: ~5x faster ("fast" quality in the site)
if [ ! -f weights/Wan2.1_I2V_14B_FusionX_LoRA.safetensors ]; then
  "$HF" download vrgamedevgirl84/Wan14BT2VFusioniX FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors \
    --local-dir weights/fusionx && \
  mv weights/fusionx/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors weights/ || \
  echo "!! FusionX LoRA download failed - InfiniteTalk will run at the slower 40 steps"
fi

echo "InfiniteTalk ready in $IT. Restarting the site with both engines..."
cd "$REPO"
bash lipsync-3d/webapp/deploy/vast_setup.sh
