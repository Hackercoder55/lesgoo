# Build from the repository root:
#   docker build -f lipsync-3d/webapp/deploy/Dockerfile -t lipsync-studio .
#   docker run --gpus all -p 8000:8000 -v lipsync_data:/data -e ADMIN_PASSWORD=change-me lipsync-studio
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt lipsync-3d/webapp/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn python-multipart "huggingface_hub[cli]"
# model weights are fetched at build time, never while processing
RUN huggingface-cli download ByteDance/LatentSync-1.5 latentsync_unet.pt whisper/tiny.pt \
      --local-dir /app/checkpoints \
 && mkdir -p /app/lipsync-3d/models \
 && curl -sL -o /app/lipsync-3d/models/yunet.onnx \
      https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
COPY . .
ENV LIPSYNC_DATA=/data
EXPOSE 8000
CMD ["python", "lipsync-3d/webapp/server.py", "--host", "0.0.0.0", "--port", "8000"]
