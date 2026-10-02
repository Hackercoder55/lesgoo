# Lip-Sync Studio - one-time setup for Windows. Run SETUP.bat (double-click).
# Installs everything into this repository: ffmpeg, Python 3.10 (via uv),
# a .venv with LatentSync's packages, and the model weights. Safe to run
# again: whatever is already there is kept.

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$Venv = Join-Path $Repo ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
Set-Location $Repo

function Say($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "`nSETUP STOPPED: $m" -ForegroundColor Red; exit 1 }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                "$env:USERPROFILE\.local\bin"
}
function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Say "Checking the NVIDIA GPU"
if (Have "nvidia-smi") {
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
} else {
    Fail "nvidia-smi not found. Install or update the NVIDIA driver from nvidia.com, then run SETUP again."
}

Say "ffmpeg"
Refresh-Path
if (-not (Have "ffmpeg")) {
    if (-not (Have "winget")) { Fail "winget is missing - install 'App Installer' from the Microsoft Store, then run SETUP again." }
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
    if (-not (Have "ffmpeg")) {
        # winget puts it under Links; make sure this session sees it
        $env:Path += ";$env:LOCALAPPDATA\Microsoft\WinGet\Links"
    }
    if (-not (Have "ffmpeg")) { Fail "ffmpeg installed but not found yet. Close this window and run SETUP again." }
}
ffmpeg -version | Select-Object -First 1

Say "uv (Python installer)"
if (-not (Have "uv")) {
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    Refresh-Path
    if (-not (Have "uv")) { Fail "uv did not install. Check the internet connection and run SETUP again." }
}

Say "Python 3.10 environment in $Venv"
if (-not (Test-Path $Py)) {
    uv venv --python 3.10 $Venv
    if ($LASTEXITCODE) { Fail "could not create the Python environment." }
}

function Install-Reqs {
    # Out-Host: otherwise uv's output becomes the function's return value
    uv pip install --python $Py -r requirements.txt --index-strategy unsafe-best-match | Out-Host
    return $LASTEXITCODE
}

Say "Installing LatentSync packages (torch is ~2.5 GB - this takes a while)"
if (Install-Reqs) {
    # almost always insightface, which needs a C++ compiler on Windows
    Say "A package needs the Microsoft C++ build tools - installing them (several GB, ~10-20 min)"
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-source-agreements `
        --accept-package-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    if (Install-Reqs) {
        Fail "packages still did not install. Restart the PC (the build tools need it), then run SETUP again. If it fails again, send a screenshot of the red error above."
    }
}
uv pip install --python $Py -r lipsync-3d\webapp\requirements.txt "huggingface_hub[cli]"
if ($LASTEXITCODE) { Fail "could not install the web app packages." }

Say "Checking torch can use the GPU"
& $Py -c "import torch,sys; ok=torch.cuda.is_available(); print(torch.__version__, ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE) {
    Say "torch has no CUDA - reinstalling the CUDA build"
    uv pip install --python $Py torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --reinstall
    & $Py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE) { Fail "torch still cannot see the GPU. Update the NVIDIA driver, restart, and run SETUP again." }
}

Say "Downloading model weights (LatentSync 1.5, VAE, face models)"
& $Py -c @"
from huggingface_hub import hf_hub_download, snapshot_download
for f in ('latentsync_unet.pt', 'whisper/tiny.pt'):
    print(hf_hub_download('ByteDance/LatentSync-1.5', f, local_dir='checkpoints'))
snapshot_download('stabilityai/sd-vae-ft-mse')
from insightface.app import FaceAnalysis
FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'], root='checkpoints/auxiliary',
             providers=['CPUExecutionProvider'])
print('weights ok')
"@
if ($LASTEXITCODE) { Fail "a model download failed. Check the internet connection and run SETUP again (finished downloads are kept)." }

$Yunet = "lipsync-3d\models\yunet.onnx"
if (-not (Test-Path $Yunet) -or (Get-Item $Yunet).Length -lt 100000) {
    New-Item -ItemType Directory -Force lipsync-3d\models | Out-Null
    Invoke-WebRequest -UseBasicParsing -OutFile $Yunet `
        https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
}

Say "Final check"
& $Py -c "import diffusers, omegaconf, insightface, decord, einops, accelerate, DeepCache, soundfile, cv2, fastapi, uvicorn; print('all packages ok')"
if ($LASTEXITCODE) { Fail "a package is missing - send a screenshot of the error above." }

Write-Host "`nSETUP DONE. Double-click START.bat to open Lip-Sync Studio." -ForegroundColor Green
