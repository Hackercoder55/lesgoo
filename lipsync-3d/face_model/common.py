"""Shared bits of the studio face model: audio features, the face channels it
predicts, and the network. No librosa / torchaudio needed - numpy + torch."""

import subprocess

import numpy as np

SR = 16000
HOP = 160                 # 10 ms -> 100 feature frames per second
N_FFT = 512
WIN = 400
N_MELS = 80

# The 52 ARKit blendshapes MediaPipe reports. The model predicts all of them;
# the mouth ones get most of the loss weight.
ARKIT = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
MOUTH = [i for i, n in enumerate(ARKIT) if n.startswith(("jaw", "mouth", "cheekPuff"))]


def load_audio(path):
    """Mono float32 at 16 kHz from any audio or video file (ffmpeg)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                          "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def _mel_filters():
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = mel2hz(np.linspace(hz2mel(20), hz2mel(7600), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, 1 / SR)
    fb = np.zeros((N_MELS, len(bins)), np.float32)
    for i in range(N_MELS):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


_FB = _mel_filters()


def log_mel(audio):
    """(frames, 80) log-mel at 100 frames/s, frame i centred on i*10 ms."""
    pad = np.pad(audio, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n = 1 + (len(pad) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    win = np.zeros(N_FFT, np.float32)
    off = (N_FFT - WIN) // 2
    win[off:off + WIN] = np.hanning(WIN)
    spec = np.abs(np.fft.rfft(pad[idx] * win, axis=1)) ** 2
    mel = np.log(spec @ _FB.T + 1e-6)
    return mel.astype(np.float32)


def model(n_out=len(ARKIT)):
    import torch.nn as nn

    class AudioToFace(nn.Module):
        """log-mel at 100 Hz -> face channels at 100 Hz (sampled at the video's
        frame times afterwards). ~0.6 s of context each side: the mouth
        starts moving before a sound is heard."""

        def __init__(self):
            super().__init__()
            ch = 192
            layers, c_in = [], N_MELS
            for d in (1, 2, 4, 8):
                layers += [nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d),
                           nn.BatchNorm1d(ch), nn.GELU()]
                c_in = ch
            self.conv = nn.Sequential(*layers)
            self.gru = nn.GRU(ch, 128, num_layers=2, batch_first=True, bidirectional=True,
                              dropout=0.1)
            self.head = nn.Linear(256, n_out)

        def forward(self, mel):                          # (B, T, 80)
            x = self.conv(mel.transpose(1, 2)).transpose(1, 2)
            x, _ = self.gru(x)
            return self.head(x).sigmoid()                # (B, T, n_out) in 0..1

    return AudioToFace()


def sample_at(seq100, fps, n_frames):
    """Values at 100 Hz -> values at each video frame time (linear)."""
    t = np.arange(n_frames) / fps * 100
    i0 = np.clip(np.floor(t).astype(int), 0, len(seq100) - 1)
    i1 = np.clip(i0 + 1, 0, len(seq100) - 1)
    w = (t - np.floor(t))[:, None]
    return seq100[i0] * (1 - w) + seq100[i1] * w
