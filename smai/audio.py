"""Audio decoding + multi-scale log-mel features (torch, CPU-friendly).

Feature layout: (T, 3, 80) float16 at exactly 100 fps, sr=16000,
FFT sizes 512/1024/2048, 80 mel bands 27.5Hz..8kHz.
"""
from __future__ import annotations

import math

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SR = 16000
HOP = 160          # 100 fps
FPS = SR // HOP
N_FFTS = (512, 1024, 2048)
N_MELS = 80
FMIN, FMAX = 27.5, 8000.0


def load_audio(path: str) -> np.ndarray:
    """Decode any soundfile-supported audio to mono float32 @ 16 kHz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def _mel_filterbank(n_fft: int, n_mels: int = N_MELS, fmin: float = FMIN,
                    fmax: float = FMAX, sr: int = SR) -> torch.Tensor:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, ce, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        up = (fft_freqs - lo) / max(ce - lo, 1e-9)
        down = (hi - fft_freqs) / max(hi - ce, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, down))
    # slaney-style area normalisation
    enorm = 2.0 / (mel_pts[2:] - mel_pts[:-2])
    fb *= enorm[:, None].astype(np.float32)
    return torch.from_numpy(fb)


_FBS: dict[int, torch.Tensor] = {}
_WINDOWS: dict[int, torch.Tensor] = {}


def features(x: np.ndarray) -> np.ndarray:
    """x: mono float32 @16k -> (T, 3, 80) float16 log-mel."""
    xt = torch.from_numpy(x)
    n_frames = 1 + len(x) // HOP
    outs = []
    for n_fft in N_FFTS:
        if n_fft not in _FBS:
            _FBS[n_fft] = _mel_filterbank(n_fft)
            _WINDOWS[n_fft] = torch.hann_window(n_fft)
        spec = torch.stft(xt, n_fft=n_fft, hop_length=HOP, window=_WINDOWS[n_fft],
                          center=True, return_complex=True, pad_mode="constant")
        mag = spec.abs() ** 2                      # (bins, T)
        mel = _FBS[n_fft] @ mag                    # (80, T)
        logmel = torch.log1p(mel).T                # (T, 80)
        outs.append(logmel[:n_frames])
    t = min(o.shape[0] for o in outs)
    feat = torch.stack([o[:t] for o in outs], dim=1)   # (T, 3, 80)
    return feat.to(torch.float16).numpy()
