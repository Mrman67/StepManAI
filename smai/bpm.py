"""Constant-BPM tempo + beat-phase estimation from mel features (no librosa)."""
from __future__ import annotations

import numpy as np

FPS = 100


def onset_envelope(feat: np.ndarray) -> np.ndarray:
    """feat: (T, 3, 80) log-mel -> onset strength (T,) via spectral flux."""
    m = feat[:, 1, :].astype(np.float32)          # 1024-fft channel
    d = np.diff(m, axis=0, prepend=m[:1])
    env = np.maximum(d, 0).sum(axis=1)
    # local mean removal
    k = 100
    pad = np.pad(env, (k // 2, k // 2), mode="edge")
    loc = np.convolve(pad, np.ones(k) / k, mode="valid")[:len(env)]
    env = np.maximum(env - loc, 0)
    if env.max() > 0:
        env = env / env.max()
    return env


def detect_bpm(feat: np.ndarray, lo: float = 70.0, hi: float = 200.0):
    """Return (bpm, first_beat_time_seconds)."""
    env = onset_envelope(feat)
    n = len(env)
    # autocorrelation
    f = np.fft.rfft(env - env.mean(), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac[:int(FPS * 60 / (hi * 1.05))] = 0

    def score_lag(lag: float) -> float:
        # comb: sum autocorr at multiples of lag
        s, w = 0.0, 0.0
        for mult, wt in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
            idx = lag * mult
            i0 = int(idx)
            if i0 + 1 >= n:
                break
            frac = idx - i0
            s += wt * (ac[i0] * (1 - frac) + ac[i0 + 1] * frac)
            w += wt
        return s / max(w, 1e-9)

    # coarse sweep over bpm
    bpms = np.arange(lo, hi, 0.5)
    scores = np.array([score_lag(FPS * 60.0 / b) for b in bpms])
    # mild preference for typical dance tempos
    pref = np.exp(-0.5 * ((bpms - 140.0) / 60.0) ** 2)
    best_b = bpms[np.argmax(scores * (0.6 + 0.4 * pref))]
    # fine sweep
    fine = np.arange(best_b - 1.0, best_b + 1.0, 0.02)
    fs = [score_lag(FPS * 60.0 / b) for b in fine]
    bpm = float(fine[int(np.argmax(fs))])

    # integer snap if very close (most songs have integer bpm)
    if abs(bpm - round(bpm)) < 0.15:
        bpm = float(round(bpm))

    # phase: fold onset env at beat period, strongest bin = beat position
    period = FPS * 60.0 / bpm
    nbins = 64
    hist = np.zeros(nbins)
    for i, v in enumerate(env):
        hist[int((i % period) / period * nbins) % nbins] += v
    # smooth circularly
    hist = np.convolve(np.tile(hist, 3), np.ones(3) / 3, mode="same")[nbins:2 * nbins]
    phase_frames = (np.argmax(hist) + 0.5) / nbins * period
    # earliest beat time >= 0
    t0 = phase_frames / FPS
    while t0 - 60.0 / bpm >= 0:
        t0 -= 60.0 / bpm
    return bpm, float(t0)
