"""Log-mel spectrograms in plain numpy (no librosa dependency).

STFT with a Hann window -> power -> triangular mel filterbank (Slaney-style,
HTK mel scale) -> log10 -> resample the time axis to a fixed number of frames
-> per-clip standardisation.  Output: (n_mels, n_frames) float32.
"""
from __future__ import annotations

import numpy as np


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def mel_filterbank(fs: float, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    fmax = min(fmax, fs / 2.0)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / fs)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-9)
        dn = (hi - bins) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(up, dn), 0.0, None)
        fb[i] /= max(fb[i].sum(), 1e-9)
    return fb


def stft_power(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2          # (frames, bins)


def log_mel(x: np.ndarray, fs: float, n_mels: int = 64, n_frames: int = 96, n_fft: int = 512,
            fmin: float = 50.0, fmax: float = 7500.0, hop: int | None = None) -> np.ndarray:
    hop = hop or max(1, len(x) // (n_frames + 1))
    P = stft_power(x, n_fft, hop)
    fb = mel_filterbank(fs, n_fft, n_mels, fmin, fmax)
    M = np.log10(P @ fb.T + 1e-10).T                          # (mels, frames)
    # Resample the time axis to exactly n_frames.
    t_src = np.linspace(0, 1, M.shape[1])
    t_dst = np.linspace(0, 1, n_frames)
    M = np.stack([np.interp(t_dst, t_src, M[i]) for i in range(n_mels)])
    M = (M - M.mean()) / (M.std() + 1e-6)
    return M.astype(np.float32)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    n_out = int(round(len(x) * fs_out / fs_in))
    t_in = np.arange(len(x)) / fs_in
    t_out = np.arange(n_out) / fs_out
    return np.interp(t_out, t_in, np.asarray(x, dtype=np.float64))


def to_uint8(M: np.ndarray) -> np.ndarray:
    """Quantise a standardised log-mel to uint8 for the small on-disk cache."""
    return np.clip((np.clip(M, -4, 4) + 4) / 8 * 255, 0, 255).astype(np.uint8)


def from_uint8(U: np.ndarray) -> np.ndarray:
    return (U.astype(np.float32) / 255 * 8 - 4).astype(np.float32)
