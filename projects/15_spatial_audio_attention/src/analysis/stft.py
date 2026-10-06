"""Thin STFT wrappers shared by the streaming localizer and offline beamforming."""

from __future__ import annotations

import numpy as np
import scipy.signal


def rfft_frame(frame: np.ndarray, window: np.ndarray) -> np.ndarray:
    """(M, nfft) time frame -> (M, F) spectrum."""
    return np.fft.rfft(frame * window[None, :], axis=-1)


def stft(x: np.ndarray, fs: int, nfft: int, hop: int):
    """(M, T) -> freqs (F,), X (M, F, N). Hann window, COLA-compatible."""
    freqs, _, X = scipy.signal.stft(
        x, fs=fs, window="hann", nperseg=nfft, noverlap=nfft - hop, axis=-1
    )
    return freqs, X


def istft(Y: np.ndarray, fs: int, nfft: int, hop: int, length: int | None = None):
    """(F, N) -> (T,). Inverse of :func:`stft`."""
    _, y = scipy.signal.istft(
        Y, fs=fs, window="hann", nperseg=nfft, noverlap=nfft - hop
    )
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y


def ema_alpha(tau_s: float, hop: int, fs: int) -> float:
    """Per-frame EMA coefficient for a time constant ``tau_s``."""
    return float(1.0 - np.exp(-hop / (fs * max(tau_s, 1e-6))))
