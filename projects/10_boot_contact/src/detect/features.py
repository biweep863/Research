"""Hand-crafted features of one IMU window.  Pure numpy, no I/O.

Each feature is chosen to separate the three physical signatures: a ball impact
is *short, hard, forward and rings*; a player contact is *longer, softer,
oblique and pushes*; the gait background is *periodic and vertical*.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "peak_mag", "peak_jerk", "width_half_ms", "width_tenth_ms", "hf_ratio",
    "spec_centroid_hz", "ring_zc", "dir_x", "dir_y", "dir_z", "gyr_peak",
    "gyr_z_peak", "kurtosis", "tail_mean", "rms_bg", "n_peaks_above_half",
]


def _mag(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[..., :3], axis=-1)


def window_features(w: np.ndarray, fs: float) -> np.ndarray:
    """``w`` is (T, 6). Returns a 1-D feature vector (len FEATURE_NAMES)."""
    a = w[:, :3].astype(np.float64)
    g = w[:, 3:].astype(np.float64)
    mag = np.linalg.norm(a, axis=1)
    T = len(mag)
    dt = 1.0 / fs

    # Peak and its location.
    k = int(np.argmax(mag))
    peak = mag[k]
    jerk = np.abs(np.diff(mag)) * fs
    peak_jerk = jerk.max() if len(jerk) else 0.0

    # Pulse widths at 50 % and 10 % of the (baseline-removed) peak.
    base = np.median(mag)
    ex = mag - base
    pk = max(ex[k], 1e-9)

    def width(frac):
        thr = frac * pk
        lo = k
        while lo > 0 and ex[lo - 1] > thr:
            lo -= 1
        hi = k
        while hi < T - 1 and ex[hi + 1] > thr:
            hi += 1
        return 1000.0 * (hi - lo + 1) * dt

    w_half, w_tenth = width(0.5), width(0.1)

    # Spectral shape around the peak.
    lo, hi = max(0, k - int(0.06 * fs)), min(T, k + int(0.06 * fs))
    seg = ex[lo:hi] - ex[lo:hi].mean()
    if len(seg) >= 8:
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        freqs = np.fft.rfftfreq(len(seg), dt)
        tot = spec.sum() + 1e-12
        hf_ratio = spec[freqs > 80].sum() / tot
        centroid = float((freqs * spec).sum() / tot)
    else:
        hf_ratio, centroid = 0.0, 0.0

    # Ringing: zero crossings of the detrended tail right after the peak.
    tl, th = k, min(T, k + int(0.03 * fs))
    tail = ex[tl:th] - ex[tl:th].mean()
    ring_zc = float(np.sum(np.diff(np.sign(tail)) != 0)) if len(tail) > 2 else 0.0

    # Direction of the peak acceleration in the boot frame.
    d = a[k] / (np.linalg.norm(a[k]) + 1e-9)

    gyr_mag = np.linalg.norm(g, axis=1)
    m = ex.mean()
    s = ex.std() + 1e-9
    kurt = float(np.mean(((ex - m) / s) ** 4))
    # Push tail: mean excess 40-120 ms after the peak (player contact pushes).
    pl, ph = min(T - 1, k + int(0.04 * fs)), min(T, k + int(0.12 * fs))
    tail_mean = float(ex[pl:ph].mean()) if ph > pl else 0.0
    rms_bg = float(np.sqrt(np.mean(np.delete(ex, slice(lo, hi)) ** 2))) if T > hi - lo else 0.0
    n_peaks = float(np.sum((ex[1:-1] > ex[:-2]) & (ex[1:-1] > ex[2:]) & (ex[1:-1] > 0.5 * pk)))

    return np.array([
        peak, peak_jerk, w_half, w_tenth, hf_ratio, centroid, ring_zc,
        d[0], d[1], d[2], gyr_mag.max(), np.abs(g[:, 2]).max(), kurt, tail_mean,
        rms_bg, n_peaks,
    ])


def featurize(X: np.ndarray, fs: float) -> np.ndarray:
    return np.stack([window_features(w, fs) for w in X])
