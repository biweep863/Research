"""Contact-onset estimation: *when* did the boot hit something?

Two estimators, both pure functions of the window:

* ``onset_threshold`` -- the first sample whose baseline-removed |a| exceeds
  ``k`` times the background noise, walking back from the peak; refined by
  linear interpolation between the two samples straddling the threshold.
* ``onset_matched``   -- a matched filter: the rising edge of a clean pulse
  template is correlated with |a| near the peak and the best lag is refined by
  a parabolic fit, giving sub-sample resolution (Turin 1960).

Timing error is what limits *who touched first*: if two onsets are estimated
with independent errors of std sigma each, the probability of getting their
order right at true separation Delta is  Phi(Delta / (sigma * sqrt 2)).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _mag(w: np.ndarray) -> np.ndarray:
    return np.linalg.norm(w[:, :3].astype(np.float64), axis=1)


def _anchor(w: np.ndarray) -> int:
    """Index of the contact pulse.  Foot strikes are mostly vertical (sole
    normal, z) while ball and player contacts are mostly horizontal, so the
    peak of the *horizontal* excess picks the contact rather than the loudest
    stride, even when a soft player contact is quieter than a foot strike."""
    h = np.linalg.norm(w[:, :2].astype(np.float64), axis=1)
    n_bg = max(8, len(h) // 4)
    return int(np.argmax(h - np.median(h[:n_bg])))


def _refine(ex: np.ndarray, i: int, thr: float, fs: float) -> float:
    """Linear interpolation of the threshold crossing between i-1 and i."""
    if i <= 0:
        return 0.0
    y0, y1 = ex[i - 1], ex[i]
    if y1 == y0:
        return i / fs
    frac = (thr - y0) / (y1 - y0)
    return (i - 1 + np.clip(frac, 0.0, 1.0)) / fs


def onset_threshold(w: np.ndarray, fs: float, k: float = 6.0, bg_frac: float = 0.25) -> float:
    mag = _mag(w)
    n_bg = max(8, int(bg_frac * len(mag)))
    base = np.median(mag[:n_bg])
    sigma = 1.4826 * np.median(np.abs(mag[:n_bg] - base)) + 1e-6
    ex = mag - base
    peak = _anchor(w)
    thr = max(k * sigma, 0.08 * ex[peak])
    i = peak
    while i > 0 and ex[i - 1] > thr:
        i -= 1
    return _refine(ex, i, thr, fs)


def onset_matched(w: np.ndarray, fs: float, template: np.ndarray, bg_frac: float = 0.25) -> float:
    """Matched-filter onset: slide the *rising edge* of a clean pulse template
    (unit peak, onset at index 0) over the baseline-removed |a| and take the lag
    of maximum normalised correlation near the peak, refined by a parabolic fit
    across the three best lags (sub-sample resolution)."""
    mag = _mag(w)
    n_bg = max(8, int(bg_frac * len(mag)))
    base = np.median(mag[:n_bg])
    ex = mag - base
    peak = _anchor(w)
    edge = template[: int(np.argmax(template)) + 1]      # onset .. peak
    L = len(edge)
    if L < 2:
        return peak / fs
    lo = max(0, peak - 3 * L)
    hi = min(len(ex) - L, peak + L)
    if hi <= lo:
        return peak / fs
    lags = np.arange(lo, hi + 1)
    en = np.sqrt(np.sum(edge ** 2))
    score = np.array([np.dot(ex[k:k + L], edge) / (en * np.sqrt(np.sum(ex[k:k + L] ** 2)) + 1e-12)
                      for k in lags])
    b = int(np.argmax(score))
    k = lags[b]
    if 0 < b < len(score) - 1:
        y0, y1, y2 = score[b - 1], score[b], score[b + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            k = k + 0.5 * (y0 - y2) / den
    return float(k) / fs


def order_probability(gap_ms: np.ndarray | float, sigma_ms: float) -> np.ndarray:
    """Theory: P(correct order) for two onsets with independent N(0, sigma)
    errors separated by ``gap_ms``."""
    return norm.cdf(np.abs(gap_ms) / (sigma_ms * np.sqrt(2.0)))
