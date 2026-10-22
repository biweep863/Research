"""Sample alignment of a recording to the played signal, from the two chirps.

Each take starts and ends with the same chirp (``noises.chirp``). A matched
filter (cross-correlation with the chirp, resampled to the recording rate)
finds both in the recording; the first gives the offset, the distance between
them gives the clock-rate ratio of the two USB devices (playback vs capture).
A playback sample index ``n`` then maps to the recording sample
``offset + n * ratio * fs_rec / fs_play``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

from .noises import chirp


@dataclass
class Alignment:
    offset: float  # recording sample where the first chirp starts
    ratio: float  # (recorded chirp distance) / (nominal chirp distance)
    fs_rec: int
    fs_play: int
    peak1: float  # normalised matched-filter peaks (quality)
    peak2: float

    def to_rec(self, n_play) -> np.ndarray:
        """Playback sample(s), measured from the first chirp, -> recording sample(s)."""
        return self.offset + np.asarray(n_play) * self.ratio * self.fs_rec / self.fs_play


def _matched(x: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    c = scipy.signal.fftconvolve(x, tmpl[::-1], mode="valid")
    e = np.sqrt(np.maximum(scipy.signal.fftconvolve(x**2, np.ones(len(tmpl)), mode="valid"), 0)) * np.linalg.norm(tmpl)
    return np.abs(c) / (e + 1e-12)


def _peak(score: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    lo, hi = max(lo, 0), min(hi, len(score))
    k = lo + int(np.argmax(score[lo:hi]))
    if 0 < k < len(score) - 1:  # parabolic sub-sample refinement
        a, b, c = score[k - 1], score[k], score[k + 1]
        d = a - 2 * b + c
        return k + (0.5 * (a - c) / d if d < 0 else 0.0), float(b)
    return float(k), float(score[k])


def align(x: np.ndarray, fs_rec: int, timeline: dict, search_s: float = 1.5) -> Alignment:
    """x: (T,) or (C, T) recording (channels are summed after high-pass)."""
    x = np.atleast_2d(x).astype(np.float64)
    b, a = scipy.signal.butter(4, 400, "high", fs=fs_rec)
    y = scipy.signal.lfilter(b, a, x, axis=1)
    y = y.sum(0)
    fs_play = timeline["fs"]
    tmpl = chirp(fs_rec)
    score = _matched(y, tmpl)
    # the first chirp is within the first ``lead + search`` seconds of the recording
    k1, p1 = _peak(score, 0, int((timeline["chirp1"] / fs_play + search_s + 0.5) * fs_rec))
    d_nom = (timeline["chirp2"] - timeline["chirp1"]) * fs_rec / fs_play
    w = int(0.05 * fs_rec)  # +-50 ms covers any USB clock mismatch over a take
    k2, p2 = _peak(score, int(k1 + d_nom - w), int(k1 + d_nom + w))
    return Alignment(offset=k1, ratio=(k2 - k1) / d_nom, fs_rec=fs_rec, fs_play=fs_play, peak1=p1, peak2=p2)


def segment(al: Alignment, timeline: dict, start: str, end: str, pad_s=(0.0, 0.0)) -> slice:
    """Recording slice between two timeline marks (playback samples)."""
    c1 = timeline["chirp1"]
    a = al.to_rec(timeline[start] - c1) + pad_s[0] * al.fs_rec
    b = al.to_rec(timeline[end] - c1) + pad_s[1] * al.fs_rec
    return slice(int(round(a)), int(round(b)))
