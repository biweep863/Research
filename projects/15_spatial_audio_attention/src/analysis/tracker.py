"""Wrap-safe speaker tracking.

home2's ``respeaker.py`` smooths the firmware DOA with a linear moving
average, so 359 deg and 1 deg average to 180 deg. Everything here works on
the circle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_deg(a):
    """Map to [-180, 180)."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles, weights=None) -> float:
    a = np.deg2rad(np.asarray(angles, dtype=float))
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    return float(np.rad2deg(np.arctan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) % 360)


@dataclass
class TrackState:
    az_deg: float = 0.0
    rate_dps: float = 0.0
    range_m: float = float("nan")
    confidence: float = 0.0
    last_t: float = -np.inf
    active: bool = False


class SpeakerTracker:
    """Alpha-beta filter on azimuth (deg) + EMA on log-range.

    Measurements with confidence below ``min_confidence`` are ignored. With no
    update the confidence decays and the track is dropped after ``hold_s``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.05,
        range_alpha: float = 0.2,
        min_confidence: float = 0.15,
        hold_s: float = 3.0,
        max_rate_dps: float = 120.0,
    ):
        self.alpha, self.beta, self.range_alpha = alpha, beta, range_alpha
        self.min_confidence, self.hold_s = min_confidence, hold_s
        self.max_rate = max_rate_dps
        self.state = TrackState()

    def update(self, t: float, az_deg: float, range_m: float | None, conf: float):
        s = self.state
        if conf < self.min_confidence:
            return self.predict(t)
        if not s.active:
            s.az_deg, s.rate_dps, s.active = az_deg % 360, 0.0, True
            s.range_m = range_m if range_m is not None else float("nan")
        else:
            dt = max(t - s.last_t, 1e-3)
            pred = s.az_deg + s.rate_dps * dt
            resid = float(wrap_deg(az_deg - pred))
            a = self.alpha * (0.5 + 0.5 * conf)
            s.az_deg = (pred + a * resid) % 360
            s.rate_dps = float(
                np.clip(s.rate_dps + self.beta * resid / dt, -self.max_rate, self.max_rate)
            )
            if range_m is not None and np.isfinite(range_m):
                if np.isfinite(s.range_m):
                    lr = (1 - self.range_alpha) * np.log(s.range_m) + self.range_alpha * np.log(range_m)
                    s.range_m = float(np.exp(lr))
                else:
                    s.range_m = range_m
        s.confidence = max(conf, 0.7 * s.confidence + 0.3 * conf)
        s.last_t = t
        return s

    def predict(self, t: float):
        s = self.state
        if not s.active:
            return s
        age = t - s.last_t
        if age > self.hold_s:
            s.active = False
            s.confidence = 0.0
        else:
            s.confidence *= 0.97
            s.rate_dps *= 0.9  # a silent speaker is assumed to stop moving
        return s
