"""Contact events injected into the gait signal, and the ball's own sensor.

Three physical signatures, all in the boot frame, all generated at the
simulation rate:

* **ball**   -- a short, hard, forward-axis deceleration (foot-ball contact
  lasts ~8-12 ms, Nunome et al. 2006), followed by a high-frequency ringing
  from the stiff boot/ball interface.
* **player** -- a longer, softer, oblique pulse (soft tissue compresses for
  tens of ms) followed by a low-frequency "push" as the bodies keep moving
  into each other.
* **both**   -- a ball pulse and a player pulse separated by a signed gap
  (negative = player first).  This is the anomalous case the paper is about:
  the boot strikes the ball and the opponent at almost the same instant.

The **ball sensor** is a separate IMU inside the ball.  It reports a pulse only
if the ball was struck, at the true ball-contact time plus a clock-sync offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gait import half_sine

CLASSES = ["none", "ball", "player", "both"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class EventTruth:
    label: str
    t_ball: float | None = None        # true ball-contact onset, s (window time)
    t_player: float | None = None      # true player-contact onset, s
    params: dict = field(default_factory=dict)

    @property
    def gap_ms(self) -> float | None:
        if self.t_ball is None or self.t_player is None:
            return None
        return 1000.0 * (self.t_player - self.t_ball)

    @property
    def ball_first(self) -> bool | None:
        """True if the ball was touched before (or without) the player."""
        if self.label == "ball":
            return True
        if self.label == "player":
            return False
        if self.label == "both":
            return self.t_ball <= self.t_player
        return None


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def ball_pulse(t: np.ndarray, t0: float, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, dict]:
    peak = rng.uniform(*cfg["peak_g"])
    dur = rng.uniform(*cfg["dur_ms"]) / 1000.0
    ring_hz = rng.uniform(*cfg["ring_hz"])
    ring_tau = rng.uniform(*cfg["ring_decay_ms"]) / 1000.0
    # Mostly along -x (foot decelerates), small random tilt.
    d = _unit(np.array([-1.0, rng.normal(0, 0.15), rng.normal(0, 0.2)]))
    p = half_sine(t, t0, dur)
    u = t - (t0 + dur)
    ring = np.where(u >= 0, 0.18 * peak * np.exp(-u / ring_tau) * np.sin(2 * np.pi * ring_hz * u), 0.0)
    acc = np.outer(peak * p + ring, d)
    gyr = np.zeros_like(acc)
    gyr[:, 1] += -rng.uniform(400, 900) * p          # foot pitch is arrested
    gyr[:, 2] += rng.normal(0, 150) * p
    return np.concatenate([acc, gyr], axis=1), {"peak_g": peak, "dur_ms": 1000 * dur}


def player_pulse(t: np.ndarray, t0: float, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, dict]:
    peak = rng.uniform(*cfg["peak_g"])
    dur = rng.uniform(*cfg["dur_ms"]) / 1000.0
    push = rng.uniform(*cfg["push_g"])
    push_ms = rng.uniform(*cfg["push_ms"]) / 1000.0
    # Oblique: the shin/foot of the opponent is rarely square-on.
    d = _unit(np.array([-rng.uniform(0.3, 1.0), rng.normal(0, 0.6), rng.normal(0, 0.4)]))
    p = half_sine(t, t0, dur)
    u = t - t0
    tail = np.where(u >= 0, push * (u / push_ms) * np.exp(1 - u / push_ms), 0.0)
    acc = np.outer(peak * p + tail, d)
    gyr = np.zeros_like(acc)
    gyr[:, 1] += -rng.uniform(100, 400) * p
    gyr[:, 2] += rng.normal(0, 250) * p              # foot twisted sideways
    gyr[:, 0] += rng.normal(0, 150) * p
    return np.concatenate([acc, gyr], axis=1), {"peak_g": peak, "dur_ms": 1000 * dur}


def inject(t: np.ndarray, label: str, rng: np.random.Generator, sim_cfg: dict,
           t_center: float | None = None, gap_ms: float | None = None
           ) -> tuple[np.ndarray, EventTruth]:
    """Return (signal (T,6) to add to the gait, truth).  ``t_center`` is where the
    first pulse starts (default: a random point in the middle 40 % of the
    window).  ``gap_ms`` overrides the config gap for the ``both`` class."""
    T = t[-1]
    if t_center is None:
        t_center = rng.uniform(0.3 * T, 0.6 * T)
    sig = np.zeros((len(t), 6))
    if label == "none":
        return sig, EventTruth("none")
    if label == "ball":
        s, prm = ball_pulse(t, t_center, rng, sim_cfg["ball"])
        return sig + s, EventTruth("ball", t_ball=t_center, params=prm)
    if label == "player":
        s, prm = player_pulse(t, t_center, rng, sim_cfg["player"])
        return sig + s, EventTruth("player", t_player=t_center, params=prm)
    if label == "both":
        gap = rng.uniform(*sim_cfg["both"]["gap_ms"]) if gap_ms is None else gap_ms
        if gap >= 0:
            tb, tp = t_center, t_center + gap / 1000.0
        else:
            tp, tb = t_center, t_center - gap / 1000.0
        sb, pb = ball_pulse(t, tb, rng, sim_cfg["ball"])
        sp, pp = player_pulse(t, tp, rng, sim_cfg["player"])
        return sig + sb + sp, EventTruth("both", t_ball=tb, t_player=tp,
                                          params={"ball": pb, "player": pp})
    raise ValueError(label)


def ball_sensor(truth: EventTruth, window_s: float, ball_cfg: dict, sim_cfg: dict,
                rng: np.random.Generator) -> dict:
    """What the ball's own IMU reports for this window: a detected impact time
    (or None) with clock-sync error, and a peak magnitude.  Only a struck ball
    reports anything.  Returned in seconds of window time."""
    fs = ball_cfg["fs_hz"]
    if truth.t_ball is None:
        return {"detected": False, "t_ball_hat": None, "peak_g": None}
    offset = rng.normal(0.0, ball_cfg["sync_jitter_ms"] / 1000.0)
    # The ball's onset estimate is quantised to its own sample grid.
    t_hat = np.round((truth.t_ball + offset) * fs) / fs
    peak = rng.uniform(*sim_cfg["ball"]["peak_g"]) * rng.uniform(1.5, 3.0)  # ball is lighter
    return {"detected": True, "t_ball_hat": float(t_hat), "peak_g": float(peak)}
