"""Background foot motion: a running gait synthesised in the boot's sensor frame.

Axes (sensor frame, fixed to the boot):
    x = forward (toe direction), y = lateral (left), z = up (sole normal).

Per stride the foot swings forward (smooth acceleration), strikes the ground
(short vertical impulse), and rolls through stance (near-zero motion, gravity
only).  Gravity rotates with the foot's pitch so that the accelerometer sees the
usual 1 g that a sensor at rest reports.  Everything is generated at the
simulation rate so that a 10 ms impact is resolved with tens of samples.
"""
from __future__ import annotations

import numpy as np


def half_sine(t: np.ndarray, t0: float, dur: float) -> np.ndarray:
    """Unit half-sine pulse of duration ``dur`` starting at ``t0`` (0 elsewhere)."""
    u = (t - t0) / dur
    out = np.zeros_like(t)
    m = (u >= 0) & (u <= 1)
    out[m] = np.sin(np.pi * u[m])
    return out


def synth_gait(duration_s: float, sim_rate: float, rng: np.random.Generator,
               cadence_hz: tuple[float, float] = (2.4, 3.2),
               strike_peak_g: tuple[float, float] = (6.0, 14.0),
               strike_dur_ms: tuple[float, float] = (25.0, 40.0),
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, acc_g (T,3), gyr_dps (T,3)) for a running foot.

    ``acc_g`` is what an ideal accelerometer would report (specific force,
    including gravity) in the sensor frame, in g.  ``gyr_dps`` is the angular
    rate in deg/s (mostly pitch about the lateral axis).
    """
    n = int(round(duration_s * sim_rate))
    t = np.arange(n) / sim_rate
    cad = rng.uniform(*cadence_hz)
    period = 1.0 / cad
    phase0 = rng.uniform(0, period)

    acc = np.zeros((n, 3))
    gyr = np.zeros((n, 3))

    # Foot pitch over the cycle: dorsiflex during swing, flat during stance.
    ph = ((t + phase0) / period) % 1.0                # 0..1 within stride
    swing = ph < 0.6                                  # ~60% swing while running
    pitch = np.where(swing, 35.0 * np.sin(np.pi * ph / 0.6) ** 2, 0.0)  # deg
    pitch_rate = np.gradient(np.deg2rad(pitch), t)     # rad/s
    gyr[:, 1] = np.rad2deg(pitch_rate)

    # Smooth swing-phase acceleration (forward then braking), in g.
    amp = rng.uniform(1.5, 3.0)
    acc[:, 0] += np.where(swing, amp * np.sin(2 * np.pi * ph / 0.6), 0.0)
    acc[:, 2] += np.where(swing, 0.6 * amp * np.sin(np.pi * ph / 0.6), 0.0)
    acc[:, 1] += 0.15 * amp * np.sin(2 * np.pi * t * cad + rng.uniform(0, 6.28))

    # Foot strikes: one vertical impulse per stride at the swing->stance boundary.
    k0 = -1
    while True:
        t_strike = k0 * period - phase0 + 0.6 * period
        if t_strike > duration_s:
            break
        if t_strike + 0.05 >= 0:
            peak = rng.uniform(*strike_peak_g)
            dur = rng.uniform(*strike_dur_ms) / 1000.0
            p = half_sine(t, t_strike, dur)
            acc[:, 2] += peak * p
            acc[:, 0] -= 0.35 * peak * p                 # braking component
            acc[:, 1] += rng.normal(0, 0.15) * peak * p
            gyr[:, 1] += -rng.uniform(150, 350) * p      # foot slaps flat
            gyr[:, 0] += rng.normal(0, 40) * p
        k0 += 1

    # Gravity in the sensor frame (foot pitched by `pitch` about y).
    pr = np.deg2rad(pitch)
    acc[:, 0] += np.sin(pr)
    acc[:, 2] += np.cos(pr)
    return t, acc, gyr
