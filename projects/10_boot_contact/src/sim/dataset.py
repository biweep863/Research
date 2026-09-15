"""Compose gait + contact event + sensor model into labelled IMU windows.

A window is a (T, 6) array [ax, ay, az, gx, gy, gz] at the sensor's sampling
rate, with acceleration in g and angular rate in deg/s.  ``make_windows`` is the
single entry point used by every experiment; the background can be the
synthetic gait or a real ankle-IMU recording (PAMAP2) resampled to the sensor
rate, with the synthetic event added on top.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import CLASSES, CLASS_ID, EventTruth, ball_sensor, inject
from .gait import synth_gait
from .imu import SensorSpec, decimate, lowpass, sense


@dataclass
class WindowSet:
    X: np.ndarray                 # (N, T, 6)
    y: np.ndarray                 # (N,) class ids into CLASSES
    truths: list[EventTruth]
    ball: list[dict]              # ball-sensor reports, one per window
    fs: float

    @property
    def labels(self) -> list[str]:
        return [CLASSES[i] for i in self.y]


def _background_synthetic(window_s, sim_rate, rng, gait_cfg):
    t, acc, gyr = synth_gait(window_s, sim_rate, rng,
                             cadence_hz=tuple(gait_cfg["cadence_hz"]),
                             strike_peak_g=tuple(gait_cfg["strike_peak_g"]),
                             strike_dur_ms=tuple(gait_cfg["strike_dur_ms"]))
    return t, acc, gyr


def _background_real(window_s, sim_rate, rng, real_segments):
    """Random crop of a real ankle recording, linearly upsampled to sim_rate.
    Real data are 100 Hz so contain nothing above 50 Hz; the injected event
    supplies the high-frequency content, which is why the real background is
    only used at its native sampling rate in the experiments."""
    seg = real_segments[rng.integers(len(real_segments))]
    fs_real = 100.0
    n_need = int(np.ceil(window_s * fs_real)) + 2
    start = rng.integers(0, len(seg) - n_need)
    crop = seg[start:start + n_need]
    t_real = np.arange(len(crop)) / fs_real
    n = int(round(window_s * sim_rate))
    t = np.arange(n) / sim_rate
    out = np.stack([np.interp(t, t_real, crop[:, j]) for j in range(6)], axis=1)
    return t, out[:, :3], out[:, 3:]


def make_windows(n_per_class: int, fs: float, sensor_cfg: dict, sim_cfg: dict,
                 rng: np.random.Generator, classes: list[str] | None = None,
                 background: str = "synthetic", real_segments=None,
                 noise_scale: float = 1.0, ball_cfg: dict | None = None,
                 gap_ms: float | None = None) -> WindowSet:
    classes = classes or CLASSES
    sim_rate = float(sim_cfg["sim_rate"])
    window_s = float(sim_cfg["window_s"])
    spec = SensorSpec.from_cfg(fs, sensor_cfg, noise_scale=noise_scale)
    X, y, truths, balls = [], [], [], []
    for label in classes:
        for _ in range(n_per_class):
            if background == "pamap2" and real_segments:
                t, acc, gyr = _background_real(window_s, sim_rate, rng, real_segments)
            else:
                t, acc, gyr = _background_synthetic(window_s, sim_rate, rng, sim_cfg["gait"])
            sig, truth = inject(t, label, rng, sim_cfg, gap_ms=gap_ms)
            a, w = sense(acc + sig[:, :3], gyr + sig[:, 3:], spec, sim_rate, rng)
            X.append(np.concatenate([a, w], axis=1))
            y.append(CLASS_ID[label])
            truths.append(truth)
            balls.append(ball_sensor(truth, window_s, ball_cfg, sim_cfg, rng) if ball_cfg else {})
    return WindowSet(np.asarray(X, dtype=np.float32), np.asarray(y), truths, balls, fs)


def clean_templates(fs: float, sim_cfg: dict, rng: np.random.Generator, n_avg: int = 40
                    ) -> dict[str, np.ndarray]:
    """Average, noise-free |a| templates of a ball and a player pulse at ``fs``,
    aligned to onset.  Used by the two-template separator."""
    sim_rate = float(sim_cfg["sim_rate"])
    L = 0.25
    n = int(round(L * sim_rate))
    t = np.arange(n) / sim_rate
    out = {}
    for label in ("ball", "player"):
        acc = np.zeros(int(round(L * fs)))
        for _ in range(n_avg):
            sig, _ = inject(t, label, rng, sim_cfg, t_center=0.02)
            mag = np.linalg.norm(sig[:, :3], axis=1)
            fc = 0.45 * fs / 2
            mag = decimate(lowpass(mag, fc, sim_rate), sim_rate, fs)
            acc += mag / n_avg
        acc = acc[int(round(0.02 * fs)):]           # align index 0 to the onset
        out[label] = acc / max(acc.max(), 1e-9)
    return out
