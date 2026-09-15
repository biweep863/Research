"""Sensor model: turn a clean high-rate acceleration / angular-rate signal into
what a boot-mounted IMU actually reports.

The chain is bandwidth (first-order low-pass at a fraction of Nyquist), additive
bias, white noise, full-scale clipping, and decimation from the simulation rate
to the sensor's sampling rate.  Every step is a config knob so that "what does a
+-16 g phone-grade part miss that a 200 g impact-rated part sees?" is a config
swap, not a code change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.80665


@dataclass
class SensorSpec:
    fs: float                     # sampling rate, Hz
    range_g: float = 200.0        # full scale, g
    acc_noise_g: float = 0.08     # white-noise std per sample, referenced to 1 kHz
    gyr_noise_dps: float = 0.5    # gyro white-noise std per sample, deg/s
    bias_g: float = 0.02          # constant accelerometer bias std, g
    bandwidth_frac: float = 0.45  # analogue low-pass cutoff / Nyquist
    noise_scale: float = 1.0      # multiplies both noise terms (SNR sweeps)

    @classmethod
    def from_cfg(cls, fs: float, d: dict, noise_scale: float = 1.0) -> "SensorSpec":
        return cls(fs=fs, range_g=d.get("range_g", 200.0),
                   acc_noise_g=d.get("acc_noise_g", 0.08),
                   gyr_noise_dps=d.get("gyr_noise_dps", 0.5),
                   bias_g=d.get("bias_g", 0.02),
                   bandwidth_frac=d.get("bandwidth_frac", 0.45),
                   noise_scale=noise_scale)


def lowpass(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """First-order IIR low-pass along axis 0 (vectorised over trailing axes)."""
    if fc <= 0 or fc >= fs / 2:
        return x.copy()
    from scipy.signal import lfilter
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * fc)
    a = dt / (rc + dt)
    # y[i] = y[i-1] + a (x[i] - y[i-1])  ==  IIR with b=[a], a=[1, -(1-a)],
    # initialised at x[0] so there is no start-up transient.
    x = np.asarray(x, dtype=np.float64)
    zi = (1 - a) * x[0]
    y, _ = lfilter([a], [1.0, -(1 - a)], x, axis=0, zi=zi[None, ...] if x.ndim > 1 else [zi])
    return y


def decimate(x: np.ndarray, sim_rate: float, fs: float) -> np.ndarray:
    """Pick every k-th sample (k = sim_rate / fs, must be an integer)."""
    k = int(round(sim_rate / fs))
    if abs(k * fs - sim_rate) > 1e-6:
        raise ValueError(f"sim_rate {sim_rate} must be an integer multiple of fs {fs}")
    return x[::k]


def sense(acc_g: np.ndarray, gyr_dps: np.ndarray, spec: SensorSpec, sim_rate: float,
          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Apply the sensor chain. Inputs are (T, 3) at ``sim_rate``; outputs are
    (T', 3) at ``spec.fs``. Acceleration in g, angular rate in deg/s."""
    fc = spec.bandwidth_frac * spec.fs / 2.0
    a = lowpass(acc_g, fc, sim_rate)
    w = lowpass(gyr_dps, fc, sim_rate)
    a = decimate(a, sim_rate, spec.fs)
    w = decimate(w, sim_rate, spec.fs)
    # Noise density is specified per sample at 1 kHz; scale with sqrt(fs/1000)
    # so a wider bandwidth means more noise per sample, as in a real part.
    k = np.sqrt(spec.fs / 1000.0) * spec.noise_scale
    a = a + rng.normal(0.0, spec.bias_g, size=3) + rng.normal(0.0, spec.acc_noise_g * k, size=a.shape)
    w = w + rng.normal(0.0, spec.gyr_noise_dps * k, size=w.shape)
    a = np.clip(a, -spec.range_g, spec.range_g)
    return a, w
