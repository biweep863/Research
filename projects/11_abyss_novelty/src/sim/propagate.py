"""Receiver-side signal: source level, transmission loss, ambient noise.

The call is scaled to its source level (dB re 1 uPa @ 1 m), attenuated by the
transmission loss at range r (src/physics/acoustics), and added to coloured
ambient noise shaped by the Wenz spectrum at the given sea state and shipping
level.  Everything is in dB re 1 uPa so that SNR at the receiver is physical.
"""
from __future__ import annotations

import numpy as np

from ..physics.acoustics import transmission_loss_db, wenz_noise_db


def wenz_noise(n: int, fs: float, sea_state: float, shipping: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with the Wenz spectrum level (uPa), via FFT shaping."""
    white = rng.normal(size=n)
    F = np.fft.rfft(white)
    fr = np.fft.rfftfreq(n, 1 / fs)
    lvl = wenz_noise_db(np.maximum(fr, 1.0), sea_state, shipping)     # dB re 1 uPa^2/Hz
    amp = 10 ** (lvl / 20.0) * np.sqrt(fs / 2.0 / n) * np.sqrt(n)       # per-bin amplitude
    return np.fft.irfft(F * amp / np.sqrt(n / 2.0), n=n)


def received(call: np.ndarray, sl_db: float, f_hz: float, range_m: float, fs: float,
             sea_state: float, shipping: float, rng: np.random.Generator,
             spreading: str = "hybrid") -> tuple[np.ndarray, float]:
    """Return (pressure at the receiver in uPa, SNR in dB in the call band)."""
    tl = float(transmission_loss_db(range_m, f_hz, spreading))
    rl = sl_db - tl
    sig = call * 10 ** (rl / 20.0)                                       # call has unit RMS
    noise = wenz_noise(len(call), fs, sea_state, shipping, rng)
    snr = 20 * np.log10(np.sqrt(np.mean(sig[call != 0] ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-9)) if np.any(call != 0) else -99.0
    return sig + noise, float(snr)
