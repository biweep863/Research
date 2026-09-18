"""The passive sonar equation with Thorp absorption and Wenz ambient noise.

    SL - TL(r, f) - (NL(f) - DI) >= DT
    TL = 20 log10 r + alpha(f) r / 1000       (spherical spreading + absorption)

alpha(f) in dB/km from Thorp (1967) with f in kHz:
    alpha = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2) + 2.75e-4 f^2 + 0.003
Ambient noise (dB re 1 uPa^2/Hz) follows the Wenz curves as parameterised in
Urick (1983): shipping (10-1000 Hz) and wind/sea-state (100 Hz-100 kHz)
components plus thermal noise above ~50 kHz.
"""
from __future__ import annotations

import numpy as np


def thorp_alpha_db_km(f_hz) -> np.ndarray:
    f = np.asarray(f_hz, dtype=float) / 1000.0
    f2 = f * f
    return 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003


def wenz_noise_db(f_hz, sea_state: float = 3.0, shipping: float = 0.5) -> np.ndarray:
    """Ambient noise spectrum level, dB re 1 uPa^2/Hz (Urick 1983 fits)."""
    f = np.asarray(f_hz, dtype=float)
    fk = f / 1000.0
    turb = 17 - 30 * np.log10(np.maximum(fk, 1e-4))
    ship = 40 + 20 * (shipping - 0.5) + 26 * np.log10(np.maximum(fk, 1e-4)) - 60 * np.log10(fk + 0.03)
    wind = 50 + 7.5 * np.sqrt(max(sea_state, 0.0)) + 20 * np.log10(np.maximum(fk, 1e-4)) - 40 * np.log10(fk + 0.4)
    therm = -15 + 20 * np.log10(np.maximum(fk, 1e-4))
    lin = lambda x: 10 ** (x / 10.0)
    return 10 * np.log10(lin(turb) + lin(ship) + lin(wind) + lin(therm))


def transmission_loss_db(r_m, f_hz, spreading: str = "spherical", transition_m: float = 4000.0) -> np.ndarray:
    """Spherical spreading, or hybrid: spherical out to the water depth
    ``transition_m`` and cylindrical beyond it (the sound is then trapped
    between surface and bottom / in the deep sound channel; Urick 1983)."""
    r = np.maximum(np.asarray(r_m, dtype=float), 1.0)
    if spreading == "hybrid":
        spread = np.where(r <= transition_m, 20 * np.log10(r),
                          20 * np.log10(transition_m) + 10 * np.log10(r / transition_m))
    else:
        spread = 20 * np.log10(r)
    return spread + thorp_alpha_db_km(f_hz) * r / 1000.0


def detection_range_m(sl_db: float, f_hz: float, sea_state: float = 3.0, shipping: float = 0.5,
                      di_db: float = 0.0, dt_db: float = 10.0, bandwidth_hz: float | None = None,
                      max_range_m: float = 5e6, spreading: str = "spherical",
                      transition_m: float = 4000.0) -> float:
    """Largest r with SL - TL - (NL - DI) >= DT.  Noise level integrates the
    spectrum level over ``bandwidth_hz`` (default: one third octave at f)."""
    bw = bandwidth_hz if bandwidth_hz else 0.23 * f_hz
    nl = wenz_noise_db(f_hz, sea_state, shipping) + 10 * np.log10(bw)
    excess = sl_db - (nl - di_db) - dt_db          # budget available for TL
    if excess <= 0:
        return 0.0
    r = np.logspace(0, np.log10(max_range_m), 6000)
    ok = transmission_loss_db(r, f_hz, spreading, transition_m) <= excess
    return float(r[ok].max()) if ok.any() else 0.0
