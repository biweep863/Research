"""Electromagnetic penetration of seawater: the skin depth.

delta = 1 / sqrt(pi f mu0 sigma), sigma ~ 4 S/m.  Field amplitude falls as
exp(-z/delta), i.e. 8.686 dB per skin depth.  A link with margin M dB
therefore reaches z = delta * M / 8.686.  Radar (GHz) has a skin depth of
millimetres; ELF submarine communication (tens of Hz) reaches tens of metres.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


def skin_depth(f_hz, sigma: float = 4.0) -> np.ndarray:
    f = np.asarray(f_hz, dtype=float)
    return 1.0 / np.sqrt(np.pi * f * MU0 * sigma)


def reach_depth(f_hz, margin_db: float, sigma: float = 4.0) -> np.ndarray:
    return skin_depth(f_hz, sigma) * margin_db / 8.686
