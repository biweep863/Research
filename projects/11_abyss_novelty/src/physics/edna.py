"""Environmental DNA as a sensor: a shed-decay-transport budget.

A source sheds S copies/h; DNA decays at rate k (1/h); a current u advects it
and eddy diffusion (D) spreads it.  The steady plume along the current axis in
1-D (Crank 1975) has concentration
    C(x) = S / (u * A_eff) * exp(-k x / u)          (x downstream, metres)
with A_eff the plume cross-section grown by diffusion, A_eff ~ pi * (2 D x/u).
Detection footprint: the x where C falls below the assay floor; sample age:
x / u.  Decay also gives the "memory" of a sample: 1/k hours.
"""
from __future__ import annotations

import numpy as np


def plume_concentration(x_m, shed_per_h: float, k_per_h: float, u_m_s: float, d_m2_s: float) -> np.ndarray:
    x = np.maximum(np.asarray(x_m, dtype=float), 1.0)
    shed_per_s = shed_per_h / 3600.0
    k_per_s = k_per_h / 3600.0
    sigma2 = 2 * d_m2_s * x / u_m_s                  # lateral variance, m^2
    a_eff = np.pi * sigma2                             # m^2
    c_m3 = shed_per_s / (u_m_s * a_eff) * np.exp(-k_per_s * x / u_m_s)
    return c_m3 / 1000.0                               # copies per litre


def footprint_m(shed_per_h, k_per_h, u_m_s, d_m2_s, floor_per_l) -> float:
    x = np.logspace(0, 6, 4000)
    ok = plume_concentration(x, shed_per_h, k_per_h, u_m_s, d_m2_s) >= floor_per_l
    return float(x[ok].max()) if ok.any() else 0.0


def memory_h(k_per_h: float) -> float:
    return 1.0 / k_per_h
