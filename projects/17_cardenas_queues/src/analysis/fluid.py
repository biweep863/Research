"""Deterministic (fluid) queueing with cumulative curves (Newell 1982).

When demand exceeds capacity only for the peak, steady-state formulas give
infinity; the fluid model gives the finite queue that builds and clears.
"""
from __future__ import annotations

import numpy as np


def fluid_queue(arrival_vph, capacity_vph, dt_h: float) -> dict:
    """Point queue on a time grid.

    arrival_vph, capacity_vph: arrays (or scalars) of rates per step.
    Returns cumulative arrivals A, departures D, queue Q = A - D (veh), total
    delay (veh h) = area between the curves, max queue and clearance step.
    """
    lam = np.asarray(arrival_vph, float)
    n = lam.size
    mu = np.broadcast_to(np.asarray(capacity_vph, float), (n,))
    A = np.zeros(n + 1)
    D = np.zeros(n + 1)
    for t in range(n):
        A[t + 1] = A[t] + lam[t] * dt_h
        D[t + 1] = min(A[t + 1], D[t] + mu[t] * dt_h)
    Q = A - D
    pos = np.nonzero(Q > 1e-9)[0]
    return {"A": A, "D": D, "Q": Q,
            "total_delay_vh": float(np.trapezoid(Q, dx=dt_h)),
            "max_queue_veh": float(Q.max()),
            "clear_step": int(pos[-1] + 1) if pos.size else 0}


def rectangular_peak(lam_peak: float, lam_off: float, mu: float, T_h: float) -> dict:
    """Closed form for demand lam_peak > mu during [0, T], lam_off < mu after.

    Q_max = (lam_peak - mu) T; clears after Q_max / (mu - lam_off);
    total delay = 1/2 Q_max (T + clearance).
    """
    if lam_peak <= mu:
        return {"max_queue_veh": 0.0, "clear_h": 0.0, "total_delay_vh": 0.0}
    qmax = (lam_peak - mu) * T_h
    clear = qmax / (mu - lam_off)
    return {"max_queue_veh": qmax, "clear_h": clear, "total_delay_vh": 0.5 * qmax * (T_h + clear)}
