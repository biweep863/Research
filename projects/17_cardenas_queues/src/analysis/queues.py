"""Steady-state queueing formulas (pure functions, rates in veh/h, times in h).

M/M/1, M/D/1, M/M/c (Erlang C), Kingman's G/G/1 approximation and a tandem
(Jackson-style) corridor in which every node receives the flow that survives
upstream exits plus its own external arrivals.
"""
from __future__ import annotations

import math

import numpy as np


def mm1(lam: float, mu: float) -> dict:
    """M/M/1. Returns rho, L (in system), Lq, W, Wq; inf when rho >= 1."""
    rho = lam / mu
    if rho >= 1:
        return {"rho": rho, "L": math.inf, "Lq": math.inf, "W": math.inf, "Wq": math.inf}
    L = rho / (1 - rho)
    Lq = rho ** 2 / (1 - rho)
    return {"rho": rho, "L": L, "Lq": Lq, "W": L / lam if lam > 0 else 1 / mu, "Wq": Lq / lam if lam > 0 else 0.0}


def md1(lam: float, mu: float) -> dict:
    """M/D/1 via Pollaczek-Khinchine with zero service variance."""
    rho = lam / mu
    if rho >= 1:
        return {"rho": rho, "Lq": math.inf, "Wq": math.inf}
    Lq = rho ** 2 / (2 * (1 - rho))
    return {"rho": rho, "Lq": Lq, "Wq": Lq / lam if lam > 0 else 0.0}


def erlang_c(c: int, a: float) -> float:
    """P(wait) in M/M/c with offered load a = lam/mu (Erlang). 1 if a >= c."""
    if a >= c:
        return 1.0
    # Stable recursion through Erlang B: B(0)=1, B(k) = a B(k-1) / (k + a B(k-1)).
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    rho = a / c
    return b / (1 - rho + rho * b)


def mmc(lam: float, mu: float, c: int) -> dict:
    """M/M/c with c identical servers (lanes) of rate mu each."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return {"rho": rho, "Pw": 1.0, "Lq": math.inf, "Wq": math.inf}
    pw = erlang_c(c, a)
    Wq = pw / (c * mu - lam)
    return {"rho": rho, "Pw": pw, "Lq": lam * Wq, "Wq": Wq}


def kingman_wq(lam: float, mu: float, ca2: float = 1.0, cs2: float = 1.0) -> float:
    """Kingman's heavy-traffic G/G/1 waiting time: rho/(1-rho) (ca2+cs2)/2 / mu."""
    rho = lam / mu
    if rho >= 1:
        return math.inf
    return rho / (1 - rho) * (ca2 + cs2) / 2 / mu


def tandem_flows(ext_in, exit_frac) -> np.ndarray:
    """Flow arriving at each node of a tandem line.

    Node k receives (flow through node k-1) * (1 - exit_frac[k-1]) + ext_in[k].
    Steady state, so the flow through a stable node equals its arrivals.
    """
    ext_in = np.asarray(ext_in, float)
    exit_frac = np.asarray(exit_frac, float)
    lam = np.zeros_like(ext_in)
    carry = 0.0
    for k in range(len(ext_in)):
        lam[k] = carry + ext_in[k]
        carry = lam[k] * (1 - exit_frac[k])
    return lam


def tandem_metrics(ext_in, exit_frac, capacity, ca2: float = 1.0, cs2: float = 0.0) -> dict:
    """Per-node utilisation and Kingman delay for a tandem corridor.

    Unstable nodes (rho >= 1) pass on only their capacity downstream: this is
    the steady-state view of a bottleneck "metering" what lies after it.
    """
    ext_in = np.asarray(ext_in, float)
    exit_frac = np.asarray(exit_frac, float)
    capacity = np.asarray(capacity, float)
    n = len(ext_in)
    lam = np.zeros(n)
    served = np.zeros(n)
    carry = 0.0
    for k in range(n):
        lam[k] = carry + ext_in[k]
        served[k] = min(lam[k], capacity[k])
        carry = served[k] * (1 - exit_frac[k])
    rho = lam / capacity
    wq = np.array([kingman_wq(l, c, ca2, cs2) for l, c in zip(lam, capacity)])
    return {"lam": lam, "served": served, "rho": rho, "wq_h": wq,
            "bottleneck": int(np.argmax(rho)), "n_unstable": int((rho >= 1).sum())}
