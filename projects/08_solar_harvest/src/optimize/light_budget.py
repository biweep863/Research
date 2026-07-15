"""Ecosystem-aware light-budget optimisation (the paper's headline).

Setup. For a set of cells (latitude bands / biomes) ``i`` and hours ``h`` of a
representative day we are given:

* ``L[i, h]`` -- surface light available (W/m^2), from the insolation model;
* ``w[i, h]`` -- the *value* of harvesting light in that hour (e.g. a grid-
  demand weight -- energy is worth more at peak-demand hours);
* ``d_min[i]`` -- the daily light integral the ecosystem must keep to stay
  intact, expressed as a fraction ``f_min`` of the natural daily total.

Decision. A harvest fraction ``a[i, h] in [0, 1]``; the ecosystem receives the
reserved light ``(1 - a[i, h]) * L[i, h]``. Maximise total harvested *value*
subject to each cell reserving at least ``d_min[i]`` over the day. The
ecosystem is flexible about *which* hours it gets its light -- that temporal
slack is exactly what lets us harvest more.

Because each cell is independent and reserving a unit of light in hour ``h``
costs ``w[i, h]`` of foregone harvest value, the optimum is greedy
water-filling: reserve the ecosystem's quota from the lowest-value hours
first. This is provably optimal for the LP and is checked against
``scipy.optimize.linprog`` in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarvestResult:
    harvest_value: float      # sum_ih w * a * L
    harvest_energy: float     # sum_ih a * L (unweighted)
    reserved_energy: float    # sum_ih (1 - a) * L
    accept: np.ndarray        # a[i, h]


def _as2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def optimal_harvest(L, w, d_min) -> HarvestResult:
    """Greedy water-filling optimum: reserve each cell's quota from its
    lowest-value hours."""
    L = _as2d(L)
    w = _as2d(w)
    d_min = np.atleast_1d(np.asarray(d_min, dtype=float))
    n_cell, n_hour = L.shape
    reserve = np.zeros_like(L)
    for i in range(n_cell):
        remaining = float(d_min[i])
        for h in np.argsort(w[i]):          # cheapest-value hours first
            if remaining <= 0:
                break
            take = min(L[i, h], remaining)
            reserve[i, h] = take
            remaining -= take
    accept = np.divide(L - reserve, L, out=np.ones_like(L), where=L > 0)
    return _summarise(L, w, accept)


def flat_harvest(L, w, f_min) -> HarvestResult:
    """Naive uniform policy: reserve the same fraction ``f_min`` every hour.

    Feasible (each cell reserves f_min of its daily total) but wasteful -- it
    holds back light during the high-value hours as well.
    """
    L = _as2d(L)
    w = _as2d(w)
    accept = np.full_like(L, 1.0 - float(f_min))
    return _summarise(L, w, accept)


def no_cap_harvest(L, w) -> HarvestResult:
    """Ecosystem-blind upper bound: harvest everything (a = 1)."""
    L = _as2d(L)
    w = _as2d(w)
    return _summarise(L, w, np.ones_like(L))


def _summarise(L, w, accept) -> HarvestResult:
    reserve = (1.0 - accept) * L
    return HarvestResult(
        harvest_value=float(np.sum(w * accept * L)),
        harvest_energy=float(np.sum(accept * L)),
        reserved_energy=float(np.sum(reserve)),
        accept=accept,
    )


def daily_totals(L) -> np.ndarray:
    """Natural daily light integral per cell (sum over hours)."""
    return _as2d(L).sum(axis=1)


def min_light_from_fraction(L, f_min) -> np.ndarray:
    """Per-cell reserve requirement d_min = f_min * (daily total)."""
    return float(f_min) * daily_totals(L)


def lp_optimum_value(L_row, w_row, d_min_scalar) -> float:
    """Exact single-cell LP optimum (harvest value), for validating the greedy
    solver. Requires SciPy."""
    from scipy.optimize import linprog

    L = np.asarray(L_row, dtype=float)
    w = np.asarray(w_row, dtype=float)
    n = L.size
    # maximise sum w*a*L  ==  minimise  -(w*L) . a
    c = -(w * L)
    # reserve constraint: sum (1 - a) L >= d_min  ->  sum(L a) <= sum L - d_min
    A_ub = L[None, :]
    b_ub = np.array([L.sum() - d_min_scalar])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0.0, 1.0)] * n, method="highs")
    return float(-res.fun)
