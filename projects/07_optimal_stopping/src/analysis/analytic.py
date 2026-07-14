"""Closed-form and exact-DP solutions for the optimal-stopping family.

Three problems share one skeleton (observe a sequence, decide stop/continue):

1. Classic secretary (rank-only): closed-form success probability and the
   asymptotic 1/e ("37%") rule.
2. Parking problem (MacQueen & Miller, 1960): exact backward induction over a
   street of Bernoulli(p) free spots — this doubles as computable ground truth
   for the RL agents.
3. Secretary with an observation cost c: net-payoff-optimal cutoff.

Everything here is pure NumPy and deterministic (no sampling), so the unit
tests can pin the Monte-Carlo estimators to these references.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classic secretary problem (rank-only)
# --------------------------------------------------------------------------- #


def secretary_success_prob(r: int, n: int) -> float:
    """P(select the overall best) for the "reject first r, then take the next
    record" rule, observing only relative ranks.

    Closed form: P(r, n) = (r/n) * sum_{i=r+1}^{n} 1/(i-1) for r >= 1,
    and 1/n for r = 0.
    """
    if not (0 <= r < n):
        raise ValueError(f"require 0 <= r < n, got r={r}, n={n}")
    if r == 0:
        return 1.0 / n
    tail = np.sum(1.0 / np.arange(r, n))  # 1/(i-1) for i in r+1..n  ->  1/r..1/(n-1)
    return (r / n) * tail


def secretary_mask_success(take_mask: np.ndarray, n: int) -> float:
    """Exact P(win) for *any* per-position policy: at a running record in
    position i (1-indexed) take iff ``take_mask[i-1]`` is True.

    Records at distinct positions are independent with P(record at j)=1/j, so
        P(win) = sum_{i: take} (1/n) * prod_{j<i, take} (j-1)/j.
    This reduces to ``secretary_success_prob`` for a reject-first-r mask and
    lets us score a learned (possibly non-threshold) policy exactly.
    """
    mask = np.asarray(take_mask, dtype=bool)
    total = 0.0
    no_stop = 1.0  # prob no earlier record-in-mask has stopped us
    for i in range(1, n + 1):
        if mask[i - 1]:
            total += (1.0 / n) * no_stop
            no_stop *= (i - 1) / i
    return float(total)


def secretary_optimal_threshold(n: int) -> dict:
    """Brute-force the optimal cutoff r* and its success probability."""
    probs = np.array([secretary_success_prob(r, n) for r in range(n)])
    r_star = int(np.argmax(probs))
    return {
        "n": n,
        "r_star": r_star,
        "x_star": r_star / n,          # fraction skipped
        "p_star": float(probs[r_star]),
        "probs": probs,
    }


def asymptotic_success(x: np.ndarray | float) -> np.ndarray | float:
    """Large-n success probability as a function of the skip fraction x:
    P(x) = -x ln x, maximised at x = 1/e with value 1/e."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(x > 0, -x * np.log(x), 0.0)
    return out


ONE_OVER_E = float(1.0 / np.e)  # 0.3678794... — the "37%" constant


# --------------------------------------------------------------------------- #
# 2. Parking problem (exact backward induction)
# --------------------------------------------------------------------------- #


def symmetric_street(k: int) -> np.ndarray:
    """Distances-to-destination encountered while driving past it:
    [k, k-1, ..., 1, 0, 1, ..., k]. Length 2k+1."""
    approach = np.arange(k, -1, -1)      # k .. 0
    depart = np.arange(1, k + 1)         # 1 .. k
    return np.concatenate([approach, depart]).astype(float)


def parking_optimal(distances: np.ndarray, p: float, fail_cost: float) -> dict:
    """Exact optimal policy for the parking MDP.

    We drive past spots in the given order. Each spot is *free* with
    probability ``p`` (independently). At a free spot we either park (pay its
    distance and stop) or skip (cannot return). Running out of spots pays
    ``fail_cost``.

    Backward induction on the expected remaining cost V_t (before we learn
    whether spot t is free):

        V_t = p * min(c_t, V_{t+1}) + (1 - p) * V_{t+1},   V_{T} = fail_cost

    The optimal action at a *free* spot t is "park" iff c_t <= V_{t+1}, so the
    optimal policy is a per-step distance threshold theta_t = V_{t+1}.
    """
    c = np.asarray(distances, dtype=float)
    T = len(c)
    V = np.empty(T + 1)
    V[T] = fail_cost
    theta = np.empty(T)               # accept-if-free threshold at each step
    for t in range(T - 1, -1, -1):
        cont = V[t + 1]
        theta[t] = cont
        V[t] = p * min(c[t], cont) + (1 - p) * cont
    return {
        "V": V,
        "theta": theta,               # park at free spot t iff c[t] <= theta[t]
        "expected_cost": float(V[0]),
        "distances": c,
        "p": p,
        "fail_cost": fail_cost,
    }


def parking_fixed_threshold_cost(
    distances: np.ndarray, p: float, fail_cost: float, theta: float
) -> float:
    """Expected cost of a *stationary* distance-threshold policy: park at the
    first free spot whose distance <= theta. Evaluated exactly via the same
    recursion with the decision fixed by ``theta``."""
    c = np.asarray(distances, dtype=float)
    T = len(c)
    V = fail_cost
    for t in range(T - 1, -1, -1):
        take = c[t] <= theta
        park_val = c[t] if take else V
        V = p * park_val + (1 - p) * V
    return float(V)


def parking_policy_cost(
    distances: np.ndarray, p: float, fail_cost: float, accept_mask: np.ndarray
) -> float:
    """Exact expected cost of an arbitrary per-step accept policy:
    at free step t, park iff ``accept_mask[t]``. Same recursion as the optimum
    but with decisions fixed by the mask (used to score learned agents)."""
    c = np.asarray(distances, dtype=float)
    mask = np.asarray(accept_mask, dtype=bool)
    V = fail_cost
    for t in range(len(c) - 1, -1, -1):
        park_val = c[t] if mask[t] else V
        V = p * park_val + (1 - p) * V
    return float(V)


def parking_best_stationary_threshold(
    distances: np.ndarray, p: float, fail_cost: float
) -> dict:
    """Best single distance threshold (grid = the distinct distances present).
    Useful as the "one number" a human would actually use."""
    c = np.asarray(distances, dtype=float)
    grid = np.unique(c)
    costs = np.array(
        [parking_fixed_threshold_cost(c, p, fail_cost, th) for th in grid]
    )
    i = int(np.argmin(costs))
    return {
        "theta_star": float(grid[i]),
        "expected_cost": float(costs[i]),
        "grid": grid,
        "costs": costs,
    }


# --------------------------------------------------------------------------- #
# 3. Secretary with a per-observation cost
# --------------------------------------------------------------------------- #


def secretary_expected_observations(r: int, n: int) -> float:
    """Expected number of candidates inspected under the reject-first-r rule
    (counting the r skipped plus those examined until a record is taken or the
    list ends). Computed exactly by linearity over stopping positions."""
    if r == 0:
        # take the first candidate immediately
        return 1.0
    # Under threshold r, stopping happens at the first i>r that is a running
    # record; if none, we inspect all n. P(stop exactly at i) for i>r is
    # r/((i-1) i). We inspect i candidates in that case; if never, inspect n.
    idx = np.arange(r + 1, n + 1)
    p_stop = r / ((idx - 1) * idx)
    p_none = 1.0 - p_stop.sum()
    return float((idx * p_stop).sum() + n * p_none)


def secretary_cost_optimal_threshold(n: int, cost: float) -> dict:
    """Optimal cutoff when each inspected candidate costs ``cost`` (net payoff
    = P(win) - cost * E[#inspected]). The cutoff shrinks as cost grows."""
    payoff = np.array(
        [
            secretary_success_prob(r, n)
            - cost * secretary_expected_observations(r, n)
            for r in range(n)
        ]
    )
    r_star = int(np.argmax(payoff))
    return {
        "n": n,
        "cost": cost,
        "r_star": r_star,
        "x_star": r_star / n,
        "net_payoff": float(payoff[r_star]),
        "payoff": payoff,
    }
