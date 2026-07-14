"""Model-free agents that learn stopping policies from experience only.

Two algorithm families, each applied to both problems:

* Tabular Q-learning over the discrete decision states (record position for
  secretary; street step for parking). Recovers the full per-state policy.
* REINFORCE over a one-parameter threshold policy (a learnable cutoff passed
  through a sigmoid). The single learned parameter *is* the interpretable
  threshold, so we can plot "learned threshold vs. analytic optimum" directly.

Neither agent is told the closed-form solution; both see only rewards.
"""
from __future__ import annotations

import numpy as np

from src.envs.secretary import SecretaryEpisode
from src.envs.parking import ParkingEpisode


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------- #
# Tabular Q-learning
# --------------------------------------------------------------------------- #

SKIP, TAKE = 0, 1


def q_learn_secretary(
    n: int, episodes: int, rng, alpha=0.1, eps=0.2,
) -> dict:
    """Q-learning over states = running-record position (1..n). Epsilon is
    annealed linearly to zero so the greedy policy sharpens late in training."""
    Q = np.zeros((n + 2, 2))
    visited = np.zeros(n + 2, dtype=bool)
    for k in range(episodes):
        frac = k / episodes
        eps_k = eps * (1.0 - frac)
        alpha_k = alpha * (1.0 - 0.9 * frac)          # decay to 10% of alpha
        ep = SecretaryEpisode(n, rng)
        stream = list(ep.record_stream())      # [(pos, is_best), ...]
        for j, (pos, is_best) in enumerate(stream):
            visited[pos] = True
            greedy = TAKE if Q[pos, TAKE] >= Q[pos, SKIP] else SKIP
            a = greedy if rng.random() > eps_k else rng.integers(2)
            if a == TAKE:
                r = 1.0 if is_best else 0.0
                Q[pos, TAKE] += alpha_k * (r - Q[pos, TAKE])
                break
            else:
                # skip -> next record, or terminal (missed) if none remain
                if j + 1 < len(stream):
                    nxt = stream[j + 1][0]
                    target = np.max(Q[nxt])
                else:
                    target = 0.0           # ran out: reward 0
                Q[pos, SKIP] += alpha_k * (target - Q[pos, SKIP])
    mask = np.array([Q[pos, TAKE] >= Q[pos, SKIP] for pos in range(1, n + 1)])
    thr = _secretary_threshold_from_Q(Q, n)
    return {"Q": Q, "take_mask": mask, "threshold_pos": thr, "x_threshold": thr / n}


def _secretary_threshold_from_Q(Q: np.ndarray, n: int) -> int:
    """Smallest position at which TAKE becomes the greedy action."""
    for pos in range(1, n + 1):
        if Q[pos, TAKE] >= Q[pos, SKIP] and Q[pos].any():
            return pos
    return n


def q_learn_parking(
    distances: np.ndarray, p: float, fail_cost: float, episodes: int, rng,
    alpha=0.1, eps=0.2,
) -> dict:
    """Q-learning over states = street step index at a free spot. Because the
    optimal policy is per-step (not a single distance threshold — near the end
    of the street you accept anything rather than fail), we return the full
    greedy accept mask and let the exact recursion score it."""
    c = np.asarray(distances, dtype=float)
    T = len(c)
    Q = np.zeros((T + 1, 2))
    for k in range(episodes):
        frac = k / episodes
        eps_k = eps * (1.0 - frac)
        alpha_k = alpha * (1.0 - 0.9 * frac)
        ep = ParkingEpisode(c, p, fail_cost, rng)
        stream = list(ep.free_stream())        # [(t, dist), ...]
        for j, (t, dist) in enumerate(stream):
            greedy = TAKE if Q[t, TAKE] >= Q[t, SKIP] else SKIP
            a = greedy if rng.random() > eps_k else rng.integers(2)
            if a == TAKE:
                r = -dist
                Q[t, TAKE] += alpha_k * (r - Q[t, TAKE])
                break
            else:
                if j + 1 < len(stream):
                    nxt = stream[j + 1][0]
                    target = np.max(Q[nxt])
                else:
                    target = -fail_cost
                Q[t, SKIP] += alpha_k * (target - Q[t, SKIP])
    accept_mask = np.array([Q[t, TAKE] >= Q[t, SKIP] for t in range(T)])
    theta = _parking_threshold_from_Q(Q, c)
    return {"Q": Q, "accept_mask": accept_mask, "theta_distance": theta}


def _parking_threshold_from_Q(Q: np.ndarray, c: np.ndarray) -> float:
    """Largest distance the greedy policy is willing to accept."""
    accept = [c[t] for t in range(len(c)) if Q[t].any() and Q[t, TAKE] >= Q[t, SKIP]]
    return float(max(accept)) if accept else float("nan")


# --------------------------------------------------------------------------- #
# REINFORCE over a one-parameter threshold policy
# --------------------------------------------------------------------------- #


def reinforce_secretary(
    n: int, episodes: int, rng, lr=0.05, beta=6.0, tau0=0.37,
) -> dict:
    """Learn skip-fraction tau. At a record in position pos the probability of
    taking is sigmoid(beta*(pos/n - tau)); tau updated by policy gradient with
    a moving-average baseline."""
    tau = tau0
    baseline = 0.0
    hist = []
    for k in range(episodes):
        ep = SecretaryEpisode(n, rng)
        logp_grad = 0.0
        reward = 0.0
        for pos, is_best in ep.record_stream():
            x = pos / n
            pi_take = _sigmoid(beta * (x - tau))
            take = rng.random() < pi_take
            # d/dtau log pi(a): for take -> -beta*(1-pi); for skip -> beta*pi
            if take:
                logp_grad += -beta * (1.0 - pi_take)
                reward = 1.0 if is_best else 0.0
                break
            else:
                logp_grad += beta * pi_take
        # (reward stays 0 if no record was ever taken)
        baseline += 0.02 * (reward - baseline)
        tau += lr * (reward - baseline) * logp_grad
        tau = float(np.clip(tau, 0.0, 1.0))
        if k % max(1, episodes // 50) == 0:
            hist.append(tau)
    return {"tau": tau, "x_threshold": tau, "tau_history": hist}


def reinforce_parking(
    distances: np.ndarray, p: float, fail_cost: float, episodes: int, rng,
    lr=0.02, beta=1.5, theta0=None,
) -> dict:
    """Learn distance threshold theta. At a free spot of distance d the
    probability of parking is sigmoid(beta*(theta - d))."""
    c = np.asarray(distances, dtype=float)
    theta = float(c.mean()) if theta0 is None else theta0
    baseline = -fail_cost
    hist = []
    for k in range(episodes):
        ep = ParkingEpisode(c, p, fail_cost, rng)
        logp_grad = 0.0
        cost = fail_cost
        for t, dist in ep.free_stream():
            pi_take = _sigmoid(beta * (theta - dist))
            take = rng.random() < pi_take
            if take:
                logp_grad += beta * (1.0 - pi_take)     # d/dtheta log pi(take)
                cost = dist
                break
            else:
                logp_grad += -beta * pi_take            # d/dtheta log pi(skip)
        reward = -cost
        baseline += 0.02 * (reward - baseline)
        theta += lr * (reward - baseline) * logp_grad
        if k % max(1, episodes // 50) == 0:
            hist.append(theta)
    return {"theta_distance": theta, "theta_history": hist}


# --------------------------------------------------------------------------- #
# Regret helpers
# --------------------------------------------------------------------------- #


def relative_regret(achieved: float, optimal: float) -> float:
    """|achieved - optimal| / |optimal|, guarded against divide-by-zero."""
    denom = abs(optimal) if abs(optimal) > 1e-12 else 1.0
    return abs(achieved - optimal) / denom
