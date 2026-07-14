"""Parking-problem environment (MacQueen & Miller, 1960).

Driving past a row of spots ordered by the distances in ``distances`` (e.g.
[k,...,1,0,1,...,k] approaching a destination). Each spot is *free*
independently with probability ``p``. At a free spot we park (pay its distance,
episode ends) or skip (no return). Reaching the end unparked pays ``fail_cost``.

The optimal policy is a per-step distance threshold; see
``analysis.analytic.parking_optimal``. Here we provide simulators for a fixed
threshold policy and an episode interface for RL agents.
"""
from __future__ import annotations

import numpy as np


def threshold_expected_cost(
    distances: np.ndarray, p: float, fail_cost: float, theta: float,
    n_episodes: int, rng,
) -> float:
    """Monte-Carlo mean cost of the stationary "park first free spot with
    distance <= theta" policy."""
    c = np.asarray(distances, dtype=float)
    T = len(c)
    free = rng.random((n_episodes, T)) < p
    eligible = c <= theta                       # spots we would accept if free
    take = free & eligible[None, :]
    total = np.full(n_episodes, fail_cost)
    # first eligible free spot per episode
    has = take.any(axis=1)
    first = np.argmax(take, axis=1)             # 0 if none, guarded by `has`
    total[has] = c[first[has]]
    return float(total.mean())


class ParkingEpisode:
    """One realization of free/occupied spots, exposed as sequential decisions.

    The agent is queried only at *free* spots (the only actionable states).
    """

    def __init__(self, distances: np.ndarray, p: float, fail_cost: float, rng):
        self.c = np.asarray(distances, dtype=float)
        self.T = len(self.c)
        self.fail_cost = fail_cost
        self.free = rng.random(self.T) < p

    def free_stream(self):
        """Yield (step_index, distance) for each free spot, in order."""
        for t in range(self.T):
            if self.free[t]:
                yield t, float(self.c[t])
