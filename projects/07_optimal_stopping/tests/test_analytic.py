"""Unit tests: closed forms and exact DP vs. brute force / Monte Carlo.

Fast (no models, tiny problems). `make test` must pass before experiments.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.analysis.analytic import (  # noqa: E402
    ONE_OVER_E,
    asymptotic_success,
    parking_optimal,
    secretary_expected_observations,
    secretary_optimal_threshold,
    secretary_success_prob,
    symmetric_street,
)
from src.envs.parking import ParkingEpisode  # noqa: E402
from src.envs.secretary import threshold_success_rate  # noqa: E402


def test_secretary_closed_form_matches_montecarlo():
    n = 6
    rng = np.random.default_rng(0)
    for r in range(n):
        analytic = secretary_success_prob(r, n)
        mc = threshold_success_rate(n, r, 60000, rng)
        assert abs(analytic - mc) < 0.01, (r, analytic, mc)


def test_asymptotic_optimum_is_one_over_e():
    xs = np.linspace(1e-3, 1 - 1e-3, 5000)
    ys = asymptotic_success(xs)
    x_star = xs[int(np.argmax(ys))]
    assert abs(x_star - ONE_OVER_E) < 1e-2
    assert abs(ys.max() - ONE_OVER_E) < 1e-3


def test_large_n_threshold_approaches_37pct():
    res = secretary_optimal_threshold(1000)
    assert abs(res["x_star"] - ONE_OVER_E) < 0.01
    assert abs(res["p_star"] - ONE_OVER_E) < 0.01


def test_parking_dp_matches_brute_force():
    # Enumerate all 2^T free/occupied realizations, weight by p, and evaluate
    # the DP-optimal per-step threshold policy; compare to V[0].
    dists = symmetric_street(3)          # length 7
    p, fail = 0.5, 10.0
    opt = parking_optimal(dists, p, fail)
    theta = opt["theta"]
    T = len(dists)
    exp_cost = 0.0
    for pattern in product([0, 1], repeat=T):   # 1 == free
        prob = np.prod([p if b else (1 - p) for b in pattern])
        cost = fail
        for t in range(T):
            if pattern[t] and dists[t] <= theta[t]:
                cost = dists[t]
                break
        exp_cost += prob * cost
    assert abs(exp_cost - opt["expected_cost"]) < 1e-9


def test_parking_optimal_beats_montecarlo_of_itself():
    dists = symmetric_street(5)
    p, fail = 0.4, 20.0
    opt = parking_optimal(dists, p, fail)
    theta = opt["theta"]
    rng = np.random.default_rng(1)
    costs = []
    for _ in range(40000):
        ep = ParkingEpisode(dists, p, fail, rng)
        cost = fail
        for t, dist in ep.free_stream():
            if dist <= theta[t]:
                cost = dist
                break
        costs.append(cost)
    assert abs(np.mean(costs) - opt["expected_cost"]) < 0.1


def test_expected_observations_bounds():
    n = 20
    for r in range(n):
        e = secretary_expected_observations(r, n)
        assert r <= e <= n + 1e-9 or r == 0
