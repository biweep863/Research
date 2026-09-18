"""Unit tests: kinematics, trip generator, closed form, simulator validation,
stairs rule, regret bookkeeping.  Seconds, no data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

from src.model.kinematics import Kinematics                                   # noqa: E402
from src.model.trips import TripGenerator, od_distribution                    # noqa: E402
from src.model.stairs import Stairs                                           # noqa: E402
from src.analytic.closed_form import (curve, expected_elevator_time,          # noqa: E402
                                      expected_trip_time, regret_stats)
from src.sim.elevator import Building, Passenger, poisson_background          # noqa: E402
from src.sim.evaluate import simulate_resident                                # noqa: E402
from utils.stats import bootstrap_ci                                          # noqa: E402

KIN = Kinematics()


def test_kinematics_closed_forms():
    k = Kinematics(v=2.5, a=1.0, floor_h=3.3)
    # One floor (3.3 m) does not reach 2.5 m/s (needs 6.25 m): 2 sqrt(d/a).
    assert abs(k.run_trapezoid(1) - 2 * np.sqrt(3.3)) < 1e-9
    # Long run reaches rated speed: d/v + v/a.
    assert abs(k.run_trapezoid(10) - (33.0 / 2.5 + 2.5)) < 1e-9
    assert k.run(0) == 0.0 and k.run(3) > k.run(2) > k.run(1)
    assert k.trapezoid_error() < 1.0          # linearised form within a second
    assert KIN.trapezoid_error() < 1e-9        # at 1.6 m/s every floor reaches speed
    assert KIN.stop(2) == KIN.t_door + 2 * KIN.t_pass


def test_trip_generator_limits_and_chaining():
    rng = np.random.default_rng(0)
    g = TripGenerator(10, 5, alpha=1.0, rng=rng)
    kinds = [g.next() for _ in range(20)]
    assert all(k[2] in ("lobby_out", "lobby_back") for k in kinds)
    assert all(kinds[i][1] == kinds[i + 1][0] for i in range(19))   # chained
    g0 = TripGenerator(10, 5, alpha=0.0, rng=rng)
    assert all(g0.next()[2].startswith("intra") for _ in range(20))
    probs = [p for _, _, p in od_distribution(10, 5, 0.3)]
    assert abs(sum(probs) - 1.0) < 1e-12


def test_closed_form_recovers_intuitions():
    # No lobby trips, car waits where you left it: only ride distance matters -> middle.
    c = curve(21, 0.0, KIN, "last")
    assert c["h_star"] == 11
    # All lobby trips with lobby parking: the ground-adjacent floor wins.
    c = curve(21, 1.0, KIN, "lobby")
    assert c["h_star"] == 2
    # Random car position (heavy traffic limit) with no lobby trips -> middle-ish.
    c = curve(21, 0.0, KIN, "random")
    assert abs(c["h_star"] - 11) <= 1
    assert c["regret_worst_pct"] > c["regret_middle_pct"] >= 0


def test_simulator_matches_closed_form_per_trip():
    b = Building(20, 1, KIN, "lobby")
    for t, o, d in [(100, 10, 1), (400, 1, 10), (700, 10, 15), (1000, 15, 10)]:
        p = Passenger(1, t, o, d, resident=True); b.add_passenger(p); b.run(t + 250)
        assert abs((p.t_alight - t) - expected_elevator_time(o, d, KIN, "lobby", 20)) < 1e-6
    assert b.cars[0].pos == 1 and b.cars[0].dir == 0          # parked again


def test_simulator_mean_matches_closed_form_at_zero_traffic():
    rng = np.random.default_rng(1)
    r = simulate_resident(10, 20, 0.7, KIN, "lobby", 0.0, 1, 7200, 150, rng)
    e = expected_trip_time(10, 20, 0.7, KIN, "lobby")["mean"]
    assert abs(r["mean"] - e) / e < 0.06 and r["mean_stops"] == 0.0


def test_simulator_serves_everyone_and_traffic_adds_stops():
    rng = np.random.default_rng(2)
    b = Building(15, 2, KIN, "lobby", capacity=6)
    ps = poisson_background(3.0, 3600, 15, 0.6, rng)
    for p in ps:
        b.add_passenger(p)
    b.run(3600 + 1800)
    assert len(b.done) == len(ps)
    assert all(p.t_alight >= p.t_board >= p.t_arrive for p in ps)
    r0 = simulate_resident(8, 15, 0.6, KIN, "lobby", 0.0, 1, 5400, 120, np.random.default_rng(3))
    r2 = simulate_resident(8, 15, 0.6, KIN, "lobby", 3.0, 1, 5400, 120, np.random.default_rng(3))
    assert r2["mean_stops"] > r0["mean_stops"] and r2["mean"] > r0["mean"]


def test_stairs_rule_and_asymmetry():
    s = Stairs(t_up=15, t_down=10, max_climb=3)
    te_1 = expected_elevator_time(5, 4, KIN, "lobby", 20)
    assert s.take_stairs(5, 4, te_1)               # one floor down: walk
    assert not s.take_stairs(5, 15, expected_elevator_time(5, 15, KIN, "lobby", 20))
    assert s.time(4, 5) > s.time(5, 4)             # up costs more than down
    assert not np.isfinite(s.time(2, 10))          # beyond the climb limit
    r = expected_trip_time(10, 20, 0.7, KIN, "lobby", stairs=s)
    assert r["frac_walked"] > 0 and r["mean"] <= r["mean_elevator_only"]


def test_regret_and_ci():
    homes = list(range(2, 11)); means = np.array([9, 7, 5, 4, 5, 7, 9, 12, 15], dtype=float)
    r = regret_stats(homes, means, 10)
    assert r["h_star"] == 5 and r["regret_worst_s"] == 11 and r["regret_middle_pct"] >= 0
    m, lo, hi = bootstrap_ci(np.random.default_rng(0).normal(30, 5, 200))
    assert lo <= m <= hi
