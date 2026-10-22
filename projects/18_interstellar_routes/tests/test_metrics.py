"""Unit tests: every optimiser checked against a brute-force reference, the
physics against closed forms / astropy, and the kinematics end-to-end against
a published Gaia result (Gliese 710). No downloads; runs in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.analysis.graph import (  # noqa: E402
    euclidean_mst, minimax_bruteforce, minimax_from_source, range_graph, reach_curve,
)
from src.analysis.routing import astar, dijkstra_expansions, fuel_bottleneck, greedy_route  # noqa: E402
from src.analysis.temporal import (  # noqa: E402
    TemporalGraph, candidate_pairs, crossing_r, intercept_time, min_separation, open_windows,
)
from src.analysis.tours import mst_length, nn_tour, path_length, two_opt  # noqa: E402
from src.model.kinematics import (  # noqa: E402
    KMS_TO_PCMYR, astrometry_to_cartesian, closest_approach,
)
from src.model.propulsion import (  # noqa: E402
    brachistochrone_times_yr, hop_range_pc, rocket_cruise_speed_kms, sail_brake_speed_kms,
)


def _cloud(n, seed=0, scale=10.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 3))


# --- static graph ---------------------------------------------------------- #
def test_emst_matches_dense_mst():
    pos = _cloud(300)
    dense = minimum_spanning_tree(cdist(pos, pos)).sum()
    assert euclidean_mst(pos, k=8).sum() / 2 == pytest.approx(dense, rel=1e-12)


def test_minimax_on_mst_equals_bruteforce_widest_path():
    pos = _cloud(80, seed=1)
    bott, hops, plen = minimax_from_source(euclidean_mst(pos), 0)
    np.testing.assert_allclose(bott, minimax_bruteforce(pos, 0), atol=1e-12)
    assert hops[0] == 0 and np.all(plen >= np.linalg.norm(pos - pos[0], axis=1) - 1e-9)


def test_reach_curve_and_crossing():
    bott = np.array([0.0, 1.0, 2.0, 3.0])
    f = reach_curve(bott, np.ones(4, bool), np.array([0.5, 1.0, 2.5, 3.0]))
    np.testing.assert_allclose(f, [0.25, 0.5, 0.75, 1.0])
    assert crossing_r(np.array([1.0, 2.0]), np.array([0.25, 0.75]), 0.5) == pytest.approx(1.5)


# --- temporal graph -------------------------------------------------------- #
def _moving(n, seed=2):
    rng = np.random.default_rng(seed)
    return rng.uniform(-8, 8, size=(n, 3)), rng.normal(scale=6.0, size=(n, 3))


def test_candidate_pairs_complete():
    pos, vel = _moving(250)
    r, T = 1.0, 1.5
    got = {tuple(p) for p in candidate_pairs(pos, vel, r, T, v_cap=1e3).tolist()}
    i, j = np.triu_indices(len(pos), 1)
    dmin = min_separation(pos[j] - pos[i], vel[j] - vel[i], 0.0, T)
    want = {(a, b) for a, b, d in zip(i, j, dmin) if d <= r}
    assert got == want


def test_open_windows_match_sampling():
    pos, vel = _moving(60, seed=3)
    i, j = np.triu_indices(len(pos), 1)
    p, w = pos[j] - pos[i], vel[j] - vel[i]
    a, b = open_windows(p, w, 2.0, 2.0)
    ts = np.linspace(0, 2.0, 4001)
    d = np.linalg.norm(p[:, None, :] + w[:, None, :] * ts[None, :, None], axis=2)
    inside = d <= 2.0
    for k in np.nonzero(inside.any(1))[0][:200]:
        tk = ts[inside[k]]
        assert a[k] <= tk.min() + 1e-3 and b[k] >= tk.max() - 1e-3


def test_intercept_time_solves_equation():
    rng = np.random.default_rng(4)
    p, w = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    tau = intercept_time(p, w, 30.0)
    np.testing.assert_allclose(np.linalg.norm(p + w * tau[:, None], axis=1), 30.0 * tau)


def test_frozen_stars_reduce_to_static_bottleneck():
    pos = _cloud(400, seed=5)
    vel = np.zeros_like(pos)
    bott, _, _ = minimax_from_source(euclidean_mst(pos), 0)
    ub = np.unique(bott)
    k = len(ub) // 2
    r = float(0.5 * (ub[k] + ub[k + 1]))      # between two MST edges: no ties
    pairs = candidate_pairs(pos, vel, r, 1.0, v_cap=1.0)
    tg = TemporalGraph(pos, vel, pairs, r, v_ship=1e6, t_max=1.0)
    arr = tg.earliest_arrival(0)
    np.testing.assert_array_equal(np.isfinite(arr), bott <= r)


def test_earliest_arrival_matches_fixed_point():
    pos, vel = _moving(120, seed=6)
    r, T, v = 1.5, 2.0, 200.0
    pairs = candidate_pairs(pos, vel, r, T, v_cap=1e3)
    tg = TemporalGraph(pos, vel, pairs, r, v, T)
    arr = tg.earliest_arrival(0, settle=0.05)
    # reference: Bellman-Ford style relaxation over all edges until stable
    best = np.full(len(pos), np.inf)
    best[0] = 0.0
    for _ in range(len(pos)):
        changed = False
        for u in range(len(pos)):
            if not np.isfinite(best[u]):
                continue
            lo, hi = tg.indptr[u], tg.indptr[u + 1]
            for e in range(lo, hi):
                dep = max(best[u] + (0.05 if u else 0.0), tg.a[e])
                if dep > tg.b[e]:
                    continue
                p = tg.P[e] + tg.W[e] * dep
                t = dep + intercept_time(p[None], tg.W[e][None], v)[0]
                if t <= T and t < best[tg.dst[e]] - 1e-12:
                    best[tg.dst[e]] = t
                    changed = True
        if not changed:
            break
    np.testing.assert_allclose(arr, best)


# --- routing --------------------------------------------------------------- #
def test_astar_equals_dijkstra_and_expands_less():
    pos = _cloud(600, seed=7)
    g = range_graph(pos, 3.0)
    d_ref = dijkstra(g, indices=0)
    tgt = int(np.nanargmax(np.where(np.isfinite(d_ref), d_ref, np.nan)))
    d_a, n_a = astar(g, pos, 0, tgt)
    d_d, n_d = dijkstra_expansions(g, 0, tgt)
    assert d_a == pytest.approx(d_ref[tgt]) and d_d == pytest.approx(d_ref[tgt])
    assert n_a <= n_d


def test_greedy_never_beats_optimal():
    pos = _cloud(500, seed=8)
    g = range_graph(pos, 3.0)
    d_ref = dijkstra(g, indices=0)
    for t in range(1, 60):
        flown, ok = greedy_route(g, pos, 0, t)
        if ok:
            assert flown >= d_ref[t] - 1e-9


def test_fuel_bottleneck_all_stations_is_minimax():
    pos = _cloud(120, seed=9)
    g = csr_matrix(cdist(pos, pos))
    need, _ = fuel_bottleneck(g, np.ones(len(pos), bool), 0)
    np.testing.assert_allclose(need, minimax_bruteforce(pos, 0), atol=1e-9)


def test_fuel_bottleneck_no_stations_is_distance():
    pos = _cloud(60, seed=10)
    g = csr_matrix(cdist(pos, pos))
    need, _ = fuel_bottleneck(g, np.zeros(len(pos), bool), 0)
    np.testing.assert_allclose(need, np.linalg.norm(pos - pos[0], axis=1), atol=1e-9)


# --- tours ----------------------------------------------------------------- #
def test_two_opt_improves_and_respects_mst_bound():
    pos = _cloud(150, seed=11)
    D = cdist(pos, pos)
    nn = nn_tour(D, 0)
    opt = two_opt(D, nn)
    assert sorted(opt.tolist()) == list(range(150)) and opt[0] == 0
    assert path_length(D, opt) <= path_length(D, nn) + 1e-9
    assert path_length(D, opt) >= mst_length(D) - 1e-9


# --- physics --------------------------------------------------------------- #
def test_rocket_speed_limits():
    v = rocket_cruise_speed_kms(4.4, 10.0)
    assert v == pytest.approx(0.5 * 4.4 * np.log(10.0), rel=1e-9)
    assert rocket_cruise_speed_kms(0.33 * 299_792.458, 1e6) < 299_792.458


def test_brachistochrone_limits_and_quadrature():
    # nonrelativistic limit: t = 2 sqrt(D/a)
    d_pc = 1e-6
    t, tau = brachistochrone_times_yr(d_pc, 1.0)
    D = d_pc * 3.0856775814913673e16
    assert t * 365.25 * 86400 == pytest.approx(2 * np.sqrt(D / 9.80665), rel=1e-6)
    # proper time = integral of dt / gamma over the coordinate-time history
    t, tau = brachistochrone_times_yr(4.37, 1.0)
    c, a = 299_792_458.0, 9.80665
    th = 0.5 * t * 365.25 * 86400
    ts = np.linspace(0, th, 200_001)
    gamma = np.sqrt(1 + (a * ts / c) ** 2)
    tau_num = 2 * np.trapezoid(1 / gamma, ts) / (365.25 * 86400)
    assert tau == pytest.approx(tau_num, rel=1e-6)
    assert t > 4.37 * 3.26156 and tau < t       # slower than light; crew ages less


def test_sail_and_range():
    assert sail_brake_speed_kms(1.52) == pytest.approx(13_800.0)
    assert sail_brake_speed_kms(10.0) > sail_brake_speed_kms(0.1)
    # 1 km/s for 1 Myr ~ 1.0227 pc
    assert hop_range_pc(1.0, 1e6) == pytest.approx(KMS_TO_PCMYR)


# --- kinematics ------------------------------------------------------------ #
def test_cartesian_matches_astropy():
    astropy = pytest.importorskip("astropy")  # noqa: F841
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(12)
    n = 20
    ra, dec = rng.uniform(0, 360, n), rng.uniform(-89, 89, n)
    plx, pmra, pmdec, rv = rng.uniform(20, 300, n), rng.normal(0, 500, n), rng.normal(0, 500, n), rng.normal(0, 40, n)
    pos, vel = astrometry_to_cartesian(ra, dec, plx, pmra, pmdec, rv)
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=(1000 / plx) * u.pc,
                  pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                  radial_velocity=rv * u.km / u.s, frame="icrs")
    np.testing.assert_allclose(pos, sc.cartesian.xyz.to(u.pc).value.T, rtol=1e-10)
    v_ap = sc.velocity.d_xyz.to(u.km / u.s).value.T * KMS_TO_PCMYR
    np.testing.assert_allclose(vel, v_ap, rtol=1e-6)


def test_gliese_710_encounter():
    """Gaia DR3 4270814637616488064. Bailer-Jones (2022): perihelion ~0.05-0.07
    pc (~13,000 au) in ~1.3 Myr. Linear motion must reproduce it."""
    pos, vel = astrometry_to_cartesian(
        np.array([274.96183629691126]), np.array([-1.938612759804832]),
        np.array([52.396292205290564]), np.array([-0.4140637404226508]),
        np.array([-0.10847507667730025]), np.array([-14.419985]))
    t, d = closest_approach(pos, vel)
    assert 1.2 < t[0] < 1.4
    assert 0.03 < d[0] < 0.09


def test_conditional_rv_fill_is_unbiased():
    """Hide half the RVs of a synthetic Gaussian velocity field; the filled
    values must be unbiased and keep the star count in a central ball stable
    (no artificial infall), unlike unconditional draws."""
    from src.model.kinematics import K_AUYR, fill_missing_rv, unit_vectors
    rng = np.random.default_rng(13)
    n = 40_000
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    dist = 50 * rng.uniform(size=n) ** (1 / 3)
    ra = np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360
    dec = np.degrees(np.arcsin(u[:, 2]))
    mu = np.array([-11.0, -20.0, -7.0])
    A = rng.normal(size=(3, 3)) * 8
    v = mu + rng.normal(size=(n, 3)) @ (A + np.diag([25.0, 15.0, 12.0]))
    r_hat, p_hat, q_hat = unit_vectors(ra, dec)
    plx = 1000 / dist
    rv_true = np.einsum("ij,ij->i", v, r_hat)
    pmra = np.einsum("ij,ij->i", v, p_hat) * plx / K_AUYR
    pmdec = np.einsum("ij,ij->i", v, q_hat) * plx / K_AUYR
    has = rng.uniform(size=n) < 0.5
    rv = np.where(has, rv_true, np.nan)
    filled = fill_missing_rv(ra, dec, plx, pmra, pmdec, rv, has, rng)
    resid = filled[~has] - rv_true[~has]
    assert abs(resid.mean()) < 0.5
    pos_true, vel_true = astrometry_to_cartesian(ra, dec, plx, pmra, pmdec, rv_true)
    _, vel_fill = astrometry_to_cartesian(ra, dec, plx, pmra, pmdec, filled)
    count = lambda vv: np.sum(np.linalg.norm(pos_true + vv * 0.5, axis=1) <= 15)  # noqa: E731
    assert count(vel_fill) == pytest.approx(count(vel_true), rel=0.08)
