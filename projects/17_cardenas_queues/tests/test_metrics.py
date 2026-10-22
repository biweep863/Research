"""Unit tests: queueing closed forms, capacity/delay, fluid queue, CTM
conservation and bottleneck physics, DES vs theory, corridor builders.
Seconds, no network."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

from src.analysis.capacity import (corridor_capacity, hcm_delay, lane_group_capacity,  # noqa: E402
                                   minor_stream_capacity, webster_delay)
from src.analysis.fluid import fluid_queue, rectangular_peak                          # noqa: E402
from src.analysis.queues import (erlang_c, kingman_wq, md1, mm1, mmc,                  # noqa: E402
                                 tandem_flows, tandem_metrics)
from src.data.network import balance_exits, base_theta, build_direction, peak_profile  # noqa: E402
from src.data.synthetic import synthetic_corridor                                       # noqa: E402
from src.sim.ctm import Carriageway, CTMParams, active_bottlenecks, simulate           # noqa: E402
from src.sim.des import multi_server_fifo, signal_queue                                 # noqa: E402

DCFG = {"per_lane_vph": {"major": [500, 800, 1100], "tertiary": [250, 450, 650],
                         "minor": [30, 100, 200], "service": [5, 25, 80], "retorno": [20, 80, 200]},
        "turn_terminal": [0.8, 1.0, 1.0], "turn_crossing": [0.15, 0.3, 0.45],
        "upstream_vph": [2500, 3500, 4500]}
CTRL = {"s0": 1900, "g_main": [0.45, 0.55, 0.65], "lost_frac": 0.1, "minor_cap_per_lane": 700,
        "link_cap_per_lane": 1600, "friction_per_access": [0.0, 0.015, 0.03]}
CCFG = {"major_classes": ["trunk", "trunk_link", "primary", "secondary"], "infer_signals": True}


# ---- closed forms ------------------------------------------------------------
def test_mm1_little_and_known_values():
    r = mm1(0.8, 1.0)
    assert r["L"] == pytest.approx(4.0)
    assert r["Lq"] == pytest.approx(3.2)
    assert r["W"] == pytest.approx(r["L"] / 0.8)          # Little's law
    assert mm1(1.0, 1.0)["L"] == float("inf")


def test_md1_is_half_of_mm1_queue():
    assert md1(0.8, 1.0)["Lq"] == pytest.approx(mm1(0.8, 1.0)["Lq"] / 2)


def test_erlang_c_table_and_reduction():
    # Textbook value: c = 2 servers, a = 1 Erlang -> P(wait) = 1/3.
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)
    # c = 1 reduces to rho.
    assert erlang_c(1, 0.7) == pytest.approx(0.7)
    assert mmc(0.7, 1.0, 1)["Wq"] == pytest.approx(mm1(0.7, 1.0)["Wq"])


def test_kingman_exact_for_mm1_and_md1():
    assert kingman_wq(0.8, 1.0, 1, 1) == pytest.approx(mm1(0.8, 1.0)["Wq"])
    assert kingman_wq(0.8, 1.0, 1, 0) == pytest.approx(md1(0.8, 1.0)["Wq"])


def test_tandem_flows_and_metering_bottleneck():
    lam = tandem_flows([100, 50, 50], [0.5, 0.0, 0.0])
    assert np.allclose(lam, [100, 100, 150])
    m = tandem_metrics([300, 0, 0], [0, 0, 0], [200, 1000, 150])
    # Node 0 meters the flow to 200, so node 2 sees 200 > 150 only once node 0's limit applies.
    assert m["served"][0] == pytest.approx(200)
    assert m["lam"][2] == pytest.approx(200)
    assert m["bottleneck"] == 0 and m["n_unstable"] == 2


# ---- capacity and delay ------------------------------------------------------
def test_capacity_formulas():
    assert lane_group_capacity(3, 0.5) == pytest.approx(2850)
    assert corridor_capacity(3, 2, 1.0, 0.5) == pytest.approx(1900 * 4)
    # Gap acceptance capacity falls with conflicting flow.
    assert minor_stream_capacity(1, 300) > minor_stream_capacity(1, 1200) > 0


def test_webster_and_hcm_agree_at_low_saturation():
    w = webster_delay(90, 0.5, 400, 950)
    h = hcm_delay(90, 0.5, 400, 950)
    assert w == pytest.approx(h, rel=0.25)
    assert np.isfinite(hcm_delay(90, 0.5, 1100, 950))
    assert webster_delay(90, 0.5, 1100, 950) == float("inf")


# ---- fluid -------------------------------------------------------------------
def test_fluid_matches_rectangular_closed_form():
    dt = 1 / 3600
    T, mu, lp, lo = 1.0, 1000.0, 1300.0, 600.0
    t = np.arange(int(3.0 / dt)) * dt
    lam = np.where(t < T, lp, lo)
    f = fluid_queue(lam, mu, dt)
    cf = rectangular_peak(lp, lo, mu, T)
    assert f["max_queue_veh"] == pytest.approx(cf["max_queue_veh"], rel=1e-3)
    assert f["total_delay_vh"] == pytest.approx(cf["total_delay_vh"], rel=1e-2)


# ---- CTM ---------------------------------------------------------------------
def _line(n=40, lanes=3, cap=None):
    lanes_a = np.full(n, float(lanes))
    cap_a = np.full(n, 1900.0 * lanes) if cap is None else cap
    return Carriageway(lanes=lanes_a, cap_vph=cap_a, beta=np.zeros(n), ramp_cell=np.array([0, 20]),
                       ramp_cap_vph=np.array([1900.0 * lanes, 1600.0]), ramp_lanes=np.array([3.0, 1.0]))


def test_ctm_free_flow_has_no_delay_and_conserves():
    p = CTMParams()
    cw = _line()
    dem = np.tile([2000.0, 500.0], (600, 1))
    r = simulate(cw, dem, p)
    assert r["delay_total_vh"] == pytest.approx(0.0, abs=1e-9)
    assert r["total_in"] == pytest.approx(r["total_out"] + r["total_exit"] + r["stored_end"])


def test_ctm_bottleneck_queue_matches_fluid_point_queue():
    """A single lane-drop bottleneck: CTM total delay ~ fluid point-queue delay
    (the CTM queue has length, so delay is slightly larger, not smaller)."""
    p = CTMParams()
    cap = np.full(40, 1900.0 * 3)
    cap[30] = 1900.0 * 2                              # bottleneck at boundary 30 -> 31
    cw = _line(cap=cap)
    cw.ramp_cell = np.array([0])
    cw.ramp_cap_vph = np.array([1900.0 * 3])
    cw.ramp_lanes = np.array([3.0])
    T = int(3.0 * 3600 / p.dt_s)
    t = (np.arange(T) + 0.5) * p.dt_h
    lam = np.where(t < 1.0, 4500.0, 2500.0)
    r = simulate(cw, lam[:, None], p)
    f = fluid_queue(lam, 3800.0, p.dt_h)
    assert r["total_in"] == pytest.approx(r["total_out"] + r["stored_end"])
    assert r["delay_total_vh"] == pytest.approx(f["total_delay_vh"], rel=0.15)
    ab = active_bottlenecks(r["density"], p.kc_vpkm)
    assert int(np.argmax(ab)) == 30


def test_ctm_merge_priority_and_metering_hold_back_ramp():
    from src.sim.ctm import Metering
    p = CTMParams()
    cw = _line()
    dem = np.tile([5000.0, 1500.0], (900, 1))           # 6500 > 5700: merge is binding
    base = simulate(cw, dem, p)
    met = simulate(cw, dem, p, Metering(ramps=np.array([1])))
    assert met["delay_ramp_vh_each"][1] >= base["delay_ramp_vh_each"][1]


# ---- DES vs theory -----------------------------------------------------------
def test_des_mmc_and_md1():
    rng = np.random.default_rng(0)
    for c in (1, 2):
        lam, mu = 0.7 * c, 1.0
        sim = multi_server_fifo(lam, mu, c, 60000, rng)["Wq"]
        assert sim == pytest.approx(mmc(lam, mu, c)["Wq"], rel=0.15)
    sim = multi_server_fifo(0.7, 1.0, 1, 60000, rng, service="det")["Wq"]
    assert sim == pytest.approx(md1(0.7, 1.0)["Wq"], rel=0.15)


def test_des_signal_close_to_webster():
    rng = np.random.default_rng(1)
    d = signal_queue(600, 1900, 90, 0.5, 30000, rng)
    assert d == pytest.approx(webster_delay(90, 0.5, 600, 950), rel=0.15)


# ---- corridor / network --------------------------------------------------------
def test_synthetic_network_builds_and_balances_exits():
    inv = synthetic_corridor(0)
    th = base_theta(DCFG, CTRL)
    for d in ("WB", "EB"):
        net = build_direction(inv, d, th, CTRL, CCFG)
        cw = net["cw"]
        assert len(cw.lanes) == len(inv["bins_m"])
        assert (cw.beta >= 0).all() and (cw.beta <= 0.5).all()
        assert net["ramp_demand_vph"][0] == pytest.approx(th["upstream_vph"])


def test_balance_exits_hits_target():
    beta = balance_exits(np.array([0, 2, 5]), np.array([1000.0, 300.0, 300.0]),
                         np.array([0, 0, 0, 1.0, 0, 0, 1.0, 0]), target=600.0)
    F, out = 0.0, 0.0
    inflow = np.zeros(8); inflow[[0, 2, 5]] = [1000, 300, 300]
    for c in range(8):
        F += inflow[c]; out += beta[c] * F; F *= 1 - beta[c]
    assert out == pytest.approx(600.0, rel=1e-6)


def test_peak_profile_shape():
    prof = peak_profile(600, 0.01, 1.0, 2.0, 0.5, 0.4)
    assert prof.min() == pytest.approx(0.4) and prof.max() == pytest.approx(1.0)


def test_capacity_drop_increases_delay():
    import dataclasses
    p = CTMParams()
    cap = np.full(40, 1900.0 * 3)
    cap[30] = 1900.0 * 2
    cw = _line(cap=cap)
    cw.ramp_cell = np.array([0]); cw.ramp_cap_vph = np.array([5700.0]); cw.ramp_lanes = np.array([3.0])
    T = int(3.0 * 3600 / p.dt_s)
    t = (np.arange(T) + 0.5) * p.dt_h
    lam = np.where(t < 1.0, 4300.0, 2500.0)[:, None]
    a = simulate(cw, lam, p)["delay_total_vh"]
    b = simulate(cw, lam, dataclasses.replace(p, cap_drop=0.1))["delay_total_vh"]
    assert b > a * 1.2


def test_partial_fifo_bounds():
    import dataclasses
    p = CTMParams()
    cap = np.full(40, 1900.0 * 3)
    cap[30] = 1900.0 * 2
    cw = _line(cap=cap)
    cw.beta = np.zeros(40); cw.beta[20] = 0.3
    cw.ramp_cell = np.array([0]); cw.ramp_cap_vph = np.array([5700.0]); cw.ramp_lanes = np.array([3.0])
    lam = np.tile([5400.0], (1200, 1))
    d = [simulate(cw, lam, dataclasses.replace(p, fifo=f))["delay_total_vh"] for f in (0.0, 0.5, 1.0)]
    assert d[0] <= d[1] <= d[2]


def test_field_counts_phase_rates_cover_intersection(tmp_path):
    from src.data.counts import apply_field_counts, hourly_range
    from src.data.network import approach_flow
    assert hourly_range({"veh_min_green": "14-18", "veh_min_red": "7-10", "green_frac": "0.5"}, 0.35) == (630.0, 840.0)
    inv = synthetic_corridor(0)
    a0 = next(a for a in inv["approaches"] if a["name"] == "Junction 1" and a["dir"] == "WB")
    f = tmp_path / "c.csv"
    f.write_text("name,dir,veh_min_green,veh_min_red,green_frac,free_lanes\nJunction 1,WB,10,10,,1\n")
    assert apply_field_counts(inv, f) == 1
    hit = [a for a in inv["approaches"] if "observed_vph" in a]
    assert all(a["dir"] == "WB" for a in hit) and a0 in hit
    th = base_theta(DCFG, CTRL)
    assert sum(approach_flow(a, th) for a in hit) == pytest.approx(600.0)
    net = build_direction(inv, "WB", th, CTRL, CCFG)
    assert net["cw"].ramp_cap_vph.min() > 0
