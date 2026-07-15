"""Unit tests: closed-form physics vs numerical integration, and the
water-filling optimum vs an exact LP. Fast (no datasets, tiny instances).
`make test` must pass before running experiments.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from src.model.insolation import (  # noqa: E402
    S0,
    band_transmittance,
    daily_toa_insolation,
    daily_toa_insolation_numeric,
    global_annual_mean_toa,
    global_mean_surface_ghi,
    loss_cascade,
    toa_intercepted_power,
)
from src.model.landuse import land_budget  # noqa: E402
from src.optimize.light_budget import (  # noqa: E402
    daily_totals,
    flat_harvest,
    lp_optimum_value,
    min_light_from_fraction,
    no_cap_harvest,
    optimal_harvest,
)


# --- Geometry -------------------------------------------------------------- #
def test_daily_insolation_matches_numeric_integration():
    for lat in (0.0, 23.45, 45.0, -30.0, 70.0):
        for doy in (1, 80, 172, 266):     # solstices/equinoxes-ish
            closed = daily_toa_insolation(lat, doy)
            numeric = daily_toa_insolation_numeric(lat, doy)
            assert abs(closed - numeric) < 1e-3 * max(closed, 1.0), (lat, doy)


def test_polar_night_and_midnight_sun():
    # North pole at northern winter solstice -> polar night (zero insolation).
    assert daily_toa_insolation(89.9, 355) < 1.0
    # North pole at northern summer solstice -> midnight sun (positive).
    assert daily_toa_insolation(89.9, 172) > 1e6


def test_global_mean_toa_is_quarter_solar_constant():
    assert abs(global_annual_mean_toa() - S0 / 4.0) < 0.5


def test_intercepted_power_sanity():
    # ~1.74e17 W intercepted by the disk.
    assert 1.7e17 < toa_intercepted_power() < 1.8e17


# --- Atmospheric cascade --------------------------------------------------- #
def test_cascade_is_monotone_and_lands_near_surface_ghi():
    levels = loss_cascade()
    vals = [lv.irradiance for lv in levels]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals   # non-increasing
    assert abs(vals[-1] - global_mean_surface_ghi()) < 1e-6
    assert 170.0 < vals[-1] < 200.0                            # ~184 W/m^2


def test_transmittance_bounded():
    for lat in range(0, 90, 5):
        assert 0.0 < band_transmittance(lat) <= 1.0


# --- Land budget ----------------------------------------------------------- #
def test_deployable_power_exceeds_world_electricity():
    lb = land_budget()
    assert lb.frac_of_electricity > 1.0        # potential is not the bottleneck
    assert lb.deployable_power > lb.deployable_area  # sanity: W > m^2 here


# --- Light-budget optimisation --------------------------------------------- #
def test_energy_conservation():
    rng = np.random.default_rng(0)
    L = rng.uniform(0, 1000, size=(4, 24))
    w = rng.uniform(0.1, 2.0, size=(4, 24))
    d_min = min_light_from_fraction(L, 0.3)
    for res in (no_cap_harvest(L, w), flat_harvest(L, w, 0.3), optimal_harvest(L, w, d_min)):
        assert abs(res.harvest_energy + res.reserved_energy - L.sum()) < 1e-6


def test_optimal_meets_reserve_and_beats_flat():
    rng = np.random.default_rng(1)
    L = rng.uniform(0, 1000, size=(5, 24))
    w = rng.uniform(0.1, 2.0, size=(5, 24))
    f_min = 0.4
    d_min = min_light_from_fraction(L, f_min)
    opt = optimal_harvest(L, w, d_min)
    flat = flat_harvest(L, w, f_min)
    nocap = no_cap_harvest(L, w)
    # both feasible policies reserve the same total; optimum harvests more value
    assert abs(opt.reserved_energy - flat.reserved_energy) < 1e-6
    assert opt.harvest_value >= flat.harvest_value - 1e-9
    assert nocap.harvest_value >= opt.harvest_value - 1e-9   # blind is the upper bound


def test_greedy_matches_exact_lp():
    rng = np.random.default_rng(2)
    for _ in range(20):
        L = rng.uniform(0, 1000, size=24)
        w = rng.uniform(0.1, 2.0, size=24)
        d_min = 0.35 * L.sum()
        greedy = optimal_harvest(L, w, d_min).harvest_value
        exact = lp_optimum_value(L, w, d_min)
        assert abs(greedy - exact) < 1e-6, (greedy, exact)
