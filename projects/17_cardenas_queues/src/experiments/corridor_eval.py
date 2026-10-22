"""Shared evaluation: analytic tandem model + CTM run, summarised per segment.

Segment predictors (computed without simulation):
  lane_ratio      feeder lanes entering the segment / mean corridor lanes
  lane_ratio_cum  feeder lanes entering up to the end of the segment / its lanes
                  (the "150 lanes into 6" argument, localised)
  feeder_cum      feeder lanes entering up to the end of the segment (numerator only)
  inv_capacity    1 / min cell capacity (geometry + signals, no demand)
  rho             max steady-state utilisation of the segment's cells
Ground truth from the CTM:
  active_share    fraction of time an active bottleneck sits in the segment
  delay_vh        mainline delay in the segment's cells + queue delay of its ramps
"""
from __future__ import annotations

import dataclasses

import numpy as np

from src.analysis.queues import tandem_metrics
from src.data.network import build_direction, segments
from src.sim.ctm import simulate


def analytic(cw, dem) -> dict:
    served = np.minimum(dem, cw.ramp_cap_vph)
    ext = np.zeros(len(cw.lanes))
    np.add.at(ext, cw.ramp_cell, served)
    m = tandem_metrics(ext, cw.beta, cw.cap_vph)
    m["ramp_rho"] = dem / cw.ramp_cap_vph
    return m


def evaluate(inv, d, th, cfg, p, demand_fn, mods=None, metering_fn=None) -> dict:
    """Build direction d, run analytic + CTM; demand_fn(base_vph) -> (T, R)."""
    if "fifo" in th:
        p = dataclasses.replace(p, fifo=th["fifo"])
    if "cap_drop" in th:
        p = dataclasses.replace(p, cap_drop=th["cap_drop"])
    net = build_direction(inv, d, th, cfg["control"], cfg["corridor"], mods)
    cw, dem = net["cw"], net["ramp_demand_vph"]
    an = analytic(cw, dem)
    metering = metering_fn(cw) if metering_fn else None
    r = simulate(cw, demand_fn(dem), p, metering=metering)
    segs = segments(inv, cw)
    kc = p.kc_vpkm
    k = r["density"]
    active_t = (k[:, :-1] > kc) & (k[:, 1:] <= kc)          # (t, boundary)
    active_t = np.concatenate([active_t, np.zeros((len(k), 1), bool)], axis=1)
    through_lanes = cw.ramp_lanes[0]
    feeder = np.zeros(len(cw.lanes))
    np.add.at(feeder, cw.ramp_cell[1:], cw.ramp_lanes[1:])
    feeder_cum = np.cumsum(feeder)
    ramp_delay_cell = np.zeros(len(cw.lanes))
    np.add.at(ramp_delay_cell, cw.ramp_cell, r["delay_ramp_vh_each"])
    rows = []
    for s in segs:
        c = s["cells"]
        lanes = float(cw.lanes[c].mean())
        rows.append({
            "c0": s["c0"], "c1": s["c1"], "n_cells": int(len(c)),
            "lanes": lanes, "feeder_lanes": float(feeder[c].sum()),
            "lane_ratio": float(feeder[c].sum() / lanes),
            "lane_ratio_cum": float(feeder_cum[c].max() / cw.lanes[c[np.argmax(feeder_cum[c])]]),
            "feeder_cum": float(feeder_cum[c].max()),
            "inv_capacity": float(1.0 / cw.cap_vph[c].min()),
            "rho": float(an["rho"][c].max()),
            "active_share": float(active_t[:, c].any(axis=1).mean()),
            "delay_vh": float(r["delay_main_cell_vh"][c].sum() + ramp_delay_cell[c].sum()),
            "n_signals": int(sum(1 for x in cw.meta["signal_cells"] if x in set(c.tolist()))),
        })
    k_cong = (k > kc)
    return {
        "dir": d, "cw": cw, "dem": dem, "analytic": an, "ctm": r, "segments": rows,
        "through_lanes": float(through_lanes),
        "delay_total_vh": r["delay_total_vh"], "delay_main_vh": r["delay_main_vh"],
        "delay_ramp_vh": r["delay_ramp_vh"],
        "max_congested_km": float(k_cong.sum(axis=1).max() * p.dx_km),
        "bottleneck_cell_ctm": int(np.argmax(active_t.mean(axis=0))) if active_t.any() else -1,
        "bottleneck_cell_rho": int(an["bottleneck"]),
        "max_rho": float(an["rho"].max()),
    }


def seg_of_cell(result: dict, cell: int) -> int:
    for i, s in enumerate(result["segments"]):
        if s["c0"] <= result["cw"].meta["chainage_m"][cell] <= s["c1"]:
            return i
    return -1
