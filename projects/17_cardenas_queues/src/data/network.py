"""From corridor inventory + demand/control parameters to simulation inputs.

For each direction the corridor becomes a line of 100 m cells in travel order.
Every approach becomes a "ramp" (point queue) entering one cell; the corridor
end is ramp 0 (through traffic).  Exits are split fractions balanced so that
the vehicles leaving through side streets equal those entering from them
(gravity by exit lanes x class rate), so the corridor carries its through
demand plus whatever has joined and not yet left.

A parameter set `theta` (one Monte Carlo draw) fixes every assumption:
per-lane class rates, turn fractions, through demand, green ratio, friction.
"""
from __future__ import annotations

import numpy as np

from src.data.corridor import BIN_M, _is_major
from src.sim.ctm import Carriageway

CLASSES = ("major", "tertiary", "minor", "service")


def base_theta(dcfg: dict, ccfg: dict, fifo_range=None, drop_range=None) -> dict:
    """Middle value of every [low, base, high] range."""
    th = {f"q_{k}": v[1] for k, v in dcfg["per_lane_vph"].items()}
    th.update(turn_terminal=dcfg["turn_terminal"][1], turn_crossing=dcfg["turn_crossing"][1],
              upstream_vph=dcfg["upstream_vph"][1], g_main=ccfg["g_main"][1],
              friction=ccfg["friction_per_access"][1])
    if fifo_range is not None:
        th["fifo"] = fifo_range[1]
    if drop_range is not None:
        th["cap_drop"] = drop_range[1]
    return th


def draw_theta(dcfg: dict, ccfg: dict, rng: np.random.Generator, fifo_range=None,
               drop_range=None) -> dict:
    """Uniform draw inside every [low, high] range (log-uniform for class rates)."""
    th = {f"q_{k}": float(np.exp(rng.uniform(np.log(v[0]), np.log(v[2]))))
          for k, v in dcfg["per_lane_vph"].items()}
    u = lambda r: float(rng.uniform(r[0], r[2]))  # noqa: E731
    th.update(turn_terminal=u(dcfg["turn_terminal"]), turn_crossing=u(dcfg["turn_crossing"]),
              upstream_vph=u(dcfg["upstream_vph"]), g_main=u(ccfg["g_main"]),
              friction=u(ccfg["friction_per_access"]))
    if fifo_range is not None:
        th["fifo"] = u(fifo_range)
    if drop_range is not None:
        th["cap_drop"] = u(drop_range)
    th["obs_u"] = float(rng.uniform())          # position inside every field-count range
    return th


def approach_flow(a: dict, th: dict) -> float:
    """Peak flow an approach delivers onto the corridor (veh/h)."""
    turn = th["turn_terminal"] if a["kind"] == "terminal" else th["turn_crossing"]
    share = 0.5 if a["dir"] == "both" else 1.0
    if "observed_vph" in a:
        lo, hi = a["observed_vph"]
        return (lo + th.get("obs_u", 0.5) * (hi - lo)) * share
    return a["lanes"] * th[f"q_{a['dclass']}"] * turn * share


def build_direction(inv: dict, d: str, th: dict, ctrl: dict, ccfg: dict,
                    mods: dict | None = None) -> dict:
    """Carriageway + peak ramp demand for direction d ("WB" or "EB").

    mods (interventions / robustness): {"extra_lanes": {cell: +lanes},
    "g_override": {cell: g}, "consolidate_minor": bool, "fill_lateral_gaps": bool}.
    """
    mods = mods or {}
    s0 = ctrl["s0"]
    bins = np.asarray(inv["bins_m"], float)
    exp_l = np.asarray(inv["lanes"][d]["express"], float)
    lat_l = np.asarray(inv["lanes"][d]["lateral"], float)
    order = np.arange(len(bins)) if d == "WB" else np.arange(len(bins))[::-1]
    exp_l, lat_l, bins_o = exp_l[order], lat_l[order], bins[order]
    # Ends of each lane group that OSM leaves unmapped take the nearest mapped
    # value; interior gaps (laterals interrupted at interchanges) are real.
    exp_l, lat_l = _fill_edges(exp_l), _fill_edges(lat_l)
    if mods.get("fill_lateral_gaps"):
        nz = np.nonzero(lat_l > 0)[0]
        lat_l = np.interp(np.arange(len(lat_l)), nz, lat_l[nz])
    lanes = exp_l + lat_l
    n_c = len(lanes)
    for c, extra in mods.get("extra_lanes", {}).items():
        lanes[c] += extra
        lat_l[c] += extra

    def cell_of(ch):
        k = int(np.clip((ch - bins[0]) // BIN_M, 0, len(bins) - 1))
        return int(np.nonzero(order == k)[0][0])

    major = set(ccfg["major_classes"])
    apps = [a for a in inv["approaches"] if a["dir"] in (d, "both")]
    deps = [a for a in inv["departures"] if a["dir"] in (d, "both")]

    if mods.get("consolidate_minor"):
        # Close minor/service accesses (same total demand, fewer merge points).
        keep = ("major", "tertiary", "retorno")
        kept = [a for a in apps if a["dclass"] in keep]
        moved = [a for a in apps if a["dclass"] not in keep]
        apps = kept
    else:
        moved = []

    # Signals: OSM-tagged on this carriageway, plus (optionally) inferred at
    # lateral junctions fed by major / >= 2-lane tertiary streets.
    sig_cells = {cell_of(s["chainage"]) for s in inv["signals"] if s["dir"] == d}
    if ccfg.get("infer_signals"):
        for a in apps:
            if _is_major(a, major) and not a["highway"].endswith("_link") and a["side"] == "lateral":
                sig_cells.add(cell_of(a["chainage"]))
    g = np.ones(n_c)
    for c in sig_cells:
        g[c] = th["g_main"]
    for c, gv in mods.get("g_override", {}).items():
        g[c] = gv

    # Ramps: 0 = through traffic entering cell 0.
    r_cell = [0]
    r_dem = [th["upstream_vph"]]
    r_cap = [lanes[0] * s0]
    r_lanes = [lanes[0]]
    r_meta = [{"name": "through", "dclass": "through", "lanes": float(lanes[0]), "major": True}]
    n_unsig = np.zeros(n_c)
    for a in apps:
        c = cell_of(a["chainage"])
        q = approach_flow(a, th)
        if "free_lanes" in a:
            # Counted intersection: continuous-turn lanes merge freely, the rest
            # discharge only on the cross street's green.
            g_corr = g[c] if g[c] < 1 else th["g_main"]
            free = min(a["free_lanes"], a["lanes"])
            cap = (free * ctrl["link_cap_per_lane"]
                   + (a["lanes"] - free) * s0 * max(1 - g_corr - ctrl["lost_frac"], 0.05))
        elif a["highway"].endswith("_link"):
            cap = a["lanes"] * ctrl["link_cap_per_lane"]
        elif c in sig_cells or a["signal"]:
            g_corr = g[c] if g[c] < 1 else th["g_main"]
            cap = a["lanes"] * s0 * max(1 - g_corr - ctrl["lost_frac"], 0.05)
        else:
            cap = a["lanes"] * ctrl["minor_cap_per_lane"]
            n_unsig[c] += 1
        r_cell.append(c)
        r_dem.append(q)
        r_cap.append(cap)
        r_lanes.append(a["lanes"])
        r_meta.append({"name": a["name"], "dclass": a["dclass"], "lanes": a["lanes"],
                       "highway": a["highway"], "chainage": a["chainage"],
                       "major": _is_major(a, major)})
    # Consolidation: the closed accesses of each segment feed a collector
    # that joins the corridor once, at the segment's last closed access,
    # through one well-designed merge lane.
    edges = sorted(j["chainage"] for j in inv["junctions"])
    groups = {}
    for a in moved:
        seg = int(np.searchsorted(edges, a["chainage"]))
        groups.setdefault(seg, []).append(a)
    for seg, grp in groups.items():
        last = max(grp, key=lambda a: a["chainage"]) if d == "WB" else min(grp, key=lambda a: a["chainage"])
        r_cell.append(cell_of(last["chainage"]))
        r_dem.append(sum(approach_flow(a, th) for a in grp))
        r_cap.append(ctrl["link_cap_per_lane"])
        r_lanes.append(1.0)
        r_meta.append({"name": f"collector {seg}", "dclass": "collector", "lanes": 1.0,
                       "highway": "collector", "chainage": last["chainage"], "major": False})

    r_cell = np.array(r_cell)
    r_dem = np.array(r_dem, float)
    r_cap = np.array(r_cap, float)

    cap = s0 * (exp_l + lat_l * g)
    cap *= np.clip(1 - th["friction"] * n_unsig, 0.5, 1.0)

    # Exits: attraction by lanes x class rate x turn, balanced to side inflow.
    attr = np.zeros(n_c)
    for a in deps:
        attr[cell_of(a["chainage"])] += approach_flow(a, th)
    beta = balance_exits(r_cell, np.minimum(r_dem, r_cap), attr, target=np.minimum(r_dem, r_cap)[1:].sum())

    cw = Carriageway(lanes=lanes, cap_vph=cap, beta=beta, ramp_cell=r_cell, ramp_cap_vph=r_cap,
                     ramp_lanes=np.array(r_lanes, float), sink_cap_vph=float(lanes[-1] * s0),
                     meta={"dir": d, "chainage_m": bins_o + BIN_M / 2, "signal_cells": sorted(sig_cells),
                           "ramps": r_meta, "express": exp_l, "lateral": lat_l, "g": g,
                           "n_unsig": n_unsig})
    return {"cw": cw, "ramp_demand_vph": r_dem}


def _fill_edges(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    nz = np.nonzero(x > 0)[0]
    if len(nz):
        x[: nz[0]] = x[nz[0]]
        x[nz[-1] + 1:] = x[nz[-1]]
    return x


def balance_exits(ramp_cell, ramp_in, attr, target, beta_max: float = 0.5) -> np.ndarray:
    """Exit fractions beta_c = min(kappa attr_c / F_c, beta_max) with kappa chosen
    (bisection) so that total exiting flow equals `target`."""
    n_c = len(attr)
    inflow = np.zeros(n_c)
    np.add.at(inflow, ramp_cell, ramp_in)

    def run(kappa):
        F = 0.0
        beta = np.zeros(n_c)
        out = 0.0
        for c in range(n_c):
            F += inflow[c]
            if attr[c] > 0 and F > 0:
                beta[c] = min(kappa * attr[c] / F, beta_max)
                out += beta[c] * F
                F *= 1 - beta[c]
        return beta, out

    if attr.sum() == 0 or target <= 0:
        return np.zeros(n_c)
    lo, hi = 0.0, 1.0
    while run(hi)[1] < target and hi < 1e6:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if run(mid)[1] < target:
            lo = mid
        else:
            hi = mid
    return run(hi)[0]


def segments(inv: dict, cw: Carriageway) -> list[dict]:
    """Corridor segments between junctions, in this direction's travel order."""
    ch = cw.meta["chainage_m"]
    cuts = sorted(j["chainage"] for j in inv["junctions"])
    edges = [-np.inf] + cuts + [np.inf]
    segs = []
    for k in range(len(edges) - 1):
        cells = np.nonzero((ch >= edges[k]) & (ch < edges[k + 1]))[0]
        if len(cells):
            segs.append({"k": k, "cells": cells, "c0": float(ch[cells].min()), "c1": float(ch[cells].max())})
    if cw.meta["dir"] == "EB":
        segs = segs[::-1]
    return segs


def peak_profile(T: int, dt_h: float, start_h: float, len_h: float, ramp_h: float,
                 offpeak: float) -> np.ndarray:
    """Trapezoidal demand multiplier: offpeak -> 1 over ramp_h, hold len_h, back down."""
    t = (np.arange(T) + 0.5) * dt_h
    up = np.clip((t - start_h) / ramp_h, 0, 1)
    down = np.clip((start_h + ramp_h + len_h + ramp_h - t) / ramp_h, 0, 1)
    return offpeak + (1 - offpeak) * np.minimum(up, down)
