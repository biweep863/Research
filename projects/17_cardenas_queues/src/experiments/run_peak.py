"""Experiment 4: the peak hour in the Cell Transmission Model.

Base parameters: space-time density, queue heads (active bottlenecks), delay
by segment, and a single-bottleneck fluid queue for comparison.  Then a grid
of demand scale x peak length, and an OSM robustness check that fills the
lateral-lane gaps.
"""
from __future__ import annotations

import numpy as np

from src.analysis.fluid import fluid_queue
from src.data.network import base_theta, peak_profile
from src.experiments.common import ctm_params, demand_matrix, dump, load_cfg, load_inventory
from src.experiments.corridor_eval import evaluate


def summarise(ev: dict, p, keep_maps: bool = False) -> dict:
    cw, r = ev["cw"], ev["ctm"]
    ch = cw.meta["chainage_m"]
    out = {k: ev[k] for k in ("delay_total_vh", "delay_main_vh", "delay_ramp_vh", "max_congested_km", "max_rho")}
    veh = float(r["total_in"])
    out["vehicles"] = veh
    out["delay_per_vehicle_s"] = ev["delay_total_vh"] * 3600 / max(veh, 1)
    b = ev["bottleneck_cell_ctm"]
    out["bottleneck_km_ctm"] = float(ch[b] / 1000) if b >= 0 else None
    out["bottleneck_km_rho"] = float(ch[ev["bottleneck_cell_rho"]] / 1000)
    out["segments"] = ev["segments"]
    if keep_maps:
        k = r["density"]
        out["density"] = k.round(1)
        out["record_dt_h"] = r["record_dt_h"]
        out["chainage_m"] = ch
        out["ramp_queue_total"] = r["ramp_queue"].sum(axis=1).round(1)
        out["congested_km_t"] = ((k > p.kc_vpkm).sum(axis=1) * p.dx_km).round(2)
        out["signal_cells"] = cw.meta["signal_cells"]
        out["lanes"] = cw.lanes
        out["active_share_cell"] = (((k[:, :-1] > p.kc_vpkm) & (k[:, 1:] <= p.kc_vpkm)).mean(axis=0)).round(3)
    return out


def main():
    cfg, out = load_cfg()
    inv = load_inventory(cfg)
    p = ctm_params(cfg)
    th = base_theta(cfg["demand"], cfg["control"], cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
    res = {"theta": th, "synthetic_data": inv.get("synthetic_data", False), "base": {}, "grid": [], "gap_filled": {},
           "fifo_bounds": {}}

    for d in ("WB", "EB"):
        ev = evaluate(inv, d, th, cfg, p, lambda dem: demand_matrix(cfg, dem, p))
        s = summarise(ev, p, keep_maps=True)
        # Single-bottleneck fluid queue: unconstrained arrivals at the rho-bottleneck.
        b = ev["bottleneck_cell_rho"]
        c = cfg["ctm"]
        T = int(round(c["horizon_h"] * 3600 / p.dt_s))
        prof = peak_profile(T, p.dt_h, c["peak_start_h"], c["peak_len_h"], c["ramp_h"],
                            cfg["demand"]["peak_factor_offpeak"])
        f = fluid_queue(prof * ev["analytic"]["lam"][b], ev["cw"].cap_vph[b], p.dt_h)
        s["fluid_bottleneck_delay_vh"] = f["total_delay_vh"]
        s["fluid_max_queue_veh"] = f["max_queue_veh"]
        res["base"][d] = s
        print(f"[peak] {d}: delay {s['delay_total_vh']:.0f} veh h ({s['delay_per_vehicle_s']:.0f} s/veh), "
              f"queue up to {s['max_congested_km']:.1f} km, head at {s['bottleneck_km_ctm']} km "
              f"(rho says {s['bottleneck_km_rho']:.2f}); single-bottleneck fluid {f['total_delay_vh']:.0f} veh h")

        evg = evaluate(inv, d, th, cfg, p, lambda dem: demand_matrix(cfg, dem, p), mods={"fill_lateral_gaps": True})
        res["gap_filled"][d] = summarise(evg, p)
        for phi in (cfg["ctm"]["fifo"][0], cfg["ctm"]["fifo"][2]):
            evf = evaluate(inv, d, dict(th, fifo=phi), cfg, p, lambda dem: demand_matrix(cfg, dem, p))
            sf = summarise(evf, p)
            sf.pop("segments")
            res.setdefault("fifo_bounds", {}).setdefault(d, {})[str(phi)] = sf

        for scale in cfg["peak"]["demand_scale_grid"]:
            for L in cfg["peak"]["peak_len_grid_h"]:
                ev = evaluate(inv, d, th, cfg, p, lambda dem: demand_matrix(cfg, dem, p, scale, L))
                g = summarise(ev, p)
                g.pop("segments")
                res["grid"].append({"dir": d, "scale": scale, "peak_len_h": L, **g})
    # Same base run with every field count removed (all demand assumed).
    if inv.get("field_counts"):
        import copy
        inv_nc = copy.deepcopy(inv)
        for a in inv_nc["approaches"]:
            a.pop("observed_vph", None)
            a.pop("free_lanes", None)
        res["no_counts"] = {}
        for d in ("WB", "EB"):
            s = summarise(evaluate(inv_nc, d, th, cfg, p, lambda dem: demand_matrix(cfg, dem, p)), p)
            s.pop("segments")
            res["no_counts"][d] = s
        from src.data.network import approach_flow
        for fc in inv["field_counts"]:
            mem = {tuple(x) for x in fc["members"]}
            fc["assumed_vph"] = sum(approach_flow(a, th) for a in inv_nc["approaches"]
                                    if (a["way"], a["node"]) in mem)
        res["field_counts"] = inv["field_counts"]
    dump(out, "peak.json", res)


if __name__ == "__main__":
    main()
