"""Experiment 2: steady-state tandem model at the base parameters.

Per cell: arriving flow, capacity, utilisation rho; per segment: the three
predictors; per ramp: utilisation.  Also the corridor-level arithmetic behind
"130-160 lanes into 5-6": what the feeders would deliver at saturation versus
what the demand model says they deliver.
"""
from __future__ import annotations

import numpy as np

from src.experiments.common import dump, load_cfg, load_inventory
from src.experiments.corridor_eval import analytic
from src.data.network import base_theta, build_direction, segments


def main():
    cfg, out = load_cfg()
    inv = load_inventory(cfg)
    th = base_theta(cfg["demand"], cfg["control"], cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
    s0 = cfg["control"]["s0"]
    res = {"theta": th, "synthetic_data": inv.get("synthetic_data", False), "dirs": {}}
    for d in ("WB", "EB"):
        net = build_direction(inv, d, th, cfg["control"], cfg["corridor"])
        cw, dem = net["cw"], net["ramp_demand_vph"]
        an = analytic(cw, dem)
        segs = []
        feeder = np.zeros(len(cw.lanes))
        np.add.at(feeder, cw.ramp_cell[1:], cw.ramp_lanes[1:])
        for s in segments(inv, cw):
            c = s["cells"]
            segs.append({"c0": s["c0"], "c1": s["c1"], "lanes": float(cw.lanes[c].mean()),
                         "feeder_lanes": float(feeder[c].sum()),
                         "lane_ratio": float(feeder[c].sum() / cw.lanes[c].mean()),
                         "rho": float(an["rho"][c].max()), "flow_max": float(an["lam"][c].max())})
        side = dem[1:]
        side_lanes = cw.ramp_lanes[1:]
        by_class = {}
        for m, q, ln in zip(cw.meta["ramps"][1:], side, side_lanes):
            b = by_class.setdefault(m["dclass"], {"lanes": 0.0, "vph": 0.0, "n": 0})
            b["lanes"] += float(ln); b["vph"] += float(q); b["n"] += 1
        order = np.argsort(-an["ramp_rho"])[:8]
        res["dirs"][d] = {
            "chainage_m": cw.meta["chainage_m"], "lanes": cw.lanes, "cap_vph": cw.cap_vph,
            "lam": an["lam"], "rho": an["rho"], "beta": cw.beta,
            "signal_cells": cw.meta["signal_cells"],
            "bottleneck_cell": an["bottleneck"],
            "bottleneck_chainage_m": float(cw.meta["chainage_m"][an["bottleneck"]]),
            "max_rho": float(an["rho"].max()), "n_unstable_cells": an["n_unstable"],
            "segments": segs,
            "side_demand_vph": float(side.sum()), "side_lanes": float(side_lanes.sum()),
            "side_at_saturation_vph": float(side_lanes.sum() * s0),
            "through_vph": float(dem[0]),
            "by_class": by_class,
            "mean_vph_per_feeder_lane": float(side.sum() / side_lanes.sum()),
            "top_ramps": [{"name": cw.meta["ramps"][i]["name"] or cw.meta["ramps"][i].get("highway", ""),
                           "rho": float(an["ramp_rho"][i]), "chainage_m": cw.meta["ramps"][i].get("chainage")}
                          for i in order],
            "peak_flow_max_vph": float(an["lam"].max()),
            "capacity_median_vph": float(np.median(cw.cap_vph)),
        }
        r = res["dirs"][d]
        print(f"[analytic] {d}: max rho {r['max_rho']:.2f} at {r['bottleneck_chainage_m'] / 1000:.2f} km, "
              f"{r['n_unstable_cells']} unstable cells, side demand {r['side_demand_vph']:.0f} veh/h from "
              f"{r['side_lanes']:.0f} lanes ({r['mean_vph_per_feeder_lane']:.0f} veh/h/lane vs {s0} at saturation)")
    dump(out, "analytic.json", res)


if __name__ == "__main__":
    main()
