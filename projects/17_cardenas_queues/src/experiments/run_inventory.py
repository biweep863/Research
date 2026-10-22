"""Experiment 1: what actually feeds Av. Lázaro Cárdenas (OSM inventory).

Counts approaches / accesses / feeder lanes by class and direction, the
corridor cross-section, signals and junctions, and sets them against the
numbers quoted in the LLM conversation that motivated the project.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.experiments.common import dump, load_cfg, load_inventory
from src.data.network import base_theta, build_direction, segments


def main():
    cfg, out = load_cfg()
    inv = load_inventory(cfg)
    A, D, acc = inv["approaches"], inv["departures"], inv["accesses"]

    def tally(rows, key):
        c = Counter(r[key] for r in rows)
        return dict(sorted(c.items()))

    # "Streets": everything except driveways/plazas and U-turns (reported apart).
    no_service = [a for a in A if a["dclass"] not in ("service", "retorno")]
    acc_no_service = [a for a in acc if a["dclass"] not in ("service", "retorno")]
    lanes = {}
    for d in ("WB", "EB"):
        e = np.asarray(inv["lanes"][d]["express"])
        lt = np.asarray(inv["lanes"][d]["lateral"])
        tot = e + lt
        lanes[d] = {"express_median": float(np.median(e[e > 0])), "lateral_median": float(np.median(lt[lt > 0])),
                    "total_median": float(np.median(tot[tot > 0])), "total_min": float(tot[tot > 0].min()),
                    "total_max": float(tot.max()), "lateral_missing_frac": float((lt == 0).mean())}

    th = base_theta(cfg["demand"], cfg["control"])
    seg_rows = {}
    sig_inferred = {}
    for d in ("WB", "EB"):
        cw = build_direction(inv, d, th, cfg["control"], cfg["corridor"])["cw"]
        sig_inferred[d] = len(cw.meta["signal_cells"])
        rows = []
        for s in segments(inv, cw):
            c = s["cells"]
            mine = [a for a in A if a["dir"] in (d, "both") and s["c0"] - 50 <= a["chainage"] <= s["c1"] + 50]
            rows.append({"c0": s["c0"], "c1": s["c1"], "lanes": float(cw.lanes[c].mean()),
                         "approaches": len(mine), "feeder_lanes": int(sum(a["lanes"] for a in mine)),
                         "by_class": tally(mine, "dclass")})
        seg_rows[d] = rows

    s0 = cfg["control"]["s0"]
    feeder_lanes_all = int(sum(a["lanes"] for a in A))
    feeder_lanes_ns = int(sum(a["lanes"] for a in no_service))
    mainline = np.mean([lanes[d]["total_median"] for d in lanes])
    res = {
        "synthetic_data": inv.get("synthetic_data", False),
        "osm_source": inv.get("osm_source"), "osm_fetched": inv.get("osm_fetched"),
        "field_counts_applied": inv.get("field_counts_applied", 0),
        "field_counts": inv.get("field_counts", []),
        "length_km": inv["length_km"],
        "corridor_ways": inv["corridor_ways"],
        "corridor_lanes_tag_coverage": inv["corridor_lanes_tag_coverage"],
        "internal_links": inv["internal_links"],
        "approaches": len(A), "approaches_no_service": len(no_service),
        "accesses": len(acc), "accesses_no_service": len(acc_no_service),
        "departures": len(D),
        "approaches_by_dir": tally(A, "dir"),
        "approaches_by_class": tally(A, "dclass"),
        "approaches_by_highway": tally(A, "highway"),
        "approaches_by_kind": tally(A, "kind"),
        "feeder_lanes": feeder_lanes_all, "feeder_lanes_no_service": feeder_lanes_ns,
        "feeder_lanes_by_class": {k: int(sum(a["lanes"] for a in A if a["dclass"] == k))
                                  for k in ("major", "tertiary", "minor", "service", "retorno")},
        "retornos": inv.get("retornos", 0),
        "feeder_lanes_by_dir": {d: int(sum(a["lanes"] for a in A if a["dir"] == d)) for d in ("WB", "EB", "both")},
        "approach_lanes_tagged_frac": float(np.mean([a["lanes_tagged"] for a in A])),
        "mainline": lanes,
        "signals_osm": len(inv["signals"]), "signals_model": sig_inferred,
        "junctions": [{"chainage_km": j["chainage"] / 1000, "name": j["name"], "n_major": j["n_major"]}
                      for j in inv["junctions"]],
        "segments": seg_rows,
        # The headline ratio and what it would mean if every feeder lane ran at saturation.
        "lane_ratio_all": feeder_lanes_all / (2 * mainline),
        "lane_ratio_no_service": feeder_lanes_ns / (2 * mainline),
        "saturation_feeder_vph": feeder_lanes_all * s0,
        "mainline_capacity_vph_per_dir": mainline * s0,
        "llm_claims": cfg["llm_claims"],
        "approaches_list": [{k: a[k] for k in ("name", "highway", "dclass", "lanes", "lanes_tagged",
                                               "chainage", "dir", "kind", "side", "signal")} for a in A],
    }
    dump(out, "inventory.json", res)
    print(f"[inventory] {res['length_km']:.1f} km, {len(A)} approaches ({len(no_service)} excl. service), "
          f"{len(acc)} accesses, {feeder_lanes_all} feeder lanes ({feeder_lanes_ns} excl. service), "
          f"mainline {lanes['WB']['total_median']:.0f}/{lanes['EB']['total_median']:.0f} lanes WB/EB")


if __name__ == "__main__":
    main()
