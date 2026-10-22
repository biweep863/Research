"""Experiment 6: what would actually shorten the queue?

For the base parameters and for Monte Carlo draws in which a direction is
congested, four interventions are applied around the CTM's principal
bottleneck b (cell index in travel order):

  metering     ALINEA on the k largest feeders within 1.5 km upstream of b
  consolidate  close minor / service accesses; their demand re-enters at the
               next remaining access downstream (fewer merges, less friction)
  extra_lane   +1 lane over `extra_lane_len_m` ending at b
  retime       +delta g/C for the corridor at signals within [b-1.5 km, b+0.5 km]
               (side streets lose the same green)

Outcome: total delay (mainline + every queue on the feeders), and its split.
"""
from __future__ import annotations

import numpy as np

from src.data.network import base_theta, draw_theta
from src.experiments.common import ctm_params, demand_matrix, dump, load_cfg, load_inventory
from src.experiments.corridor_eval import evaluate
from src.sim.ctm import Metering
from utils.stats import bootstrap_ci

NAMES = ("metering", "consolidate", "extra_lane", "retime")


def interventions(inv, d, th, cfg, p, dm, base) -> dict:
    icfg = cfg["interventions"]
    b = base["bottleneck_cell_ctm"]
    cw = base["cw"]
    out = {}

    def meter_fn(cw_):
        cand = [i for i in range(1, len(cw_.ramp_cell)) if b - 15 <= cw_.ramp_cell[i] <= b]
        top = sorted(cand, key=lambda i: -base["dem"][i])[: icfg["metering_top_k"]]
        return Metering(ramps=np.array(top, int))

    out["metering"] = evaluate(inv, d, th, cfg, p, dm, metering_fn=meter_fn)
    out["consolidate"] = evaluate(inv, d, th, cfg, p, dm, mods={"consolidate_minor": True})
    n = max(1, int(icfg["extra_lane_len_m"] // 100))
    cells = range(max(0, b - n + 1), min(len(cw.lanes), b + 2))
    out["extra_lane"] = evaluate(inv, d, th, cfg, p, dm, mods={"extra_lanes": {c: 1.0 for c in cells}})
    sig = [c for c in cw.meta["signal_cells"] if b - 15 <= c <= b + 5]
    g_new = {c: min(0.85, cw.meta["g"][c] + icfg["retime_delta_g"]) for c in sig}
    out["retime"] = evaluate(inv, d, th, cfg, p, dm, mods={"g_override": g_new})
    out["_n_retimed"] = len(sig)
    return out


def row(base, iv) -> dict:
    r = {"base_vh": base["delay_total_vh"], "base_main_vh": base["delay_main_vh"],
         "base_ramp_vh": base["delay_ramp_vh"], "n_retimed": iv["_n_retimed"]}
    for k in NAMES:
        r[f"{k}_vh"] = iv[k]["delay_total_vh"]
        r[f"{k}_main_vh"] = iv[k]["delay_main_vh"]
        r[f"{k}_ramp_vh"] = iv[k]["delay_ramp_vh"]
        r[f"{k}_change"] = (iv[k]["delay_total_vh"] - base["delay_total_vh"]) / base["delay_total_vh"]
    return r


def main():
    cfg, out = load_cfg()
    inv = load_inventory(cfg)
    p = ctm_params(cfg)
    dm = lambda dem: demand_matrix(cfg, dem, p)  # noqa: E731
    th0 = base_theta(cfg["demand"], cfg["control"], cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
    res = {"base": {}, "draws": []}
    for d in ("WB", "EB"):
        base = evaluate(inv, d, th0, cfg, p, dm)
        if base["bottleneck_cell_ctm"] < 0:
            continue
        r = row(base, interventions(inv, d, th0, cfg, p, dm, base))
        r["bottleneck_km"] = float(base["cw"].meta["chainage_m"][base["bottleneck_cell_ctm"]] / 1000)
        res["base"][d] = r
        print(f"[interventions] base {d}: " + ", ".join(f"{k} {r[f'{k}_change']:+.0%}" for k in NAMES))

    rng = np.random.default_rng(cfg["seed"] + 2)
    for i in range(cfg["interventions"]["draws"]):
        th = draw_theta(cfg["demand"], cfg["control"], rng, cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
        for d in ("WB", "EB"):
            base = evaluate(inv, d, th, cfg, p, dm)
            if base["bottleneck_cell_ctm"] < 0 or base["delay_total_vh"] < 1:
                continue
            r = row(base, interventions(inv, d, th, cfg, p, dm, base))
            r.update(draw=i, dir=d, fifo=th["fifo"], cap_drop=th["cap_drop"])
            res["draws"].append(r)
    summ = {}
    for k in NAMES:
        ch = [r[f"{k}_change"] for r in res["draws"]]
        if ch:
            m, lo, hi = bootstrap_ci(ch, seed=cfg["seed"])
            summ[k] = {"mean": m, "lo": lo, "hi": hi, "median": float(np.median(ch)),
                       "frac_better": float(np.mean(np.array(ch) < -0.01)),
                       "frac_worse": float(np.mean(np.array(ch) > 0.01))}
    res["summary"] = summ
    res["n_congested_draws"] = len(res["draws"])
    dump(out, "interventions.json", res)
    print(f"[interventions] {len(res['draws'])} congested draw-directions: " +
          ", ".join(f"{k} {v['median']:+.0%} (better in {v['frac_better']:.0%})" for k, v in summ.items()))


if __name__ == "__main__":
    main()
