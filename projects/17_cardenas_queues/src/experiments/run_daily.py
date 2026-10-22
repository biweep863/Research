"""Experiment 7: a working day on the corridor, 7:00-20:00.

Each direction gets an hourly demand multiplier (config `daily`): at 7:00 the
flow is tidal, south -> north (towards San Pedro, WB); the evening reverses.
For every hour: demand arriving at the bottleneck versus its maximum
capacity, and from the CTM the queue length and delay over the day, when the
queue starts and when it clears.  A Monte Carlo over the demand assumptions
gives the spread of congested hours.
"""
from __future__ import annotations

import numpy as np

from src.data.network import base_theta, build_direction, draw_theta
from src.experiments.common import ctm_params, dump, load_cfg, load_inventory
from src.experiments.corridor_eval import analytic, evaluate
from utils.stats import bootstrap_ci


def day_profile(cfg: dict, d: str, p) -> np.ndarray:
    dc = cfg["daily"]
    hrs = np.asarray(dc[d], float)
    T = int(round(dc["hours"] * 3600 / p.dt_s))
    t = (np.arange(T) + 0.5) * p.dt_h                       # hours since start_h
    return np.interp(t, np.arange(len(hrs)) + 0.5, hrs)      # hourly values at mid-hour


def run_day(inv, d, th, cfg, p) -> dict:
    prof = day_profile(cfg, d, p)
    ev = evaluate(inv, d, th, cfg, p, lambda dem: prof[:, None] * dem[None, :])
    cw, dem = ev["cw"], ev["dem"]
    k = ev["ctm"]["density"]
    rec_h = ev["ctm"]["record_dt_h"]
    cong_km = (k > p.kc_vpkm).sum(axis=1) * p.dx_km
    hours = cfg["daily"]["hours"]
    per_h = int(round(1 / rec_h))
    hourly_q = [float(cong_km[h * per_h:(h + 1) * per_h].max()) for h in range(hours)]
    # Utilisation at the worst cell, hour by hour (steady state at that hour's demand).
    rho_h, lam_h = [], []
    b = ev["bottleneck_cell_rho"]
    for mult in cfg["daily"][d]:
        an = analytic(cw, dem * mult)
        rho_h.append(float(an["rho"].max()))
        lam_h.append(float(an["lam"][b]))
    cong = np.nonzero(cong_km > 0.2)[0]
    start = cfg["daily"]["start_h"]
    return {"rho_h": rho_h, "lam_bottleneck_h": lam_h, "cap_bottleneck": float(cw.cap_vph[b]),
            "bottleneck_km": float(cw.meta["chainage_m"][b] / 1000),
            "queue_km_h": hourly_q, "queue_km_t": cong_km.round(2), "record_dt_h": rec_h,
            "delay_vh": ev["delay_total_vh"], "vehicles": float(ev["ctm"]["total_in"]),
            "delay_per_vehicle_s": ev["delay_total_vh"] * 3600 / max(float(ev["ctm"]["total_in"]), 1),
            "queue_start_h": float(start + cong[0] * rec_h) if cong.size else None,
            "queue_end_h": float(start + cong[-1] * rec_h) if cong.size else None,
            "congested_h": float(cong.size * rec_h), "max_queue_km": float(cong_km.max())}


def main():
    cfg, out = load_cfg()
    inv = load_inventory(cfg)
    p = ctm_params(cfg)
    th = base_theta(cfg["demand"], cfg["control"], cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
    res = {"start_h": cfg["daily"]["start_h"], "hours": cfg["daily"]["hours"], "draws": cfg["daily"]["draws"],
           "profile": {d: cfg["daily"][d] for d in ("WB", "EB")}, "base": {}, "mc": {}}
    for d in ("WB", "EB"):
        r = run_day(inv, d, th, cfg, p)
        res["base"][d] = r
        print(f"[daily] {d}: max rho {max(r['rho_h']):.2f}, queue {r['max_queue_km']:.1f} km, "
              f"congested {r['congested_h']:.1f} h ({r['queue_start_h']} -> {r['queue_end_h']}), "
              f"delay {r['delay_vh']:.0f} veh h ({r['delay_per_vehicle_s']:.0f} s/veh); "
              f"bottleneck capacity {r['cap_bottleneck']:.0f} veh/h at {r['bottleneck_km']:.1f} km")
    rng = np.random.default_rng(cfg["seed"] + 3)
    rows = {"WB": [], "EB": []}
    for _ in range(cfg["daily"]["draws"]):
        t = draw_theta(cfg["demand"], cfg["control"], rng, cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
        for d in ("WB", "EB"):
            r = run_day(inv, d, t, cfg, p)
            rows[d].append({"congested_h": r["congested_h"], "delay_vh": r["delay_vh"],
                            "max_queue_km": r["max_queue_km"], "queue_km_h": r["queue_km_h"]})
    for d in ("WB", "EB"):
        ch = [x["congested_h"] for x in rows[d]]
        m, lo, hi = bootstrap_ci(ch, seed=cfg["seed"])
        q = np.array([x["queue_km_h"] for x in rows[d]])
        res["mc"][d] = {"congested_h": {"mean": m, "lo": lo, "hi": hi, "median": float(np.median(ch))},
                        "p_any_queue": float(np.mean(np.array(ch) > 0)),
                        "p_queue_by_hour": (q > 0.2).mean(axis=0).tolist(),
                        "delay_median_vh": float(np.median([x["delay_vh"] for x in rows[d]]))}
        print(f"[daily] MC {d}: P(queue) {res['mc'][d]['p_any_queue']:.0%}, congested hours median "
              f"{res['mc'][d]['congested_h']['median']:.1f}")
    dump(out, "daily.json", res)


if __name__ == "__main__":
    main()
