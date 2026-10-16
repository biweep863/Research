"""Experiment 5: does the lane ratio or the utilisation predict where the queue forms?

Monte Carlo over every demand/control assumption.  For each draw and direction
the CTM gives the ground truth per segment (share of time an active
bottleneck sits there, delay there).  Three predictors computed without
simulation are scored by Spearman correlation and by whether their top
segment is the CTM's principal bottleneck:

  lane_ratio, lane_ratio_cum   (the "150 lanes into 6" argument)
  feeder_cum                   (its numerator alone)
  inv_capacity                 (geometry and signals alone, no demand)
  rho                          (steady-state utilisation: demand / capacity)

Counterfactual "equal rates": every feeder lane carries the same flow, so
demand is proportional to lanes and the lane ratio gets its best chance.
Falsification: the hypothesis fails if the lane ratio does at least as well
as rho under the realistic, class-dependent rates.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.data.network import draw_theta
from src.experiments.common import ctm_params, demand_matrix, dump, load_cfg, load_inventory
from src.experiments.corridor_eval import evaluate
from utils.stats import bootstrap_ci

PREDICTORS = ("lane_ratio", "lane_ratio_cum", "feeder_cum", "inv_capacity", "rho")


def score(segs) -> dict | None:
    act = np.array([s["active_share"] for s in segs])
    dly = np.array([s["delay_vh"] for s in segs])
    if act.max() <= 0:
        return None
    out = {"principal": int(np.argmax(act))}
    for k in PREDICTORS:
        x = np.array([s[k] for s in segs])
        out[f"{k}_rho_active"] = float(spearmanr(x, act).statistic) if np.ptp(x) > 0 else 0.0
        out[f"{k}_rho_delay"] = float(spearmanr(x, dly).statistic) if np.ptp(x) > 0 and np.ptp(dly) > 0 else 0.0
        out[f"{k}_hit"] = bool(int(np.argmax(x)) == out["principal"])
    return out


def _auc(x, y) -> float:
    """P(score of a congested run > score of a free one), ties count half."""
    pos, neg = x[y], x[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def equalise(th: dict, inv: dict, d: str) -> dict:
    """Same total side demand, but every feeder lane at the same rate."""
    from src.data.network import approach_flow
    apps = [a for a in inv["approaches"] if a["dir"] in (d, "both")]
    total = sum(approach_flow(a, th) for a in apps)
    probe = dict(th, q_major=1.0, q_tertiary=1.0, q_minor=1.0, q_service=1.0)
    unit = sum(approach_flow(a, probe) for a in apps)
    q = total / unit
    return dict(th, q_major=q, q_tertiary=q, q_minor=q, q_service=q)


def main():
    cfg, out = load_cfg()
    inv = load_inventory(cfg)
    p = ctm_params(cfg)
    rng = np.random.default_rng(cfg["seed"] + 1)
    draws = cfg["montecarlo"]["draws"]
    dm = lambda dem: demand_matrix(cfg, dem, p)  # noqa: E731
    records = {"realistic": [], "equal_rates": [], "gap_filled": []}
    principal_km = {"WB": [], "EB": []}
    seg_bounds = {}
    for i in range(draws):
        th = draw_theta(cfg["demand"], cfg["control"], rng, cfg["ctm"]["fifo"], cfg["ctm"]["cap_drop"])
        for d in ("WB", "EB"):
            for scen in records:
                t = equalise(th, inv, d) if scen == "equal_rates" else th
                mods = {"fill_lateral_gaps": True} if scen == "gap_filled" else None
                ev = evaluate(inv, d, t, cfg, p, dm, mods=mods)
                sc = score(ev["segments"])
                rec = {"draw": i, "dir": d, "delay_total_vh": ev["delay_total_vh"],
                       "max_rho": ev["max_rho"], "max_congested_km": ev["max_congested_km"],
                       "theta": t, "scored": sc is not None}
                if sc:
                    rec.update(sc)
                    if scen == "realistic":
                        s = ev["segments"][sc["principal"]]
                        principal_km[d].append((s["c0"] + s["c1"]) / 2000)
                if scen == "realistic":
                    seg_bounds[d] = [(s["c0"], s["c1"]) for s in ev["segments"]]
                    rec["active_share"] = [s["active_share"] for s in ev["segments"]]
                records[scen].append(rec)
        if (i + 1) % 10 == 0:
            print(f"[montecarlo] {i + 1}/{draws} draws")

    summary = {}
    for scen, recs in records.items():
        sc = [r for r in recs if r["scored"]]
        s = {"n": len(recs), "n_congested": len(sc)}
        for k in PREDICTORS:
            for m in ("rho_active", "rho_delay"):
                v = [r[f"{k}_{m}"] for r in sc]
                mean, lo, hi = bootstrap_ci(v, seed=cfg["seed"]) if v else (np.nan,) * 3
                s[f"{k}_{m}"] = {"mean": mean, "lo": lo, "hi": hi}
            s[f"{k}_hit"] = float(np.mean([r[f"{k}_hit"] for r in sc])) if sc else np.nan
        # Does max rho >= 1 tell congested runs from free-flowing ones?
        y = np.array([r["scored"] for r in recs])
        x = np.array([r["max_rho"] for r in recs])
        s["rho_classifier_acc"] = float(((x >= 1) == y).mean())
        s["rho_auc"] = _auc(x, y)
        s["delay_vh"] = dict(zip(("mean", "lo", "hi"), bootstrap_ci([r["delay_total_vh"] for r in recs],
                                                                      seed=cfg["seed"])))
        s["delay_vh_median"] = float(np.median([r["delay_total_vh"] for r in recs]))
        summary[scen] = s

    # Share of draws in which each segment hosts the principal bottleneck.
    where = {}
    for d in ("WB", "EB"):
        recs = [r for r in records["realistic"] if r["dir"] == d and r["scored"]]
        counts = np.zeros(len(seg_bounds[d]))
        for r in recs:
            counts[r["principal"]] += 1
        where[d] = {"segments": seg_bounds[d], "share": (counts / max(len(recs), 1)).tolist(),
                    "mean_active": np.mean([r["active_share"] for r in records["realistic"] if r["dir"] == d],
                                           axis=0).tolist()}
    r, e = summary["realistic"], summary["equal_rates"]
    falsified = r["lane_ratio_rho_active"]["mean"] >= r["rho_rho_active"]["mean"]
    res = {"draws": draws, "summary": summary, "where": where, "hypothesis_falsified": bool(falsified),
           "records": records}
    dump(out, "montecarlo.json", res)
    print(f"[montecarlo] Spearman with CTM bottleneck share (realistic): lane ratio "
          f"{r['lane_ratio_rho_active']['mean']:.2f}, cumulative {r['lane_ratio_cum_rho_active']['mean']:.2f}, "
          f"rho {r['rho_rho_active']['mean']:.2f}; hit rates {r['lane_ratio_hit']:.0%} / "
          f"{r['lane_ratio_cum_hit']:.0%} / {r['rho_hit']:.0%} (feeders alone {r['feeder_cum_hit']:.0%}, "
          f"capacity alone {r['inv_capacity_hit']:.0%}); equal rates: lane ratio "
          f"{e['lane_ratio_rho_active']['mean']:.2f}, rho {e['rho_rho_active']['mean']:.2f}; "
          f"max rho>=1 classifies congestion with accuracy {r['rho_classifier_acc']:.0%} (AUC {r['rho_auc']:.2f}). "
          f"Falsified: {falsified}")


if __name__ == "__main__":
    main()
