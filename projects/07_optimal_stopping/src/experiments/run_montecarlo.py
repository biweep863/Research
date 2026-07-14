"""Monte-Carlo + closed-form study of the stopping family.

Produces results/montecarlo.json with:
  * secretary success-vs-skip-fraction curves (MC) and their analytic optima,
  * the parking optimum (exact DP) vs. a naive "37% rule" transplanted onto
    parking, across occupancy p,
  * secretary net-payoff optima under a per-observation cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import yaml

from utils.seeds import set_seed
from src.analysis.analytic import (
    ONE_OVER_E,
    parking_best_stationary_threshold,
    parking_fixed_threshold_cost,
    parking_optimal,
    secretary_cost_optimal_threshold,
    secretary_optimal_threshold,
    secretary_success_prob,
    symmetric_street,
)
from src.envs.secretary import threshold_success_rate


def secretary_curves(cfg, rng):
    out = []
    for n in cfg["secretary"]["n_values"]:
        analytic = secretary_optimal_threshold(n)
        # Monte-Carlo success rate across a grid of skip fractions
        rs = np.unique(np.linspace(0, n - 1, min(n, 40)).astype(int))
        mc = [threshold_success_rate(n, int(r), cfg["secretary"]["mc_episodes"], rng)
              for r in rs]
        out.append({
            "n": n,
            "r_star": analytic["r_star"],
            "x_star": analytic["x_star"],
            "p_star": analytic["p_star"],
            "grid_r": rs.tolist(),
            "grid_x": (rs / n).tolist(),
            "mc_success": mc,
            "analytic_success": [secretary_success_prob(int(r), n) for r in rs],
        })
    return out


def parking_study(cfg):
    k = cfg["parking"]["street_half_length"]
    fail = cfg["parking"]["fail_cost"]
    dists = symmetric_street(k)
    out = []
    for p in cfg["parking"]["occupancy_p"]:
        opt = parking_optimal(dists, p, fail)
        best_stat = parking_best_stationary_threshold(dists, p, fail)
        # Naive "37% rule" transplanted: skip the first 37% of spots, then take
        # the first free one (theta = accept any distance after the cutoff).
        cutoff = int(round(ONE_OVER_E * len(dists)))
        naive_cost = _naive_parking_cost(dists, p, fail, cutoff)
        out.append({
            "p": p,
            "optimal_cost": opt["expected_cost"],
            "optimal_stationary_theta": best_stat["theta_star"],
            "optimal_stationary_cost": best_stat["expected_cost"],
            "naive37_cost": naive_cost,
            "naive37_regret": (naive_cost - opt["expected_cost"]) / abs(opt["expected_cost"]),
        })
    return {"street_half_length": k, "fail_cost": fail, "by_p": out}


def _naive_parking_cost(dists, p, fail, cutoff):
    """Skip first `cutoff` spots, then take the first free spot regardless of
    distance. Evaluated exactly by recursion over the tail."""
    c = np.asarray(dists, dtype=float)
    T = len(c)
    V = fail
    for t in range(T - 1, -1, -1):
        if t < cutoff:
            V = V                       # forced skip in the reject window
        else:
            V = p * c[t] + (1 - p) * V  # take first free spot after cutoff
    return float(V)


def secretary_cost_study(cfg):
    out = []
    n = cfg["secretary"].get("cost_n", 100)
    for cost in cfg["secretary"]["costs"]:
        res = secretary_cost_optimal_threshold(n, cost)
        out.append({
            "n": n, "cost": cost,
            "r_star": res["r_star"], "x_star": res["x_star"],
            "net_payoff": res["net_payoff"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))
    rng = np.random.default_rng(cfg.get("seed", 0))

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "one_over_e": ONE_OVER_E,
        "secretary_curves": secretary_curves(cfg, rng),
        "parking": parking_study(cfg),
        "secretary_cost": secretary_cost_study(cfg),
    }
    path = out_dir / "montecarlo.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[montecarlo] wrote {path}")


if __name__ == "__main__":
    main()
