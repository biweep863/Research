"""Train model-free agents and compare learned thresholds to the analytic optima.

Produces results/agents.json:
  * secretary: Q-learning and REINFORCE learned skip-fractions across seeds,
    vs. the 1/e optimum, with regret in success probability;
  * parking: Q-learning and REINFORCE learned distance thresholds across seeds
    and occupancy p, vs. the exact-DP optimum, with regret in expected cost.
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
from utils.stats import bootstrap_ci
from src.analysis.analytic import (
    parking_optimal,
    parking_policy_cost,
    secretary_mask_success,
    secretary_optimal_threshold,
    symmetric_street,
)
from src.analysis.policies import (
    q_learn_parking,
    q_learn_secretary,
    reinforce_parking,
    reinforce_secretary,
    relative_regret,
)
from src.envs.secretary import threshold_success_rate


def _summ(xs):
    mean, lo, hi = bootstrap_ci(xs)
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "values": list(map(float, xs))}


def secretary_agents(cfg, rng):
    out = []
    n_values = cfg["secretary"].get("agent_n_values", cfg["secretary"]["n_values"])
    for n in n_values:
        opt = secretary_optimal_threshold(n)
        q_x, re_x, q_reg, re_reg = [], [], [], []
        for s in cfg["agents"]["seeds"]:
            r_rng = np.random.default_rng(1000 * s + n)
            q = q_learn_secretary(
                n, cfg["agents"]["q_episodes"], r_rng,
                alpha=cfg["agents"]["alpha"], eps=cfg["agents"]["epsilon"])
            re = reinforce_secretary(
                n, cfg["agents"]["reinforce_episodes"], r_rng)
            q_x.append(q["x_threshold"])
            re_x.append(re["x_threshold"])
            # exact success of each learned greedy policy vs analytic optimum
            q_succ = secretary_mask_success(q["take_mask"], n)
            re_mask = np.arange(1, n + 1) / n > re["tau"]   # REINFORCE cutoff
            re_succ = secretary_mask_success(re_mask, n)
            q_reg.append(relative_regret(q_succ, opt["p_star"]))
            re_reg.append(relative_regret(re_succ, opt["p_star"]))
        out.append({
            "n": n,
            "x_star": opt["x_star"], "p_star": opt["p_star"],
            "qlearn_x": _summ(q_x), "reinforce_x": _summ(re_x),
            "qlearn_regret": _summ(q_reg), "reinforce_regret": _summ(re_reg),
        })
    return out


def parking_agents(cfg, rng):
    k = cfg["parking"]["street_half_length"]
    fail = cfg["parking"]["fail_cost"]
    dists = symmetric_street(k)
    out = []
    for p in cfg["parking"]["occupancy_p"]:
        opt = parking_optimal(dists, p, fail)
        q_cost, re_theta, q_reg, re_reg = [], [], [], []
        re_cost = []
        for s in cfg["agents"]["seeds"]:
            r_rng = np.random.default_rng(2000 * s + int(p * 100))
            q = q_learn_parking(
                dists, p, fail, cfg["agents"]["q_episodes"], r_rng,
                alpha=cfg["agents"]["alpha"], eps=cfg["agents"]["epsilon"])
            re = reinforce_parking(
                dists, p, fail, cfg["agents"]["reinforce_episodes"], r_rng)
            # score each learned policy's exact expected cost
            qc = parking_policy_cost(dists, p, fail, q["accept_mask"])
            re_mask = dists <= re["theta_distance"]
            rc = parking_policy_cost(dists, p, fail, re_mask)
            q_cost.append(qc)
            re_cost.append(rc)
            re_theta.append(re["theta_distance"])
            q_reg.append(relative_regret(qc, opt["expected_cost"]))
            re_reg.append(relative_regret(rc, opt["expected_cost"]))
        out.append({
            "p": p,
            "optimal_cost": opt["expected_cost"],
            "qlearn_cost": _summ(q_cost), "reinforce_cost": _summ(re_cost),
            "reinforce_theta": _summ(re_theta),
            "qlearn_regret": _summ(q_reg), "reinforce_regret": _summ(re_reg),
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
        "secretary": secretary_agents(cfg, rng),
        "parking": parking_agents(cfg, rng),
    }
    path = out_dir / "agents.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[agents] wrote {path}")


if __name__ == "__main__":
    main()
