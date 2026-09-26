"""Experiment 1 -- the primary pre-registered test (doc/05_decisions.md):
for each landscape ruggedness K, sweep population size N, measure the GA's
success probability within a fixed generation budget, and fit the
population size N_c(K) at which success probability crosses 0.5. Then test
whether N_c(K) scales log-linearly (log N_c = a + b*K) or linearly
(N_c = a + b*K) with K, comparing the two fits by AIC.

Writes results/threshold_sweep.json.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .common import dump, load_cfg, project_root, resolve_device
from ..analysis.effective_size import fit_threshold
from ..ga.simple_ga import run_ga
from ..landscape.nk import NKLandscape


def _aic(y, yhat, n_params):
    resid = y - yhat
    n = len(y)
    rss = float(np.sum(resid ** 2))
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2 * n_params


def main():
    cfg, out_dir = load_cfg()
    sc = cfg["sweep"]
    seed = cfg.get("seed", 0)
    device = resolve_device(cfg)
    L, R = sc["L"], sc["replicates"]
    N_grid = np.array(sc["n_grid"])
    K_grid = sc["k_grid"]
    if cfg.get("_limit"):
        K_grid = K_grid[: cfg["_limit"]]

    landscapes_per_k = sc.get("landscapes_per_k", 1)
    rows = []
    for K in K_grid:
        t0 = time.time()
        # Average over several independent landscape draws per K: a single
        # random NK landscape's N_c is dominated by instance-to-instance
        # ruggedness variance at fixed K, not the K-dependence we're testing
        # (see doc/05_decisions.md) -- averaging isolates the latter.
        p_success_per_landscape = []
        for li in range(landscapes_per_k):
            landscape = NKLandscape(L=L, K=K, seed=seed + K * 1000 + li, device=device)
            _, opt = landscape.brute_force_optimum()
            p_success = []
            for N in N_grid:
                res = run_ga(landscape, N=int(N), R=R, generations=sc["generations"],
                            p_c=sc["p_crossover"], p_m=sc["p_mutation"], optimum_fitness=opt,
                            success_tol=sc["success_tol"], seed=seed + K * 1000 + li, device=device)
                p_success.append(float((res.gen_to_success >= 0).float().mean().item()))
            p_success_per_landscape.append(p_success)
        p_success = np.array(p_success_per_landscape).mean(axis=0)
        n_c, ci_lo, ci_hi = fit_threshold(N_grid, p_success, n_reps=R * landscapes_per_k, seed=seed + K)
        rows.append({"K": int(K), "n_grid": N_grid.tolist(), "landscapes_per_k": landscapes_per_k,
                    "p_success": p_success.tolist(), "n_c": n_c, "n_c_ci": [ci_lo, ci_hi],
                    "time_s": time.time() - t0})
        print(f"[threshold] K={K}: N_c={n_c:.1f} (CI {ci_lo:.1f}-{ci_hi:.1f})  "
              f"p_success={np.round(p_success, 2).tolist()}  ({rows[-1]['time_s']:.1f}s)")

    valid = [r for r in rows if np.isfinite(r["n_c"])]
    fit_summary = {}
    if len(valid) >= 3:
        Ks = np.array([r["K"] for r in valid], dtype=float)
        Ncs = np.array([r["n_c"] for r in valid])
        b_lin, a_lin = np.polyfit(Ks, Ncs, 1)
        aic_lin = _aic(Ncs, a_lin + b_lin * Ks, 2)
        b_log, a_log = np.polyfit(Ks, np.log(Ncs), 1)
        aic_log = _aic(Ncs, np.exp(a_log + b_log * Ks), 2)
        fit_summary = {"linear": {"slope": float(b_lin), "intercept": float(a_lin), "aic": float(aic_lin)},
                       "log_linear": {"slope": float(b_log), "intercept": float(a_log), "aic": float(aic_log)},
                       "log_linear_preferred": bool(aic_log < aic_lin)}
        print(f"[threshold] linear AIC={aic_lin:.2f}  log-linear AIC={aic_log:.2f}  "
              f"preferred={'log-linear' if fit_summary['log_linear_preferred'] else 'linear'}")

    dump(out_dir, "threshold_sweep.json", {"L": L, "replicates": R, "rows": rows, "fit": fit_summary})


if __name__ == "__main__":
    main()
