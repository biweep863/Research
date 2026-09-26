"""Experiment 2 -- the independent theoretical cross-check (doc/00_overview.md):
for each K, measure the landscape-level selection coefficient s(K) from the
distribution of single-mutant fitness effects, and Kimura's implied
critical population size N_c^Kimura(K) = 1/s(K). Also measures the
temporal effective population size at a fixed N to confirm the GA behaves
like a finite Wright-Fisher population (Ne of the right order of magnitude,
not wildly larger from e.g. undetected selection sweeps).

Writes results/effective_size.json.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import dump, load_cfg, project_root, resolve_device
from ..analysis.effective_size import critical_n_kimura, dfe_selection_coefficient, temporal_ne
from ..ga.simple_ga import run_ga
from ..landscape.nk import NKLandscape


def main():
    cfg, out_dir = load_cfg()
    sc = cfg["sweep"]
    es = cfg["effective_size"]
    seed = cfg.get("seed", 0)
    device = resolve_device(cfg)
    L = sc["L"]
    K_grid = sc["k_grid"]
    if cfg.get("_limit"):
        K_grid = K_grid[: cfg["_limit"]]

    landscapes_per_k = sc.get("landscapes_per_k", 1)
    rows = []
    for K in K_grid:
        s_vals, ne_vals = [], []
        for li in range(landscapes_per_k):
            landscape = NKLandscape(L=L, K=K, seed=seed + K * 1000 + li, device=device)
            _, opt = landscape.brute_force_optimum()
            s_vals.append(dfe_selection_coefficient(landscape, n_samples=es["dfe_samples"],
                                                     seed=seed + K * 1000 + li, device=device))
            res = run_ga(landscape, N=es["ne_probe_n"], R=es["ne_replicates"],
                        generations=es["ne_generations"], p_c=sc["p_crossover"], p_m=sc["p_mutation"],
                        optimum_fitness=opt, success_tol=sc["success_tol"], seed=seed + K * 1000 + li,
                        device=device)
            ne = temporal_ne(res.freq_history)
            ne_finite = ne[torch.isfinite(ne)]
            if ne_finite.numel():
                ne_vals.append(float(ne_finite.median().item()))

        s = float(np.mean(s_vals))
        n_c_kimura = critical_n_kimura(s)
        ne_median = float(np.median(ne_vals)) if ne_vals else float("nan")
        rows.append({"K": int(K), "s": s, "n_c_kimura": n_c_kimura, "ne_probe_n": es["ne_probe_n"],
                    "ne_median": ne_median, "landscapes_per_k": landscapes_per_k})
        print(f"[effective_size] K={K}: s={s:.4f}  N_c(Kimura)={n_c_kimura:.1f}  "
              f"Ne_median(N={es['ne_probe_n']})={ne_median:.1f}")

    valid = [r for r in rows if r["s"] > 0 and np.isfinite(r["n_c_kimura"])]
    fit_summary = {}
    if len(valid) >= 3:
        Ks = np.array([r["K"] for r in valid], dtype=float)
        s_vals = np.array([r["s"] for r in valid])
        b, log_a = np.polyfit(Ks, np.log(s_vals), 1)
        fit_summary = {"s_log_linear_slope": float(b), "s_log_linear_intercept": float(log_a)}
        print(f"[effective_size] s(K) ~ exp({log_a:.3f} + {b:.3f}*K)")

    dump(out_dir, "effective_size.json", {"L": L, "rows": rows, "fit": fit_summary})


if __name__ == "__main__":
    main()
