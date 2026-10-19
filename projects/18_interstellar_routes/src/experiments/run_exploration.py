"""Experiment 4 -- visiting every star nearby with a fleet of probes.

Reproduces the nearest-neighbour exploration rule of Bjork (2007) on the real
Gaia neighbourhood and measures how much 2-opt local search shortens it (the
improvement Cotta & Morales 2009 reported was ~10% on synthetic stars), with
the MST of each probe's star set as a lower bound. Each probe takes one sky
sector (k-means on directions) and flies an open path from the Sun.

Outputs results/exploration.json: per fleet size, total and makespan path
length for NN and 2-opt, gap to the MST bound, and coverage-vs-time curves.
"""
from __future__ import annotations

import numpy as np

from src.analysis.tours import path_length, probe_tours
from src.experiments.common import setup, write
from src.model.kinematics import C_PCMYR, phase_space


def coverage(tours, key, v_pcyr, t_grid_yr):
    """Fraction of all stars visited by time t (each probe flies its own path)."""
    times = []
    for tr in tours:
        o = tr[key]
        seg = tr["D"][o[:-1], o[1:]]
        times.append(np.cumsum(seg) / v_pcyr)
    times = np.sort(np.concatenate(times))
    return (np.searchsorted(times, t_grid_yr, side="right") / times.size).tolist()


def main():
    cfg, cat, rng = setup("exploration")
    xc = cfg["exploration"]
    pos, _ = phase_space(cat, rng)
    dist = np.linalg.norm(pos, axis=1)
    stars = pos[(dist > 0) & (dist <= xc["radius_pc"])]
    v_pcyr = xc["v_probe_c"] * C_PCMYR / 1e6

    fleets = []
    t_grid = None
    for k in xc["n_probes"]:
        tours = probe_tours(stars, k, np.random.default_rng(cfg.get("seed", 0) + k))
        nn = np.array([path_length(t["D"], t["nn"]) for t in tours])
        op = np.array([path_length(t["D"], t["opt"]) for t in tours])
        mst = np.array([t["mst"] for t in tours])
        if t_grid is None:
            t_grid = np.linspace(0, 1.05 * nn.max() / v_pcyr, 80)
        fleets.append({
            "n_probes": k,
            "nn_total_pc": float(nn.sum()), "opt_total_pc": float(op.sum()),
            "mst_total_pc": float(mst.sum()),
            "improvement": float(1 - op.sum() / nn.sum()),
            "nn_gap_to_bound": float(nn.sum() / mst.sum() - 1),
            "opt_gap_to_bound": float(op.sum() / mst.sum() - 1),
            "nn_makespan_yr": float(nn.max() / v_pcyr),
            "opt_makespan_yr": float(op.max() / v_pcyr),
            "coverage_nn": coverage(tours, "nn", v_pcyr, t_grid),
            "coverage_opt": coverage(tours, "opt", v_pcyr, t_grid),
        })
        print(f"[exploration] k={k}: NN {nn.sum():.0f} pc, 2-opt {op.sum():.0f} pc "
              f"({fleets[-1]['improvement']:.1%} shorter), MST bound {mst.sum():.0f} pc")

    write(cfg, "exploration", {
        "radius_pc": xc["radius_pc"], "n_stars": int(len(stars)),
        "v_probe_c": xc["v_probe_c"],
        "t_grid_yr": t_grid.tolist(),
        "fleets": fleets,
    })


if __name__ == "__main__":
    main()
