"""Experiment 5 -- the (alpha, traffic) map and three archetypes.

A grid of lobby-trip fraction x background rate with stairs on: optimal
floor (mean and p90), 5 % indifference band, regret of the middle-floor and
top-floor heuristics.  Then residential / office / hospital archetypes with
more replications.  Writes results/robust.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from .run_traffic import sweep
from src.model.stairs import Stairs


def main():
    cfg, out = load_cfg()
    rb = cfg["robust"]; seed = cfg.get("seed", 0)
    stairs = Stairs.from_cfg(rb["stairs"])
    N = rb["n_floors"]
    grid = []
    for alpha in rb["alpha_grid"]:
        for rate in rb["rate_grid"]:
            b = sweep(cfg, N, alpha, rb["parking"], rate, rb["n_cars"], rb["reps"], seed, stairs=stairs,
                      horizon=rb["horizon_s"], interval=rb["trip_interval_s"])
            means = [r["mean"] for r in b["rows"]]
            grid.append({"alpha": alpha, "rate": rate, "h_star": b["h_star"], "h_star_p90": b["h_star_p90"],
                         "h_star_frac": (b["h_star"] - 1) / (N - 1), "best": b["best"],
                         "band_5pct": b["band_5pct"], "band_5pct_width": b["band_5pct_width"],
                         "regret_middle_pct": b["regret_middle_pct"], "regret_top_pct": b["regret_top_pct"],
                         "regret_worst_pct": b["regret_worst_pct"], "regret_second_pct": float(100 * (means[0] / b["best"] - 1)),
                         "means": means})
            print(f"[robust] alpha={alpha:.2f} rate={rate:3.1f}  h*={b['h_star']:2d}/{N} p90 h*={b['h_star_p90']:2d} "
                  f"band={b['band_5pct']} regret mid={b['regret_middle_pct']:5.1f}% top={b['regret_top_pct']:5.1f}%")
    arche = {}
    for name, a in rb["archetypes"].items():
        b = sweep(cfg, a["n_floors"], a["alpha"], a["parking"], a["rate"], a["n_cars"], rb["archetype_reps"], seed,
                  stairs=stairs, horizon=rb["horizon_s"], interval=rb["trip_interval_s"])
        for r in b["rows"]:
            del r["car_pos"]
        arche[name] = {**a, **{k: v for k, v in b.items() if k != "homes"}, "homes": b["homes"]}
        print(f"[robust] {name:12s} h*={b['h_star']:2d}/{a['n_floors']} p90 h*={b['h_star_p90']:2d} best={b['best']:5.1f}s "
              f"band={b['band_5pct']} regret mid={b['regret_middle_pct']:5.1f}% top={b['regret_top_pct']:5.1f}% worst={b['regret_worst_pct']:5.1f}%")
    dump(out, "robust.json", {"n_floors": N, "n_cars": rb["n_cars"], "parking": rb["parking"], "stairs": rb["stairs"],
                              "grid": grid, "archetypes": arche})


if __name__ == "__main__":
    main()
