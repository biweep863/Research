"""Experiment 3 -- the stairs, with an up/down asymmetry.

Same building and traffic, several stairs variants (no stairs; 15/10 s per
floor up/down with a 3-floor climb limit; a steeper 20/10; a symmetric 12/12;
unlimited climb; descend only).  Reports the optimal floor, the fraction of
trips walked, and the mean by home floor.  Writes results/stairs.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from .run_traffic import sweep
from src.analytic.closed_form import curve
from src.model.kinematics import Kinematics
from src.model.stairs import Stairs


def main():
    cfg, out = load_cfg()
    st = cfg["stairs"]; seed = cfg.get("seed", 0); kin = Kinematics.from_cfg(cfg["kinematics"])
    N, alpha = st["n_floors"], st["alpha"]
    variants = []
    for v in st["variants"]:
        stairs = Stairs.from_cfg(v)
        b = sweep(cfg, N, alpha, st["parking"], st["rate"], st["n_cars"], st["reps"], seed, stairs=stairs)
        ana = curve(N, alpha, kin, st["parking"], stairs=stairs)
        b.update({"name": v["name"], "t_up": v.get("t_up"), "t_down": v.get("t_down"),
                  "max_climb": v.get("max_climb"), "analytic_h_star": ana["h_star"],
                  "analytic_mean": ana["mean"].tolist(),
                  "frac_walked_overall": float(np.mean([r["frac_walked"] for r in b["rows"]]))})
        for r in b["rows"]:
            del r["car_pos"]
        variants.append(b)
        print(f"[stairs] {v['name']:18s} h*={b['h_star']:2d} (analytic {ana['h_star']:2d}) best={b['best']:5.1f}s "
              f"walked={100*b['frac_walked_overall']:4.1f}%  regret mid={b['regret_middle_pct']:5.1f}% worst={b['regret_worst_pct']:5.1f}%")
    dump(out, "stairs.json", {"n_floors": N, "alpha": alpha, "rate": st["rate"], "parking": st["parking"],
                              "variants": variants})


if __name__ == "__main__":
    main()
