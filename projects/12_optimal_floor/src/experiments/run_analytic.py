"""Experiment 1 -- closed form: one car, no traffic.

E[T(h)] for every home floor, over building height, lobby-trip fraction and
parking policy; the optimal floor, the regret of the middle and the worst
floor, and the 5 % indifference band.  Writes results/analytic.json.
"""
from __future__ import annotations

from .common import dump, load_cfg
from src.analytic.closed_form import curve
from src.model.kinematics import Kinematics


def main():
    cfg, out = load_cfg()
    a = cfg["analytic"]; kin = Kinematics.from_cfg(cfg["kinematics"])
    rows, curves = [], []
    for N in a["n_floors_grid"]:
        for alpha in a["alpha_grid"]:
            for pol in a["parking"]:
                c = curve(N, alpha, kin, pol)
                row = {"n_floors": N, "alpha": alpha, "parking": pol,
                       **{k: v for k, v in c.items() if k not in ("homes", "mean")}}
                rows.append(row)
                if N == a["curve_n_floors"]:
                    curves.append({"alpha": alpha, "parking": pol, "homes": c["homes"], "mean": c["mean"].tolist()})
                print(f"[analytic] N={N:3d} alpha={alpha:.2f} {pol:6s} h*={c['h_star']:3d}/{N} "
                      f"best={c['best']:5.1f}s  regret mid={c['regret_middle_pct']:5.1f}% worst={c['regret_worst_pct']:5.1f}% "
                      f"band={c['band_5pct']}")
    dump(out, "analytic.json", {"kinematics": cfg["kinematics"], "rows": rows, "curves": curves,
                                "curve_n_floors": a["curve_n_floors"]})


if __name__ == "__main__":
    main()
