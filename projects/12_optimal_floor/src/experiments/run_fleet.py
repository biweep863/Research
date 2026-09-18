"""Experiment 4 -- fleet size.  K cars with nearest-car dispatch and a
background load that scales with K.  Does a bigger fleet flatten the
floor curve?  Writes results/fleet.json.
"""
from __future__ import annotations

from .common import dump, load_cfg
from .run_traffic import sweep


def main():
    cfg, out = load_cfg()
    f = cfg["fleet"]; seed = cfg.get("seed", 0)
    blocks = []
    for K in f["cars_grid"]:
        rate = f["rate_per_car"] * K
        b = sweep(cfg, f["n_floors"], f["alpha"], f["parking"], rate, K, f["reps"], seed)
        for r in b["rows"]:
            del r["car_pos"]
        b.update({"n_cars": K, "rate": rate})
        blocks.append(b)
        print(f"[fleet] K={K} rate={rate:.1f}/min  h*={b['h_star']:2d} best={b['best']:5.1f}s "
              f"regret mid={b['regret_middle_pct']:5.1f}% worst={b['regret_worst_pct']:5.1f}% band={b['band_5pct']}")
    dump(out, "fleet.json", {"n_floors": f["n_floors"], "alpha": f["alpha"], "parking": f["parking"], "blocks": blocks})


if __name__ == "__main__":
    main()
