"""Experiment 2 -- background traffic, one car (discrete-event simulation).

For each background rate and parking policy, sweep the home floor and record
mean, p90, wait, ride and intermediate stops of the resident's trips; the
car's position when the resident calls (stationary distribution under `last`
parking); and the validation of the simulator against the closed form at zero
traffic.  Writes results/traffic.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.analytic.closed_form import expected_trip_time, regret_stats
from src.model.kinematics import Kinematics
from src.sim.evaluate import simulate_resident


def sweep(cfg, N, alpha, parking, rate, n_cars, reps, seed, stairs=None, horizon=None, interval=None):
    """Sweep the home floor.  Common random numbers: replication r uses the same
    seed for every home floor, so the background traffic is identical across
    floors and the curves compare floors rather than traffic draws."""
    s = cfg["sim"]; kin = Kinematics.from_cfg(cfg["kinematics"])
    homes = list(range(2, N + 1))
    rows = []
    for h in homes:
        runs = [simulate_resident(h, N, alpha, kin, parking, rate, n_cars,
                                  horizon or s["horizon_s"], interval or s["trip_interval_s"],
                                  np.random.default_rng(seed + r), stairs=stairs,
                                  capacity=s["capacity"]) for r in range(reps)]
        tot = np.concatenate([r["totals"] for r in runs])
        rows.append({"home": h, "mean": float(tot.mean()), "p90": float(np.quantile(tot, 0.9)),
                     "sem": float(tot.std() / np.sqrt(len(tot))),
                     "mean_wait": float(np.mean([r["mean_wait"] for r in runs])),
                     "mean_ride": float(np.mean([r["mean_ride"] for r in runs])),
                     "mean_stops": float(np.mean([r["mean_stops"] for r in runs])),
                     "frac_walked": float(np.mean([r["frac_walked"] for r in runs])),
                     "n_trips": int(len(tot)),
                     "car_pos": [p for r in runs for pos in r["car_pos_at_call"] for p in pos]})
    means = [r["mean"] for r in rows]; p90s = [r["p90"] for r in rows]
    return {"homes": homes, "rows": rows, **regret_stats(homes, means, N),
            "h_star_p90": int(homes[int(np.argmin(p90s))])}


def main():
    cfg, out = load_cfg()
    t = cfg["traffic"]; seed = cfg.get("seed", 0); kin = Kinematics.from_cfg(cfg["kinematics"])
    N, alpha = t["n_floors"], t["alpha"]
    blocks = []
    for parking in t["parking_grid"]:
        for rate in t["rate_grid"]:
            b = sweep(cfg, N, alpha, parking, rate, t["n_cars"], t["reps"], seed)
            b.update({"parking": parking, "rate": rate})
            if rate == 0.0:
                ana = [expected_trip_time(h, N, alpha, kin, parking)["mean"] for h in b["homes"]]
                b["analytic_mean"] = ana
                b["validation_rel_err"] = float(np.mean(np.abs(np.array([r["mean"] for r in b["rows"]]) - ana) / np.array(ana)))
            # Stationary car position at call, pooled over homes.
            pos = np.concatenate([r["car_pos"] for r in b["rows"]]) if b["rows"][0]["car_pos"] else np.array([])
            b["car_pos_hist"] = np.bincount(pos.astype(int), minlength=N + 1)[1:].tolist() if len(pos) else []
            for r in b["rows"]:
                del r["car_pos"]
            blocks.append(b)
            print(f"[traffic] {parking:6s} rate={rate:3.1f}/min  h*={b['h_star']:2d} (p90 h*={b['h_star_p90']:2d}) "
                  f"best={b['best']:5.1f}s regret mid={b['regret_middle_pct']:5.1f}% worst={b['regret_worst_pct']:5.1f}%"
                  + (f"  valid err={b['validation_rel_err']*100:.1f}%" if rate == 0 else ""))
    dump(out, "traffic.json", {"n_floors": N, "alpha": alpha, "n_cars": t["n_cars"], "reps": t["reps"],
                               "horizon_s": cfg["sim"]["horizon_s"], "blocks": blocks})


if __name__ == "__main__":
    main()
