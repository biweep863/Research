"""Experiment 5 -- where was the contact?  Foot INS + ZUPT + UWB anchors.

For each IMU grade and UWB fix rate (0 = inertial only), run the foot through
a 3-D running trajectory with a lateral cut, estimate its position with the
ZUPT-aided Kalman INS, and record the position error over the second half of
the run (a contact can happen at any instant of it).  From the horizontal error the probability of
placing a contact on the wrong side of a line (the penalty-area call) follows
as a Gaussian tail for each contact-to-line distance.  Writes
results/localize.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.localize.ins import imu_measurements, run_ins, true_trajectory, wrong_side_probability


def main():
    cfg, out_dir = load_cfg()
    lc = cfg["localize"]
    seed = cfg.get("seed", 0)
    fs, dur, spd = lc["fs"], lc["duration_s"], lc["speed_mps"]
    # Contacts can happen at any instant of the late run; score the error over
    # every sample in the second half so that the phase of the UWB fixes
    # relative to one particular instant does not drive the result.
    i_from = int(0.5 * dur * fs)

    rows = []
    for grade, gcfg in lc["imu_grades"].items():
        for rate in lc["uwb_rate_hz"]:
            for zupt in ([False, True] if rate == 0 else [True]):
                errs, errs_h = [], []
                rng = np.random.default_rng(seed + 100)
                for _ in range(lc["n_runs"]):
                    t, p, v, a, pitch, yaw, stance = true_trajectory(dur, fs, spd, rng)
                    f_b, w_b = imu_measurements(t, a, pitch, yaw, fs, gcfg, rng)
                    est = run_ins(t, f_b, w_b, fs, float(rate), lc["uwb_sigma_m"], p, rng, zupt=zupt)
                    e = est[i_from:] - p[i_from:]
                    errs.append(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))
                    errs_h.append(np.sqrt(np.mean(np.sum(e[:, :2] ** 2, axis=1))))
                rmse = float(np.sqrt(np.mean(np.square(errs)))); rmse_h = float(np.sqrt(np.mean(np.square(errs_h))))
                rows.append({"grade": grade, "uwb_rate_hz": rate, "zupt": zupt,
                             "rmse_m": rmse, "rmse_h_m": rmse_h, "median_m": float(np.median(errs)),
                             "p_wrong_side": {str(d): float(wrong_side_probability(d, rmse_h)) for d in lc["line_distances_m"]}})
                print(f"[localize] {grade:8s} uwb={rate:4.1f} Hz zupt={zupt!s:5s}  rmse={rmse:7.3f} m  "
                      f"(horizontal {rmse_h:.3f})")

    dump(out_dir, "localize.json", {"fs": fs, "duration_s": dur, "speed_mps": spd,
                                    "uwb_sigma_m": lc["uwb_sigma_m"], "n_runs": lc["n_runs"],
                                    "line_distances_m": lc["line_distances_m"], "rows": rows})


if __name__ == "__main__":
    main()
