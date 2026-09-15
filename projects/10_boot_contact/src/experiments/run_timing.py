"""Experiment 2 -- onset timing precision and who-touched-first.

(a) Timing: RMSE of the estimated contact onset (threshold rule and matched
    filter) for ball and player contacts, over sampling rate and noise level.
(b) Ordering: using the measured boot-onset error sigma at each rate together
    with the ball sensor's own timing error, the probability of ordering the
    ball contact and the player contact correctly as a function of their true
    separation -- both the closed form Phi(gap / (sigma sqrt 2)) and an
    empirical check with the boot + ball-sensor rule at each sampling rate.
Writes results/timing.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.detect.onset import onset_matched, onset_threshold, order_probability
from src.detect.separate import separate
from src.sim.dataset import clean_templates, make_windows


def _rmse(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2)))


def main():
    cfg, out_dir = load_cfg()
    tcfg, sim, sensors = cfg["timing"], cfg["sim"], cfg["sensors"]
    seed = cfg.get("seed", 0)
    ball_cfg = sensors["ball"]

    timing_rows = []
    sigma_at = {}
    for fs in tcfg["fs_grid"]:
        rng = np.random.default_rng(seed + 10 * fs)
        tpl = clean_templates(fs, sim, rng)
        for ns in tcfg["noise_scale"]:
            row = {"fs": fs, "noise_scale": ns}
            for label in ("ball", "player"):
                ws = make_windows(tcfg["n_events"] // 2, fs, sensors["sport"], sim, rng,
                                  classes=[label], noise_scale=ns)
                truth = np.array([t.t_ball if label == "ball" else t.t_player for t in ws.truths])
                e_thr = 1000 * (np.array([onset_threshold(w, fs) for w in ws.X]) - truth)
                e_mf = 1000 * (np.array([onset_matched(w, fs, tpl[label]) for w in ws.X]) - truth)
                row[label] = {"rmse_thr_ms": _rmse(e_thr), "bias_thr_ms": float(e_thr.mean()),
                              "rmse_mf_ms": _rmse(e_mf), "bias_mf_ms": float(e_mf.mean()),
                              "mad_thr_ms": float(np.median(np.abs(e_thr - np.median(e_thr))) * 1.4826)}
            timing_rows.append(row)
            if ns == 1.0:
                sigma_at[fs] = min(row["ball"]["rmse_thr_ms"], row["ball"]["rmse_mf_ms"])
            print(f"[timing] fs={fs:5d} noise x{ns:<4}  ball thr/mf = "
                  f"{row['ball']['rmse_thr_ms']:.2f}/{row['ball']['rmse_mf_ms']:.2f} ms   "
                  f"player thr/mf = {row['player']['rmse_thr_ms']:.2f}/{row['player']['rmse_mf_ms']:.2f} ms")

    # Ordering: the boot's *first* contact onset (from the two-template
    # separator) vs the ball sensor's time.  Rule: ball first iff
    # t_ball_hat - t_boot_first < tau, with tau = 2 * sigma_combined.
    gaps = np.array(tcfg["gap_grid_ms"], dtype=float)
    ordering = []
    for fs in tcfg["fs_grid"]:
        rng = np.random.default_rng(seed + 7 * fs + 1)
        tpl = clean_templates(fs, sim, rng)
        s_boot = sigma_at[fs]
        s_ball = np.sqrt(ball_cfg["sync_jitter_ms"] ** 2 + (1000.0 / ball_cfg["fs_hz"]) ** 2 / 12.0)
        s_comb = float(np.sqrt(s_boot ** 2 + s_ball ** 2))
        tau = 2.0 * s_comb
        theory = order_probability(gaps, s_comb).tolist()
        emp = []
        for gap in gaps:
            n = max(40, tcfg["n_events"] // len(gaps))
            correct = 0
            total = 0
            for sign in (+1.0, -1.0):
                ws = make_windows(n // 2, fs, sensors["sport"], sim, rng, classes=["both"],
                                  gap_ms=sign * gap, ball_cfg=ball_cfg)
                for w, tr, rep in zip(ws.X, ws.truths, ws.ball):
                    t_boot = separate(w, fs, tpl)["t_first"]
                    d_ms = 1000.0 * (rep["t_ball_hat"] - t_boot)
                    pred_ball_first = d_ms < tau
                    correct += int(pred_ball_first == tr.ball_first)
                    total += 1
            emp.append(correct / total)
        ordering.append({"fs": fs, "sigma_boot_ms": s_boot, "sigma_ball_ms": s_ball,
                         "sigma_combined_ms": s_comb, "tau_ms": tau,
                         "gaps_ms": gaps.tolist(), "p_theory": theory, "p_empirical": emp})
        print(f"[timing] ordering fs={fs:5d} sigma={s_comb:.2f} ms  "
              f"empirical={np.round(emp, 2).tolist()}")

    dump(out_dir, "timing.json", {"rows": timing_rows, "ordering": ordering,
                                  "n_events": tcfg["n_events"]})


if __name__ == "__main__":
    main()
