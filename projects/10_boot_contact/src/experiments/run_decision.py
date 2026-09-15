"""Experiment 4 -- the probabilistic call: ball first, or straight to the player?

Each case is a challenge: a situation is drawn from the knowledge base (with
its base rate of playing the ball first), the truth is drawn from that base
rate, and the boot + ball-sensor signals are simulated.  Four evidence sources
are scored per case and fused in log-odds:

    prior   -- knowledge-base base rate for the situation
    boot    -- classifier posterior from the boot window
    ball    -- ball-sensor impact time vs boot onset (timing consistency)
    sep     -- the two-template separator's order and its gap-based confidence

Per-source calibration is fitted on one half of the cases and evaluated on the
other.  Reported: accuracy, Brier score, expected calibration error, and the
reliability diagram, for prior-only, boot-only, boot+prior, +ball, +sep (all).
Then the decision rule: expected-cost threshold with a review band, over a
sweep of the false-foul / missed-foul cost ratio.  Writes results/decision.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from src.decide.bayes import Fusion, brier, decide, ece, timing_score
from src.detect.ml import GBDetector
from src.detect.onset import onset_threshold, order_probability
from src.detect.separate import separate
from src.sim.dataset import clean_templates, make_windows
from src.sim.events import CLASS_ID
from src.sim.imu import SensorSpec
from src.sim.events import ball_sensor, inject
from src.sim.gait import synth_gait
from src.sim.imu import sense


def _one_case(label, gap, fs, scfg, sim, ball_cfg, rng):
    t, acc, gyr = synth_gait(sim["window_s"], sim["sim_rate"], rng,
                             cadence_hz=tuple(sim["gait"]["cadence_hz"]),
                             strike_peak_g=tuple(sim["gait"]["strike_peak_g"]),
                             strike_dur_ms=tuple(sim["gait"]["strike_dur_ms"]))
    sig, truth = inject(t, label, rng, sim, gap_ms=gap)
    a, w = sense(acc + sig[:, :3], gyr + sig[:, 3:], SensorSpec.from_cfg(fs, scfg), sim["sim_rate"], rng)
    return np.concatenate([a, w], axis=1).astype(np.float32), truth, ball_sensor(truth, sim["window_s"], ball_cfg, sim, rng)


def main():
    cfg, out_dir = load_cfg()
    dc, sim, sensors = cfg["decision"], cfg["sim"], cfg["sensors"]
    seed = cfg.get("seed", 0)
    fs, scfg, ball_cfg = dc["fs"], sensors[dc["sensor"]], sensors["ball"]
    rng = np.random.default_rng(seed + 44)

    # Detector and templates trained on the general distribution.
    tr = make_windows(cfg["detect"]["n_per_class"], fs, scfg, sim, rng)
    gb = GBDetector(max_iter=cfg["detect"]["gb_max_iter"], seed=seed).fit(tr.X, tr.y, fs)
    tpl = clean_templates(fs, sim, rng)
    # Boot-onset error at this rate, for the timing likelihood.
    wb = make_windows(120, fs, scfg, sim, rng, classes=["ball"])
    sig_boot = float(np.sqrt(np.mean([(1000 * (onset_threshold(w, fs) - t.t_ball)) ** 2 for w, t in zip(wb.X, wb.truths)])))
    sig_ball = float(np.sqrt(ball_cfg["sync_jitter_ms"] ** 2 + (1000.0 / ball_cfg["fs_hz"]) ** 2 / 12.0))
    sig_comb = float(np.sqrt(sig_boot ** 2 + sig_ball ** 2))

    sit_names = list(dc["situations"].keys())
    freqs = np.array([dc["situations"][s]["freq"] for s in sit_names]); freqs /= freqs.sum()
    priors = np.array([dc["situations"][s]["prior_ball_first"] for s in sit_names])

    n = dc["n_cases"]
    y = np.zeros(n, dtype=int); prior = np.zeros(n); sit = np.zeros(n, dtype=int)
    s_boot = np.zeros(n); s_ball = np.zeros(n); s_sep = np.zeros(n); labels = []
    for i in range(n):
        k = rng.choice(len(sit_names), p=freqs)
        sit[i] = k; prior[i] = priors[k]
        ball_first = rng.random() < priors[k]
        y[i] = int(ball_first)
        # Half of each outcome is a clean single contact, half a near-simultaneous one.
        if rng.random() < 0.5:
            label, gap = ("ball" if ball_first else "player"), None
        else:
            label = "both"
            gap = float(rng.uniform(2.0, 40.0)) * (1.0 if ball_first else -1.0)
        labels.append(label)
        w, truth, rep = _one_case(label, gap, fs, scfg, sim, ball_cfg, rng)
        p = gb.predict_proba(w[None])[0]
        s_boot[i] = p[CLASS_ID["ball"]] + 0.5 * p[CLASS_ID["both"]]
        r = separate(w, fs, tpl, step_ms=cfg["overlap"]["grid_step_ms"])
        s_ball[i] = timing_score(rep["t_ball_hat"], r["t_first"], sig_comb)
        if r["model"] == "ball":
            s_sep[i] = 0.9
        elif r["model"] == "player":
            s_sep[i] = 0.1
        else:
            conf = float(order_probability(r["gap_ms"], sig_boot))
            s_sep[i] = conf if r["order"] == "ball_first" else 1.0 - conf

    scores = {"boot": s_boot, "ball": s_ball, "sep": s_sep}
    half = n // 2
    cal = {k: v[:half] for k, v in scores.items()}
    ev = {k: v[half:] for k, v in scores.items()}
    fusion = Fusion(["boot", "ball", "sep"]).fit(cal, y[:half])
    y_ev, prior_ev, sit_ev = y[half:], prior[half:], sit[half:]
    lab_ev = np.array(labels[half:])

    configs = {
        "prior_only": ([], True), "boot_only": (["boot"], False), "boot+prior": (["boot"], True),
        "boot+ball+prior": (["boot", "ball"], True), "all": (["boot", "ball", "sep"], True),
        "all_stacked": (["boot", "ball", "sep"], True),
    }
    results = {}
    # Stacked alternative: a logistic regression on the calibrated log-odds of
    # every source plus the prior, fitted on the calibration half.  Unlike the
    # naive-Bayes sum it can down-weight sources that are correlated (boot and
    # sep both read the same window).
    from sklearn.linear_model import LogisticRegression
    from src.decide.bayes import logit as _logit
    def _stack_feats(sc, pr):
        return np.stack([_logit(fusion.cal[s](sc[s])) for s in fusion.sources] + [_logit(pr)], axis=1)
    stack = LogisticRegression(C=1.0, max_iter=1000).fit(_stack_feats(cal, prior[:half]), y[:half])
    p_stack = stack.predict_proba(_stack_feats(ev, prior_ev))[:, 1]
    for name, (use, with_prior) in configs.items():
        pr = prior_ev if with_prior else np.full_like(prior_ev, fusion.base)
        p = fusion.posterior(ev, pr, use=use) if use else pr.copy()
        if name == "all_stacked":
            p = p_stack
        e, bins = ece(p, y_ev.astype(float), dc["calib_bins"])
        both_m = lab_ev == "both"
        results[name] = {
            "acc": float(np.mean((p > 0.5) == y_ev)), "brier": brier(p, y_ev), "ece": e, "bins": bins,
            "acc_single": float(np.mean((p[~both_m] > 0.5) == y_ev[~both_m])),
            "acc_both": float(np.mean((p[both_m] > 0.5) == y_ev[both_m])),
            "log_loss": float(-np.mean(y_ev * np.log(p) + (1 - y_ev) * np.log(1 - p))),
        }
        print(f"[decision] {name:16s} acc={results[name]['acc']:.3f} brier={results[name]['brier']:.3f} "
              f"ece={e:.3f}  (single {results[name]['acc_single']:.3f} / overlap {results[name]['acc_both']:.3f})")

    # Decision rule: cost sweep with the best posterior (lowest Brier).
    best_name = min(results, key=lambda k: results[k]["brier"])
    p_all = p_stack if best_name == "all_stacked" else fusion.posterior(
        ev, prior_ev if configs[best_name][1] else np.full_like(prior_ev, fusion.base), use=configs[best_name][0])
    sweep = []
    for ratio in (0.25, 0.5, 1.0, 2.0, 4.0):
        for band in (0.0, dc["review_band"]):
            d = decide(p_all, ratio * dc["cost_missed_foul"], dc["cost_missed_foul"], review_band=band)
            decided = d != 2
            fouls = d == 0
            false_foul = np.mean(fouls & (y_ev == 1))
            missed = np.mean((d == 1) & (y_ev == 0))
            sweep.append({"cost_ratio_false_to_missed": ratio, "review_band": band,
                          "threshold": dc["cost_missed_foul"] / (ratio * dc["cost_missed_foul"] + dc["cost_missed_foul"]),
                          "review_rate": float(np.mean(~decided)), "foul_rate": float(np.mean(fouls)),
                          "false_foul_rate": float(false_foul), "missed_foul_rate": float(missed),
                          "error_among_decided": float(np.mean(((d == 1) != (y_ev == 1))[decided])) if decided.any() else None,
                          "expected_cost": float(ratio * false_foul + missed)})

    per_sit = []
    for k, s in enumerate(sit_names):
        m = sit_ev == k
        per_sit.append({"situation": s, "prior": float(priors[k]), "n": int(m.sum()),
                        "acc_all": float(np.mean((p_all[m] > 0.5) == y_ev[m])),
                        "acc_prior": float(np.mean((prior_ev[m] > 0.5) == y_ev[m]))})

    dump(out_dir, "decision.json", {
        "fs": fs, "sensor": dc["sensor"], "n_cases": n, "n_eval": int(n - half),
        "sigma_boot_ms": sig_boot, "sigma_ball_ms": sig_ball, "sigma_combined_ms": sig_comb,
        "base_rate_ball_first": float(y.mean()), "results": results, "cost_sweep": sweep,
        "best_config": best_name, "stack_coef": stack.coef_[0].tolist(), "stack_sources": fusion.sources + ["prior"],
        "per_situation": per_sit, "situations": dc["situations"],
        "frac_overlap_cases": float(np.mean(np.array(labels) == "both")),
    })


if __name__ == "__main__":
    main()
