"""Figures + LaTeX macros for the boot-contact paper.

Reads results/{detect,timing,overlap,decision,localize}.json, writes six PDFs
to paper/figures and every quoted number to paper/macros.tex, so recomputing
the figures rewrites the prose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import apply_style, text_fig

C = {"ball": "#c62828", "player": "#1565c0", "both": "#6a1b9a", "none": "#616161",
     "thr": "#ef6c00", "gb": "#2e7d32", "sport": "#2e7d32", "consumer": "#ef6c00",
     "theory": "#9e9e9e"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()
    res = Path(args.results)
    det = json.loads((res / "detect.json").read_text())
    tim = json.loads((res / "timing.json").read_text())
    ovl = json.loads((res / "overlap.json").read_text())
    dec = json.loads((res / "decision.json").read_text())
    loc = json.loads((res / "localize.json").read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros: list[tuple[str, str]] = []
    m = macros.append

    # ------------------------------------------------------------------ #
    # Fig 1: the four signatures
    # ------------------------------------------------------------------ #
    fs_ex = det["fs_example"]
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.9), sharey=True)
    for ax, cls in zip(axes, det["classes"]):
        ex = det["examples"][cls]
        a = np.asarray(ex["acc"]); mag = np.linalg.norm(a, axis=1)
        t = np.arange(len(mag)) / fs_ex * 1000
        ax.plot(t, mag, color=C[cls], lw=0.9)
        for key, col in (("t_ball", C["ball"]), ("t_player", C["player"])):
            if ex[key] is not None:
                ax.axvline(1000 * ex[key], color=col, ls="--", lw=0.7)
        ax.set_title({"none": "gait only", "ball": "ball", "player": "player", "both": "ball + player"}[cls])
        ax.set_xlabel("time (ms)")
        lo = 1000 * (min([v for v in (ex["t_ball"], ex["t_player"]) if v is not None] or [0.3]) - 0.06)
        ax.set_xlim(lo, lo + 200)
    axes[0].set_ylabel("|a| (g)")
    fig.tight_layout(); fig.savefig(out / "signatures.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: classification vs sampling rate + confusion
    # ------------------------------------------------------------------ #
    rows = det["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    for sensor, mk in (("sport", "o"), ("consumer", "s")):
        r = [x for x in rows if x["background"] == "synthetic" and x["sensor"] == sensor]
        fs = [x["fs"] for x in r]
        a1.semilogx(fs, [x["gb"]["f1"] for x in r], mk + "-", color=C[sensor], label=f"boosting, {sensor}")
        a1.semilogx(fs, [x["threshold"]["f1"] for x in r], mk + "--", color=C[sensor], alpha=0.7,
                    label=f"threshold, {sensor}")
        rp = [x for x in rows if x["background"] == "pamap2" and x["sensor"] == sensor]
        if rp:
            a1.scatter([rp[0]["fs"]], [rp[0]["gb"]["f1"]], marker="*", s=70, color=C[sensor],
                       edgecolors="black", zorder=5, label=f"boosting, real bg" if sensor == "sport" else None)
    a1.set_xlabel("sampling rate (Hz)"); a1.set_ylabel("macro-F1 (4 classes)")
    a1.set_title("Ball / player / both / none"); a1.legend(fontsize=6, loc="lower right"); a1.set_ylim(0.4, 1.0)
    best = max((x for x in rows if x["background"] == "synthetic" and x["sensor"] == "sport"), key=lambda x: x["gb"]["f1"])
    M = np.asarray(best["gb"]["confusion"], dtype=float); M = M / M.sum(axis=1, keepdims=True)
    im = a2.imshow(M, cmap="Blues", vmin=0, vmax=1)
    a2.set_xticks(range(4)); a2.set_yticks(range(4))
    a2.set_xticklabels(det["classes"], fontsize=7); a2.set_yticklabels(det["classes"], fontsize=7)
    for i in range(4):
        for j in range(4):
            a2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 else "black")
    a2.set_xlabel("predicted"); a2.set_ylabel("true"); a2.set_title(f"Boosting @ {best['fs']} Hz, sport")
    a2.grid(False)
    fig.tight_layout(); fig.savefig(out / "detect.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: onset timing and ordering probability
    # ------------------------------------------------------------------ #
    tr = [x for x in tim["rows"] if x["noise_scale"] == 1.0]
    fs = [x["fs"] for x in tr]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    a1.loglog(fs, [x["ball"]["rmse_thr_ms"] for x in tr], "o--", color=C["ball"], alpha=0.7, label="ball, threshold")
    a1.loglog(fs, [x["ball"]["rmse_mf_ms"] for x in tr], "o-", color=C["ball"], label="ball, matched filter")
    a1.loglog(fs, [x["player"]["rmse_mf_ms"] for x in tr], "s-", color=C["player"], label="player, matched filter")
    a1.loglog(fs, [x["player"]["mad_thr_ms"] for x in tr], "s:", color=C["player"], alpha=0.7, label="player, robust spread")
    a1.loglog(fs, [1000 / f / np.sqrt(12) for f in fs], "-", color=C["theory"], lw=0.8, label="sample quantisation")
    a1.set_xlabel("sampling rate (Hz)"); a1.set_ylabel("onset error (ms, RMSE)")
    a1.set_title("How precisely is the contact timed?"); a1.legend(fontsize=6)
    for o, col in zip(tim["ordering"], plt.cm.viridis(np.linspace(0.1, 0.9, len(tim["ordering"])))):
        a2.semilogx(o["gaps_ms"], o["p_empirical"], "o-", color=col, label=f"{o['fs']} Hz (σ={o['sigma_combined_ms']:.1f} ms)")
        a2.semilogx(o["gaps_ms"], o["p_theory"], "--", color=col, alpha=0.5, lw=0.8)
    a2.axhline(0.9, color="grey", lw=0.6, ls=":")
    a2.set_xlabel("true ball-player separation (ms)"); a2.set_ylabel("P(correct order)")
    a2.set_title("Boot + ball sensor: who touched first?"); a2.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "timing.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: the overlap
    # ------------------------------------------------------------------ #
    orow = ovl["rows"]; gaps = [x["gap_ms"] for x in orow]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    a1.plot(gaps, [x["sep_order_acc"] for x in orow], "o-", color=C["both"], label="separator: order correct")
    a1.plot(gaps, [x["sep_frac_two"] for x in orow], "s--", color=C["both"], alpha=0.6, label="separator: two pulses found")
    a1.plot(gaps, [x["gb_frac_both"] for x in orow], "^:", color=C["gb"], label="boosting: labelled 'both'")
    a1.axhline(0.5, color="grey", lw=0.6, ls=":")
    a1.axvspan(-ovl["ball_dur_ms"][1], ovl["ball_dur_ms"][1], color="grey", alpha=0.12, lw=0)
    a1.set_xlabel("player onset − ball onset (ms)"); a1.set_ylabel("fraction")
    a1.set_title("Near-simultaneous contacts"); a1.legend(fontsize=6, loc="lower right"); a1.set_ylim(0, 1.05)
    a2.plot(gaps, [x["sep_gap_rmse_ms"] or np.nan for x in orow], "o-", color=C["both"])
    a2.set_xlabel("player onset − ball onset (ms)"); a2.set_ylabel("gap estimate RMSE (ms)")
    a2.set_title("How well is the gap recovered?")
    fig.tight_layout(); fig.savefig(out / "overlap.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: the probabilistic decision
    # ------------------------------------------------------------------ #
    names = ["prior_only", "boot_only", "boot+prior", "boot+ball+prior", "all", "all_stacked"]
    pretty = ["prior", "boot", "boot+prior", "+ball", "+separator", "stacked"]
    R = dec["results"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    xx = np.arange(len(names))
    a1.bar(xx - 0.2, [R[n]["acc"] for n in names], 0.4, color=C["gb"], label="accuracy")
    a1.bar(xx + 0.2, [1 - R[n]["brier"] for n in names], 0.4, color=C["ball"], alpha=0.8, label="1 − Brier")
    a1.set_xticks(xx); a1.set_xticklabels(pretty, fontsize=6.5, rotation=20)
    a1.set_ylim(0.5, 1.0); a1.set_ylabel("score"); a1.set_title("Evidence added one source at a time")
    a1.legend(fontsize=6, loc="upper left")
    bins = R[dec["best_config"]]["bins"]
    a2.plot([0, 1], [0, 1], "--", color=C["theory"], lw=0.8)
    a2.plot([b["conf"] for b in bins], [b["acc"] for b in bins], "o-", color=C["both"],
            label=f"{dec['best_config']} (ECE {R[dec['best_config']]['ece']:.3f})")
    for b in bins:
        a2.annotate(str(b["n"]), (b["conf"], b["acc"]), textcoords="offset points", xytext=(4, -8), fontsize=5, color="grey")
    a2.set_xlabel("predicted P(ball first)"); a2.set_ylabel("observed fraction ball first")
    a2.set_title("Is the probability honest?"); a2.legend(fontsize=6, loc="upper left")
    a2.set_xlim(0, 1); a2.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / "decision.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 6: localisation
    # ------------------------------------------------------------------ #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    for grade in loc["rows"][0]["grade"], loc["rows"][-1]["grade"]:
        r = [x for x in loc["rows"] if x["grade"] == grade and x["zupt"]]
        rates = [x["uwb_rate_hz"] for x in r]; rmse = [x["rmse_h_m"] for x in r]
        xs = [max(rt, 0.3) for rt in rates]
        a1.loglog(xs, rmse, "o-", color=C[grade], label=f"{grade} IMU + ZUPT")
        free = [x for x in loc["rows"] if x["grade"] == grade and not x["zupt"]][0]
        a1.scatter([0.3], [free["rmse_h_m"]], marker="x", color=C[grade], label=f"{grade}, no ZUPT")
    a1.set_xticks([0.3, 1, 2, 5, 10, 20]); a1.set_xticklabels(["none", "1", "2", "5", "10", "20"])
    a1.set_xlabel("UWB fix rate (Hz)"); a1.set_ylabel("horizontal RMSE (m)")
    a1.set_title("Where was the boot?"); a1.legend(fontsize=6)
    d = np.linspace(0.02, 1.0, 100)
    from scipy.stats import norm
    for x in [r for r in loc["rows"] if r["grade"] == "sport" and r["zupt"] and r["uwb_rate_hz"] in (0, 1, 5, 20)]:
        a2.semilogy(d, norm.cdf(-d / x["rmse_h_m"]), label=f"UWB {x['uwb_rate_hz']} Hz (σ={x['rmse_h_m']:.2f} m)")
    a2.set_ylim(1e-4, 0.6); a2.set_xlabel("contact distance to the line (m)")
    a2.set_ylabel("P(wrong side of the line)"); a2.set_title("The penalty-area call"); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "localize.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    def row(bg, sensor, fs_):
        return next(x for x in rows if x["background"] == bg and x["sensor"] == sensor and x["fs"] == fs_)
    sport = [x for x in rows if x["background"] == "synthetic" and x["sensor"] == "sport"]
    cons = [x for x in rows if x["background"] == "synthetic" and x["sensor"] == "consumer"]
    pam = [x for x in rows if x["background"] == "pamap2"]
    m(("DetNPerClass", str((sport[0]["n_train"] + sport[0]["n_test"]) // 4)))
    m(("DetNTrain", str(sport[0]["n_train"]))); m(("DetNTest", str(sport[0]["n_test"])))
    m(("DetGbFBest", f"{max(x['gb']['f1'] for x in sport):.3f}"))
    m(("DetGbFBestFs", str(best["fs"])))
    m(("DetGbFHundred", f"{row('synthetic', 'sport', 100)['gb']['f1']:.3f}"))
    m(("DetGbFOneK", f"{row('synthetic', 'sport', 1000)['gb']['f1']:.3f}"))
    m(("DetThrFBest", f"{max(x['threshold']['f1'] for x in sport):.3f}"))
    m(("DetThrFMin", f"{min(x['threshold']['f1'] for x in sport + cons):.3f}"))
    m(("DetThrFMax", f"{max(x['threshold']['f1'] for x in sport + cons):.3f}"))
    m(("DetConsFHundred", f"{row('synthetic', 'consumer', 100)['gb']['f1']:.3f}"))
    m(("DetConsFBest", f"{max(x['gb']['f1'] for x in cons):.3f}"))
    m(("DetConsClipPct", f"{100 * np.mean([x['clip_frac'] for x in cons]):.0f}"))
    m(("DetPamGbSport", f"{next(x for x in pam if x['sensor'] == 'sport')['gb']['f1']:.3f}" if pam else "--"))
    m(("DetPamGbCons", f"{next(x for x in pam if x['sensor'] == 'consumer')['gb']['f1']:.3f}" if pam else "--"))
    m(("DetPamThrSport", f"{next(x for x in pam if x['sensor'] == 'sport')['threshold']['f1']:.3f}" if pam else "--"))
    m(("DetPamSegments", "10"))

    t1k = next(x for x in tr if x["fs"] == 1000); t100 = next(x for x in tr if x["fs"] == 100)
    t500 = next(x for x in tr if x["fs"] == 500)
    m(("TimBallOneK", f"{min(t1k['ball']['rmse_thr_ms'], t1k['ball']['rmse_mf_ms']):.2f}"))
    m(("TimBallFiveH", f"{min(t500['ball']['rmse_thr_ms'], t500['ball']['rmse_mf_ms']):.2f}"))
    m(("TimBallHundred", f"{min(t100['ball']['rmse_thr_ms'], t100['ball']['rmse_mf_ms']):.1f}"))
    m(("TimPlayerOneK", f"{t1k['player']['rmse_mf_ms']:.1f}"))
    m(("TimPlayerMadOneK", f"{t1k['player']['mad_thr_ms']:.1f}"))
    m(("TimPlayerHundred", f"{t100['player']['rmse_mf_ms']:.1f}"))
    o1k = next(o for o in tim["ordering"] if o["fs"] == 1000); o100 = next(o for o in tim["ordering"] if o["fs"] == 100)
    m(("OrdSigmaOneK", f"{o1k['sigma_combined_ms']:.1f}")); m(("OrdSigmaHundred", f"{o100['sigma_combined_ms']:.1f}"))
    m(("OrdSigmaBall", f"{o1k['sigma_ball_ms']:.1f}"))

    def gap90(o):
        for g, p in zip(o["gaps_ms"], o["p_empirical"]):
            if p >= 0.9:
                return f"{g:.0f}"
        return ">" + f"{o['gaps_ms'][-1]:.0f}"
    m(("OrdGapNinetyOneK", gap90(o1k))); m(("OrdGapNinetyHundred", gap90(o100)))
    m(("OrdPTenOneK", f"{o1k['p_empirical'][o1k['gaps_ms'].index(10.0)]:.2f}"))
    m(("OrdPTenHundred", f"{o100['p_empirical'][o100['gaps_ms'].index(10.0)]:.2f}"))
    m(("OrdPTwoOneK", f"{o1k['p_empirical'][o1k['gaps_ms'].index(2.0)]:.2f}"))

    def oacc(g):
        return np.mean([x["sep_order_acc"] for x in orow if abs(x["gap_ms"]) == g])
    m(("OvlAccThree", f"{oacc(3):.2f}")); m(("OvlAccSix", f"{oacc(6):.2f}")); m(("OvlAccTen", f"{oacc(10):.2f}"))
    m(("OvlAccFifteen", f"{oacc(15):.2f}")); m(("OvlAccTwenty", f"{oacc(20):.2f}"))
    m(("OvlAccZero", f"{oacc(0):.2f}"))
    m(("OvlTwoMin", f"{min(x['sep_frac_two'] for x in orow):.2f}"))
    m(("OvlGbBothMin", f"{min(x['gb_frac_both'] for x in orow):.2f}"))
    m(("OvlGbBothMax", f"{max(x['gb_frac_both'] for x in orow):.2f}"))
    wide = [x["sep_gap_rmse_ms"] for x in orow if abs(x["gap_ms"]) >= 20 and x["sep_gap_rmse_ms"]]
    narrow = [x["sep_gap_rmse_ms"] for x in orow if abs(x["gap_ms"]) <= 6 and x["sep_gap_rmse_ms"]]
    m(("OvlGapRmseWide", f"{np.mean(wide):.0f}")); m(("OvlGapRmseNarrow", f"{np.mean(narrow):.0f}"))
    m(("OvlNPerGap", str(orow[0]["n"]))); m(("OvlFs", str(ovl["fs"])))

    for key, nm in (("prior_only", "Prior"), ("boot_only", "Boot"), ("boot+prior", "BootPrior"),
                    ("boot+ball+prior", "BootBall"), ("all", "All"), ("all_stacked", "Stack")):
        m((f"Dec{nm}Acc", f"{R[key]['acc']:.3f}")); m((f"Dec{nm}Brier", f"{R[key]['brier']:.3f}"))
        m((f"Dec{nm}Ece", f"{R[key]['ece']:.3f}"))
        m((f"Dec{nm}AccSingle", f"{R[key]['acc_single']:.3f}")); m((f"Dec{nm}AccBoth", f"{R[key]['acc_both']:.3f}"))
    m(("DecNCases", str(dec["n_cases"]))); m(("DecNEval", str(dec["n_eval"])))
    m(("DecBaseRate", f"{dec['base_rate_ball_first']:.2f}"))
    m(("DecBest", dec["best_config"].replace("_", " ")))
    sw = next(s for s in dec["cost_sweep"] if s["cost_ratio_false_to_missed"] == 1.0 and s["review_band"] > 0)
    sw0 = next(s for s in dec["cost_sweep"] if s["cost_ratio_false_to_missed"] == 1.0 and s["review_band"] == 0)
    m(("DecReviewRate", f"{100 * sw['review_rate']:.0f}")); m(("DecErrDecided", f"{100 * sw['error_among_decided']:.1f}"))
    m(("DecErrNoReview", f"{100 * sw0['error_among_decided']:.1f}"))
    sw4 = next(s for s in dec["cost_sweep"] if s["cost_ratio_false_to_missed"] == 4.0 and s["review_band"] == 0)
    m(("DecThrFour", f"{sw4['threshold']:.2f}")); m(("DecFalseFoulFour", f"{100 * sw4['false_foul_rate']:.1f}"))
    m(("DecFalseFoulOne", f"{100 * sw0['false_foul_rate']:.1f}")); m(("DecMissedFour", f"{100 * sw4['missed_foul_rate']:.1f}"))
    m(("DecMissedOne", f"{100 * sw0['missed_foul_rate']:.1f}"))
    coef = dict(zip(dec["stack_sources"], dec["stack_coef"]))
    m(("DecCoefBoot", f"{coef['boot']:.2f}")); m(("DecCoefBall", f"{coef['ball']:.2f}"))
    m(("DecCoefSep", f"{coef['sep']:.2f}")); m(("DecCoefPrior", f"{coef['prior']:.2f}"))
    late = next(s for s in dec["per_situation"] if s["situation"] == "late_challenge")
    m(("DecLateAccAll", f"{late['acc_all']:.2f}")); m(("DecLateAccPrior", f"{late['acc_prior']:.2f}"))

    def lrow(grade, rate, zupt=True):
        return next(x for x in loc["rows"] if x["grade"] == grade and x["uwb_rate_hz"] == rate and x["zupt"] == zupt)
    m(("LocFreeSport", f"{lrow('sport', 0, False)['rmse_h_m']:.1f}")); m(("LocZuptSport", f"{lrow('sport', 0)['rmse_h_m']:.2f}"))
    m(("LocFreeCons", f"{lrow('consumer', 0, False)['rmse_h_m']:.1f}")); m(("LocZuptCons", f"{lrow('consumer', 0)['rmse_h_m']:.2f}"))
    m(("LocUwbOneSport", f"{lrow('sport', 1)['rmse_h_m']:.2f}")); m(("LocUwbFiveSport", f"{lrow('sport', 5)['rmse_h_m']:.2f}"))
    m(("LocUwbTwentySport", f"{lrow('sport', 20)['rmse_h_m']:.2f}")); m(("LocUwbTwentyCons", f"{lrow('consumer', 20)['rmse_h_m']:.2f}"))
    m(("LocUwbTenSport", f"{lrow('sport', 10)['rmse_h_m']:.2f}"))
    m(("LocWrongHalfTen", f"{100 * lrow('sport', 10)['p_wrong_side']['0.5']:.2f}"))
    m(("LocWrongQuarterTen", f"{100 * lrow('sport', 10)['p_wrong_side']['0.25']:.1f}"))
    m(("LocWrongTenTen", f"{100 * lrow('sport', 10)['p_wrong_side']['0.1']:.0f}"))
    m(("LocWrongHalfZupt", f"{100 * lrow('sport', 0)['p_wrong_side']['0.5']:.0f}"))
    m(("LocUwbSigma", f"{loc['uwb_sigma_m']:.2f}")); m(("LocNRuns", str(loc["n_runs"])))
    m(("LocDuration", f"{loc['duration_s']:.0f}")); m(("LocSpeed", f"{loc['speed_mps']:.0f}"))

    Path(args.macros).write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 6 figures")


if __name__ == "__main__":
    main()
