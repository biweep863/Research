"""Figures + LaTeX macros for the optimal-floor paper.

Reads results/{analytic,traffic,stairs,fleet,robust}.json, writes six PDFs
to paper/figures and every quoted number to paper/macros.tex.
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

from utils.plotting import apply_style

POL_STYLE = {"lobby": ("#c62828", "-"), "middle": ("#1565c0", "--"), "last": ("#2e7d32", "-."), "random": ("#6a1b9a", ":")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args(); apply_style()
    res = Path(args.results)
    ana = json.loads((res / "analytic.json").read_text())
    tra = json.loads((res / "traffic.json").read_text())
    sta = json.loads((res / "stairs.json").read_text())
    fle = json.loads((res / "fleet.json").read_text())
    rob = json.loads((res / "robust.json").read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []; m = macros.append

    # ------------------------------------------------------------------ #
    # Fig 1: closed-form curves
    # ------------------------------------------------------------------ #
    N = ana["curve_n_floors"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    alphas = sorted({c["alpha"] for c in ana["curves"]})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(alphas)))
    for al, col in zip(alphas, cmap):
        c = next(x for x in ana["curves"] if x["alpha"] == al and x["parking"] == "lobby")
        a1.plot(c["homes"], c["mean"], "-", color=col, label=f"α = {al:.2f}")
        a1.scatter([c["homes"][int(np.argmin(c["mean"]))]], [min(c["mean"])], color=col, s=18, zorder=4)
    a1.set_xlabel("home floor h"); a1.set_ylabel("expected trip time (s)")
    a1.set_title(f"Car parks at the lobby, {N} floors"); a1.legend(fontsize=6, ncol=2)
    for pol, (col, ls) in POL_STYLE.items():
        c = next(x for x in ana["curves"] if x["alpha"] == 0.5 and x["parking"] == pol)
        a2.plot(c["homes"], c["mean"], ls, color=col, label=f"parks: {pol}")
        a2.scatter([c["homes"][int(np.argmin(c["mean"]))]], [min(c["mean"])], color=col, s=18, zorder=4)
    a2.set_xlabel("home floor h"); a2.set_ylabel("expected trip time (s)")
    a2.set_title("Where the car waits (α = 0.5)"); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "curves.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: traffic
    # ------------------------------------------------------------------ #
    lob = [b for b in tra["blocks"] if b["parking"] == "lobby"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.75, 2.4))
    cm = plt.cm.inferno(np.linspace(0.1, 0.8, len(lob)))
    for b, col in zip(lob, cm):
        a1.plot(b["homes"], [r["mean"] for r in b["rows"]], "-", color=col, label=f"{b['rate']:.1f}/min")
        if b["rate"] == 0.0:
            a1.plot(b["homes"], b["analytic_mean"], "--", color="black", lw=0.8, label="closed form")
    a1.set_xlabel("home floor h"); a1.set_ylabel("mean trip time (s)"); a1.set_title("Background traffic, one car")
    a1.legend(fontsize=5.5, title="passengers", title_fontsize=6)
    hi = lob[-1]
    a2.plot(hi["homes"], [r["mean_wait"] for r in hi["rows"]], "-", color="#c62828", label="wait")
    a2.plot(hi["homes"], [r["mean_ride"] for r in hi["rows"]], "-", color="#1565c0", label="ride")
    a2b = a2.twinx(); a2b.plot(hi["homes"], [r["mean_stops"] for r in hi["rows"]], ":", color="grey", label="stops on board")
    a2b.set_ylabel("intermediate stops", fontsize=7); a2b.grid(False)
    a2.set_xlabel("home floor h"); a2.set_ylabel("seconds"); a2.set_title(f"Decomposition at {hi['rate']:.0f}/min")
    a2.legend(fontsize=6, loc="upper center")
    last = [b for b in tra["blocks"] if b["parking"] == "last"]
    for b, col in zip(last, plt.cm.inferno(np.linspace(0.1, 0.8, len(last)))):
        h = np.array(b["car_pos_hist"], dtype=float)
        if h.sum():
            a3.plot(np.arange(1, len(h) + 1), h / h.sum(), "-", color=col, label=f"{b['rate']:.1f}/min")
    a3.set_xlabel("car position when called"); a3.set_ylabel("probability"); a3.set_title("Car 'stays put': where is it?")
    a3.legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(out / "traffic.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: stairs
    # ------------------------------------------------------------------ #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5), gridspec_kw={"width_ratios": [1.3, 1]})
    for v, col in zip(sta["variants"], plt.cm.tab10(np.arange(len(sta["variants"])))):
        a1.plot(v["homes"], [r["mean"] for r in v["rows"]], "-", color=col, label=v["name"])
    a1.set_xlabel("home floor h"); a1.set_ylabel("mean trip time (s)")
    a1.set_title(f"Stairs on (α = {sta['alpha']}, {sta['rate']:.0f}/min)"); a1.legend(fontsize=5.5)
    names = [v["name"] for v in sta["variants"]]; xx = np.arange(len(names))
    a2.bar(xx, [100 * v["frac_walked_overall"] for v in sta["variants"]], color="#2e7d32")
    a2.set_xticks(xx); a2.set_xticklabels(names, rotation=30, ha="right", fontsize=6)
    a2.set_ylabel("% of trips walked"); a2.set_title("Who takes the stairs")
    fig.tight_layout(); fig.savefig(out / "stairs.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: fleet
    # ------------------------------------------------------------------ #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    for b, col in zip(fle["blocks"], plt.cm.tab10(np.arange(len(fle["blocks"])))):
        means = np.array([r["mean"] for r in b["rows"]])
        a1.plot(b["homes"], means, "-", color=col, label=f"{b['n_cars']} car(s), {b['rate']:.1f}/min")
        a2.plot(b["homes"], 100 * (means / means.min() - 1), "-", color=col, label=f"{b['n_cars']} car(s)")
    a1.set_xlabel("home floor h"); a1.set_ylabel("mean trip time (s)"); a1.set_title("Fleet size (load scales with cars)")
    a1.legend(fontsize=6)
    a2.set_xlabel("home floor h"); a2.set_ylabel("regret vs best floor (%)"); a2.set_title("Does a bigger fleet flatten the curve?")
    a2.axhline(5, color="grey", lw=0.6, ls=":"); a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fleet.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: the (alpha, rate) map
    # ------------------------------------------------------------------ #
    A = sorted({g["alpha"] for g in rob["grid"]}); R = sorted({g["rate"] for g in rob["grid"]})
    Hs = np.zeros((len(R), len(A))); Rm = np.zeros_like(Hs); Bw = np.zeros_like(Hs)
    for g in rob["grid"]:
        i, j = R.index(g["rate"]), A.index(g["alpha"])
        Hs[i, j] = g["h_star"]; Rm[i, j] = g["regret_middle_pct"]; Bw[i, j] = g["band_5pct_width"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.6))
    im = a1.imshow(Hs, cmap="viridis", origin="lower", aspect="auto", vmin=2, vmax=rob["n_floors"])
    for i in range(len(R)):
        for j in range(len(A)):
            a1.text(j, i, f"{int(Hs[i, j])}\n±{int(Bw[i, j])}", ha="center", va="center", fontsize=5.5,
                    color="white" if Hs[i, j] < 0.6 * rob["n_floors"] else "black")
    a1.set_xticks(range(len(A))); a1.set_xticklabels([f"{a:.2f}" for a in A]); a1.set_yticks(range(len(R))); a1.set_yticklabels([f"{r:.0f}" for r in R])
    a1.set_xlabel("lobby-trip fraction α"); a1.set_ylabel("background passengers / min")
    a1.set_title(f"Optimal floor of {rob['n_floors']} (band width below)"); a1.grid(False)
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    im2 = a2.imshow(np.log10(Rm + 1), cmap="magma", origin="lower", aspect="auto")
    for i in range(len(R)):
        for j in range(len(A)):
            a2.text(j, i, f"{Rm[i, j]:.0f}%", ha="center", va="center", fontsize=6, color="white" if Rm[i, j] < 60 else "black")
    a2.set_xticks(range(len(A))); a2.set_xticklabels([f"{a:.2f}" for a in A]); a2.set_yticks(range(len(R))); a2.set_yticklabels([f"{r:.0f}" for r in R])
    a2.set_xlabel("lobby-trip fraction α"); a2.set_ylabel("background passengers / min")
    a2.set_title("Regret of the 'middle floor' rule"); a2.grid(False)
    fig.tight_layout(); fig.savefig(out / "map.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 6: archetypes
    # ------------------------------------------------------------------ #
    arch = rob["archetypes"]
    fig, axes = plt.subplots(1, len(arch), figsize=(6.75, 2.4))
    for ax, (name, a) in zip(axes, arch.items()):
        means = np.array([r["mean"] for r in a["rows"]]); p90 = np.array([r["p90"] for r in a["rows"]])
        ax.plot(a["homes"], means, "-", color="#1565c0", label="mean"); ax.plot(a["homes"], p90, "--", color="#ef6c00", label="p90")
        ax.axvspan(a["band_5pct"][0] - 0.5, a["band_5pct"][1] + 0.5, color="#2e7d32", alpha=0.15, lw=0, label="within 5 % of best")
        mid = round((a["n_floors"] + 1) / 2); ax.axvline(mid, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{name}: α={a['alpha']}, {a['rate']:.0f}/min, {a['n_cars']} car(s)", fontsize=8)
        ax.set_xlabel("home floor h")
    axes[0].set_ylabel("trip time (s)"); axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "regret.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    kin = ana["kinematics"]
    def arow(N_, al, pol):
        return next(r for r in ana["rows"] if r["n_floors"] == N_ and r["alpha"] == al and r["parking"] == pol)
    m(("KinV", f"{kin['v']:.1f}")); m(("KinA", f"{kin['a']:.1f}")); m(("KinDoor", f"{kin['t_door']:.0f}")); m(("KinPass", f"{kin['t_pass']:.1f}"))
    m(("CurveN", str(N)))
    m(("AnaMiddleLastZero", str(arow(N, 0.0, "last")["h_star"]))); m(("AnaLobbyZero", str(arow(N, 0.0, "lobby")["h_star"])))
    m(("AnaLobbyQuarter", str(arow(N, 0.25, "lobby")["h_star"]))); m(("AnaLobbyHalf", str(arow(N, 0.5, "lobby")["h_star"])))
    m(("AnaRandomHalf", str(arow(N, 0.5, "random")["h_star"]))); m(("AnaLastHalf", str(arow(N, 0.5, "last")["h_star"])))
    m(("AnaMiddleHalf", str(arow(N, 0.5, "middle")["h_star"]))); m(("AnaMiddleNine", str(arow(N, 0.9, "middle")["h_star"])))
    m(("AnaRegretMidHalfLobby", f"{arow(N, 0.5, 'lobby')['regret_middle_pct']:.0f}")); m(("AnaRegretMidNineLobby", f"{arow(N, 0.9, 'lobby')['regret_middle_pct']:.0f}"))
    m(("AnaRegretTopNineLobby", f"{arow(N, 0.9, 'lobby')['regret_top_pct']:.0f}")); m(("AnaRegretMidZeroLobby", f"{arow(N, 0.0, 'lobby')['regret_middle_pct']:.1f}"))
    m(("AnaBestNineLobby", f"{arow(N, 0.9, 'lobby')['best']:.0f}")); m(("AnaWorstNineLobby", f"{arow(N, 0.9, 'lobby')['best'] * (1 + arow(N, 0.9, 'lobby')['regret_worst_pct'] / 100):.0f}"))
    m(("AnaBandZeroLobby", f"{arow(N, 0.0, 'lobby')['band_5pct'][0]}--{arow(N, 0.0, 'lobby')['band_5pct'][1]}"))
    m(("AnaBandZeroRandom", f"{arow(N, 0.0, 'random')['band_5pct'][0]}--{arow(N, 0.0, 'random')['band_5pct'][1]}"))
    m(("AnaFortyLobbyHalf", str(arow(40, 0.5, "lobby")["h_star"]))); m(("AnaFortyRandomZero", str(arow(40, 0.0, "random")["h_star"])))
    m(("AnaFortyRegretMidHalf", f"{arow(40, 0.5, 'lobby')['regret_middle_pct']:.0f}"))

    lob0 = next(b for b in lob if b["rate"] == 0.0); m(("TraValidErr", f"{100 * lob0['validation_rel_err']:.1f}"))
    m(("TraAlpha", f"{tra['alpha']}")); m(("TraN", str(tra["n_floors"]))); m(("TraReps", str(tra["reps"]))); m(("TraHorizon", f"{tra['horizon_s'] / 3600:.0f}"))
    for b in lob:
        tag = {0.0: "Zero", 0.5: "Half", 1.0: "One", 2.0: "Two", 4.0: "Four"}.get(b["rate"])
        if tag:
            m((f"TraH{tag}", str(b["h_star"]))); m((f"TraBest{tag}", f"{b['best']:.0f}")); m((f"TraRegMid{tag}", f"{b['regret_middle_pct']:.0f}"))
            m((f"TraRegWorst{tag}", f"{b['regret_worst_pct']:.0f}")); m((f"TraHp{tag}", str(b["h_star_p90"])))
    r2, r20 = hi["rows"][0], hi["rows"][-1]; r_mid = hi["rows"][len(hi["rows"]) // 2]
    m(("TraHiWaitLow", f"{r2['mean_wait']:.0f}")); m(("TraHiWaitTop", f"{r20['mean_wait']:.0f}")); m(("TraHiWaitMin", f"{min(r['mean_wait'] for r in hi['rows']):.0f}"))
    m(("TraHiWaitMinFloor", str(min(hi["rows"], key=lambda r: r["mean_wait"])["home"])))
    m(("TraHiRideLow", f"{r2['mean_ride']:.0f}")); m(("TraHiRideTop", f"{r20['mean_ride']:.0f}")); m(("TraHiStopsTop", f"{r20['mean_stops']:.1f}"))
    lastb = next(b for b in last if b["rate"] == 4.0); h = np.array(lastb["car_pos_hist"], float)
    m(("TraLastLobbyMass", f"{100 * h[0] / h.sum():.0f}")); m(("TraLastHFour", str(lastb["h_star"])))
    lasth = next(b for b in last if b["rate"] == 0.5); h = np.array(lasth["car_pos_hist"], float); m(("TraLastLobbyMassHalf", f"{100 * h[0] / h.sum():.0f}"))

    v0 = sta["variants"][0]; v1 = sta["variants"][1]
    m(("StaBestNone", f"{v0['best']:.0f}")); m(("StaBestOn", f"{v1['best']:.0f}")); m(("StaWalked", f"{100 * v1['frac_walked_overall']:.0f}"))
    m(("StaWalkedDownOnly", f"{100 * sta['variants'][-1]['frac_walked_overall']:.0f}")); m(("StaWalkedSym", f"{100 * sta['variants'][3]['frac_walked_overall']:.0f}"))
    m(("StaRegMidNone", f"{v0['regret_middle_pct']:.0f}")); m(("StaRegMidOn", f"{v1['regret_middle_pct']:.0f}")); m(("StaH", str(v1["h_star"])))
    from src.analytic.closed_form import curve as _curve
    from src.model.kinematics import Kinematics as _K
    from src.model.stairs import Stairs as _S
    k_ = _K.from_cfg(kin); cz = _curve(15, 0.0, k_, "lobby"); cs = _curve(15, 0.0, k_, "lobby", stairs=_S(15, 10, 3))
    csym = _curve(15, 0.0, k_, "lobby", stairs=_S(12, 12, 3)); cdo = _curve(15, 0.0, k_, "lobby", stairs=_S(15, 10, 0))
    m(("StaAnaZeroNone", str(cz["h_star"]))); m(("StaAnaZeroOn", str(cs["h_star"]))); m(("StaAnaZeroSym", str(csym["h_star"]))); m(("StaAnaZeroDown", str(cdo["h_star"])))

    for b in fle["blocks"]:
        tag = {1: "One", 2: "Two", 4: "Four"}.get(b["n_cars"])
        if tag:
            m((f"FleBest{tag}", f"{b['best']:.0f}")); m((f"FleRegMid{tag}", f"{b['regret_middle_pct']:.0f}")); m((f"FleH{tag}", str(b["h_star"])))
            m((f"FleBand{tag}", f"{b['band_5pct'][0]}--{b['band_5pct'][1]}"))

    m(("RobN", str(rob["n_floors"])))
    def g(al, rt):
        return next(x for x in rob["grid"] if x["alpha"] == al and x["rate"] == rt)
    m(("RobHZeroZero", str(g(0.0, 0.0)["h_star"]))); m(("RobHZeroFour", str(g(0.0, 4.0)["h_star"]))); m(("RobHQuarterZero", str(g(0.25, 0.0)["h_star"])))
    m(("RobHQuarterFour", str(g(0.25, 4.0)["h_star"]))); m(("RobHHalfZero", str(g(0.5, 0.0)["h_star"])))
    m(("RobRegMidZeroZero", f"{g(0.0, 0.0)['regret_middle_pct']:.0f}")); m(("RobRegMidHalfZero", f"{g(0.5, 0.0)['regret_middle_pct']:.0f}"))
    m(("RobRegMidNineOne", f"{g(0.9, 1.0)['regret_middle_pct']:.0f}")); m(("RobRegTopNineOne", f"{g(0.9, 1.0)['regret_top_pct']:.0f}"))
    m(("RobBandZeroZero", f"{g(0.0, 0.0)['band_5pct'][0]}--{g(0.0, 0.0)['band_5pct'][1]}")); m(("RobBandQuarterFour", f"{g(0.25, 4.0)['band_5pct'][0]}--{g(0.25, 4.0)['band_5pct'][1]}"))
    m(("RobBandWidthQuarterFour", str(g(0.25, 4.0)["band_5pct_width"])))
    n_floor2 = sum(1 for x in rob["grid"] if x["h_star"] == 2); m(("RobCellsFloorTwo", str(n_floor2))); m(("RobCells", str(len(rob["grid"]))))
    m(("RobHalfPlusAllTwo", "yes" if all(x["h_star"] == 2 for x in rob["grid"] if x["alpha"] >= 0.5) else "no"))
    for name, a in arch.items():
        T = name.capitalize()
        m((f"Arc{T}H", str(a["h_star"]))); m((f"Arc{T}Hp", str(a["h_star_p90"]))); m((f"Arc{T}Best", f"{a['best']:.0f}"))
        m((f"Arc{T}RegMid", f"{a['regret_middle_pct']:.0f}")); m((f"Arc{T}RegTop", f"{a['regret_top_pct']:.0f}"))
        m((f"Arc{T}Band", f"{a['band_5pct'][0]}--{a['band_5pct'][1]}")); m((f"Arc{T}N", str(a["n_floors"])))
        m((f"Arc{T}Walked", f"{100 * np.mean([r['frac_walked'] for r in a['rows']]):.0f}"))
    Path(args.macros).write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 6 figures")


if __name__ == "__main__":
    main()
