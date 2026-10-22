"""Figures + LaTeX macros for the interstellar-routes paper.

Reads results/{catalog,static,timedep,routing,exploration}.json; writes PDFs to
paper/figures and every quoted number to paper/macros.tex, so recomputing the
figures rewrites the prose.

Colours: categorical slots in fixed order from the project palette (blue,
orange, aqua, yellow, magenta), always paired with a line style or marker so
identity never rests on colour alone; magnitudes use one blue ramp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from utils.plotting import apply_style, col_fig, text_fig  # noqa: E402

S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK2, MUTED = "#52514e", "#a3a29c"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])


def _f(x, nd=1):
    return f"{x:.{nd}f}"


def _grp(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()
    res = Path(args.results)
    cat = json.loads((res / "catalog.json").read_text())
    st = json.loads((res / "static.json").read_text())
    td = json.loads((res / "timedep.json").read_text())
    rt = json.loads((res / "routing.json").read_text())
    ex = json.loads((res / "exploration.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    M: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Fig 1: the solar neighbourhood map (x-y projection, 8 pc)
    # ------------------------------------------------------------------ #
    xyz = np.array(st["map"]["xyz"])
    bott = np.array(st["map"]["bott"])
    fig, ax = col_fig(height_ratio=0.95)
    for a, b in st["map"]["edges"]:
        ax.plot(xyz[[a, b], 0], xyz[[a, b], 1], color=MUTED, lw=0.6, zorder=1)
    sc = ax.scatter(xyz[1:, 0], xyz[1:, 1], c=bott[1:], cmap=BLUES, s=9,
                    edgecolors="white", linewidths=0.4, vmin=1.2, vmax=2.3, zorder=2)
    # name a few landmarks (positions are ICRS x-y projections)
    d = np.linalg.norm(xyz, axis=1)
    for name, dd, off in (("$\\alpha$ Cen", 1.34, (-14, -16)), ("Sirius", 2.64, (4, 3))):
        k = int(np.argmin(np.abs(d - dd)))
        ax.annotate(name, xyz[k, :2], xytext=off, textcoords="offset points",
                    fontsize=6, color=INK2)
    ax.scatter([0], [0], marker="*", s=90, color=S2, edgecolors="white",
               linewidths=0.6, zorder=3, label="Sun")
    ax.set_aspect("equal")
    ax.set_xlim(-8.5, 8.5)
    ax.set_ylim(-8.5, 8.5)
    ax.set_xlabel("x (pc, ICRS)")
    ax.set_ylabel("y (pc, ICRS)")
    ax.set_title("Stars within 8 pc and their MST")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("bottleneck hop from the Sun (pc)")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "map.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: static percolation curves
    # ------------------------------------------------------------------ #
    r = np.array(st["r_fine"])
    cv = st["curves"]
    fig, ax = col_fig(height_ratio=0.7)
    ax.plot(r, cv["gaia_plus"], color=S1, lw=2, label="Gaia DR3 (GCNS) + supplement")
    ax.plot(r, cv["gaia_only"], color=S2, lw=1.5, ls="--", label="Gaia only")
    ax.plot(r, cv["bd_0.2"], color=S3, lw=1.5, ls="-.", label="+20% unseen brown dwarfs")
    ax.plot(r, cv["poisson"], color=INK2, lw=1.2, ls=":", label="Poisson, same density")
    ax.set_xlabel("maximum hop range $r$ (pc)")
    ax.set_ylabel("fraction of stars reachable")
    ax.set_title(f"Frozen stars, $\\leq${st['eval_radius_pc']} pc")
    ax.set_xlim(1.5, 2.8)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.36), ncol=2, frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "percolation.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: moving stars -- reach vs horizon, and r_q(T)
    # ------------------------------------------------------------------ #
    H = np.array(td["horizons_myr"])
    rg = np.array(td["r_grid"])
    frac = np.array(td["reach_frac"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 3.0))
    apply_style()
    show = [(0.75, S5, ":", "v"), (1.0, S4, "-.", "D"), (1.25, S3, "--", "s"),
            (1.5, S2, "-", "^"), (2.0, S1, "-", "o")]
    for rv, col, ls, mk in show:
        k = int(np.argmin(np.abs(rg - rv)))
        a1.plot(H * 1000, frac[k], color=col, ls=ls, marker=mk, ms=4, lw=1.6,
                label=f"$r$ = {rg[k]:g} pc")
    a1.set_xlabel("horizon $T$ (kyr)")
    a1.set_ylabel("fraction of stars reached")
    a1.set_title("Waiting for stars to drift into range")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    r50 = np.array(td["r_q_of_T"]["0.5"], dtype=float)
    ci = np.array(td["r50_ci95"])
    a2.fill_between(H * 1000, ci[:, 0], ci[:, 1], color=S1, alpha=0.18, lw=0)
    a2.plot(H * 1000, r50, color=S1, marker="o", ms=4, lw=2, label="50% of stars")
    for q, col, ls, mk in (("0.1", S3, "--", "s"), ("0.9", S2, "-.", "^")):
        a2.plot(H * 1000, np.array(td["r_q_of_T"][q], dtype=float), color=col, ls=ls,
                marker=mk, ms=4, lw=1.4, label=f"{int(float(q) * 100)}% of stars")
    a2.axhline(r50[0], color=MUTED, lw=0.8, ls=":")
    a2.text(H[-1] * 1000, r50[0] - 0.04, "frozen stars", ha="right", va="top",
            fontsize=6, color=INK2)
    a2.set_xlabel("horizon $T$ (kyr)")
    a2.set_ylabel("hop range needed (pc)")
    a2.set_title("Range needed to reach a fraction of stars")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "moving.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: per-hop survival time needed to percolate, by technology
    # ------------------------------------------------------------------ #
    rows = [t for t in rt["techs"] if t["mass_ratio"] == 10]
    names = [t["tech"] for t in rows]
    ys = np.arange(len(rows))
    fig, ax = col_fig(height_ratio=0.62)
    ax.barh(ys + 0.19, [t["tau_star_static_yr"] for t in rows], height=0.36,
            color=S1, label="frozen stars")
    if rows[0].get("tau_star_moving_yr"):
        ax.barh(ys - 0.19, [t["tau_star_moving_yr"] for t in rows], height=0.36,
                color=S2, hatch="////", edgecolor="white", linewidth=0,
                label=f"moving stars, wait $\\leq${rt['r50_moving']['T_myr'] * 1000:.0f} kyr")
    ax.set_xscale("log")
    ax.set_xlim(5, 2e6)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("hop time the ship must survive (yr)")
    ax.set_title("Survival per hop to reach half the stars (mass ratio 10)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.36), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "propulsion.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: refuelling at exoplanet hosts; photon sails vs rockets
    # ------------------------------------------------------------------ #
    fu = rt["fuel"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.4))
    apply_style()
    tk = [x["tank_pc"] for x in fu["reach_frac_by_tank"]]
    a1.plot(tk, [x["frac"] for x in fu["reach_frac_by_tank"]], color=S1, marker="o",
            ms=4, lw=2, label=f"refuel at {fu['n_hosts']} exoplanet hosts")
    a1.axvline(fu["refuel_anywhere_r50"], color=S2, ls="--", lw=1.4,
               label="refuel anywhere (50%)")
    a1.set_xlabel("tank range between refuels (pc)")
    a1.set_ylabel("fraction of stars reachable")
    a1.set_title(f"Refuelling only at known planet hosts ($\\leq${fu['radius_pc']} pc)")
    a1.legend(loc="lower right", frameon=False)
    sl = rt["sail"]["by_lifetime"]
    tau = [x["tau_yr"] for x in sl]
    a2.plot(tau, [x["sail"]["reach"] for x in sl], color=S1, marker="o", ms=4, lw=2,
            label="photon sail (brakes on target light)")
    a2.plot(tau, [x["rocket_same_speed"]["reach"] for x in sl], color=S2, marker="s",
            ms=4, lw=1.4, ls="--", label=f"rocket at {rt['sail']['v_launch_c']:g} c")
    a2.set_xscale("log")
    a2.set_xlabel("hop time the craft survives (yr)")
    a2.set_ylabel("fraction of stars reachable")
    a2.set_title("Sails need bright targets to stop")
    a2.set_ylim(-0.05, 1.12)
    a2.legend(loc="center left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "routing.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 6: exploration coverage, NN (Bjork) vs 2-opt
    # ------------------------------------------------------------------ #
    t = np.array(ex["t_grid_yr"]) / 1000
    fig, ax = col_fig(height_ratio=0.68)
    for fl, col, mk in ((ex["fleets"][0], S1, "o"), (ex["fleets"][2], S2, "s")):
        ax.plot(t, fl["coverage_opt"], color=col, lw=2,
                label=f"{fl['n_probes']} probe{'s' if fl['n_probes'] > 1 else ''}, 2-opt")
        ax.plot(t, fl["coverage_nn"], color=col, lw=1.3, ls="--",
                label=f"{fl['n_probes']} probe{'s' if fl['n_probes'] > 1 else ''}, nearest-neighbour")
    ax.set_xlabel(f"time at {ex['v_probe_c']:g} c (kyr)")
    ax.set_ylabel("fraction of stars visited")
    ax.set_title(f"Visiting all {ex['n_stars']} stars within {ex['radius_pc']} pc")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "exploration.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    rq, ci95 = st["r_quantiles"], st["r_quantiles_ci95"]
    M["NStars"] = _grp(cat["n_stars"])
    M["NWithRV"] = _grp(cat["n_with_rv"])
    M["PctRV"] = f"{100 * cat['n_with_rv'] / cat['n_stars']:.0f}"
    M["NHosts"] = _grp(cat["n_hosts"])
    M["CatRadius"] = f"{cat['radius_pc']}"
    M["SpeedMedian"] = f"{cat['speed_kms_median']:.0f}"
    M["Density"] = f"{st['density_pc3']:.3f}"
    M["EvalRadius"] = f"{st['eval_radius_pc']}"
    M["NEval"] = _grp(st["n_eval"])
    M["NNMedian"] = _f(st["nn_median_pc"], 2)
    M["SunFirstHop"] = _f(st["sun_first_hop_pc"], 2)
    M["RTen"] = _f(rq["0.1"], 2)
    M["RFifty"] = _f(rq["0.5"], 2)
    M["RNinety"] = _f(rq["0.9"], 2)
    M["RNinetyNine"] = _f(rq["0.99"], 2)
    M["RFiftyLo"] = _f(ci95["0.5"][0], 2)
    M["RFiftyHi"] = _f(ci95["0.5"][1], 2)
    M["RFiftyGaiaOnly"] = _f(st["r_quantiles_gaia_only"]["0.5"], 2)
    M["RFiftyPoisson"] = _f(st["r_quantiles_poisson"]["0.5"], 2)
    M["RcPoisson"] = _f(st["r_c_poisson_analytic"], 2)
    M["PctAbovePoisson"] = f"{100 * (rq['0.5'] / st['r_quantiles_poisson']['0.5'] - 1):.0f}"
    M["PocketGaiaSize"] = f"{st['pocket_gaia_only_size'] - 1}"
    M["PocketBridge"] = _f(st["bridge_gaia_only"][0]["length_pc"], 2)
    bd = {b["frac"]: b for b in st["brown_dwarf_sweep"]}
    M["RFiftyBDTwenty"] = _f(bd[0.2]["0.5"], 2)
    M["PctBDTwenty"] = f"{100 * (1 - bd[0.2]['0.5'] / rq['0.5']):.0f}"
    M["RFiftyBDThirty"] = _f(bd[0.3]["0.5"], 2)
    M["StretchMedian"] = _f(st["route_stretch_median"], 1)
    M["HopsMedian"] = f"{st['route_hops_median']:.0f}"
    sw = st["radius_sweep"]
    M["RFiftySweepMin"] = _f(min(x["r50"] for x in sw), 2)
    M["RFiftySweepMax"] = _f(max(x["r50"] for x in sw), 2)

    red = td["r50_reduction"]
    rci = td["r50_reduction_ci95"]
    Hl = td["horizons_myr"]

    def at(T):
        return Hl.index(T)

    for T, name in ((0.1, "Hundred"), (0.2, "TwoHundred"), (0.5, "FiveHundred"), (1.0, "Thousand")):
        if T in Hl:
            k = at(T)
            M[f"Red{name}"] = f"{100 * red[k]:.0f}"
            M[f"RedLo{name}"] = f"{100 * rci[k][0]:.0f}" if rci else ""
            M[f"RedHi{name}"] = f"{100 * rci[k][1]:.0f}" if rci else ""
            M[f"RFiftyT{name}"] = _f(td["r_q_of_T"]["0.5"][k], 2)
    M["TdEval"] = f"{td['eval_radius_pc']}"
    M["TdNEval"] = _grp(td["n_eval"])
    M["TdPairs"] = _grp(td["n_candidate_pairs"])
    M["TdNBoot"] = f"{td['n_boot']}"
    M["TdVship"] = f"{td['v_ship_c']:g}"
    M["SweepR"] = f"{td['sweep_r_pc']:g}"
    kf = {round(f["r"], 3): f for f in td["front"]}
    if kf.get(1.0) and kf[1.0]["v_eff_pc_myr"]:
        M["FrontOne"] = f"{kf[1.0]['v_eff_pc_myr']:.0f}"
        M["FrontOneKms"] = f"{kf[1.0]['v_eff_pc_myr'] / 1.0227:.0f}"
        M["GalaxyCrossGyrOne"] = _f(30_000 / kf[1.0]["v_eff_pc_myr"] / 1000, 1)
    if kf.get(1.5) and kf[1.5]["v_eff_pc_myr"]:
        M["FrontOneHalf"] = f"{kf[1.5]['v_eff_pc_myr']:.0f}"
        M["GalaxyCrossGyrOneHalf"] = _f(30_000 / kf[1.5]["v_eff_pc_myr"] / 1000, 1)
    k2 = at(0.2)
    ss = {x["settle_myr"]: x["frac"][k2] for x in td["settle_sweep"]}
    M["SettleZero"] = f"{100 * ss[0.0]:.0f}"
    M["SettleTen"] = f"{100 * ss[0.01]:.0f}"
    M["SettleHundred"] = f"{100 * ss[0.1]:.0f}"
    vs = {x["v_c"]: x["frac"][k2] for x in td["speed_sweep"]}
    M["SpeedSlow"] = f"{100 * vs[0.001]:.0f}"
    M["SpeedMid"] = f"{100 * vs[0.01]:.0f}"
    M["SpeedFast"] = f"{100 * vs[0.1]:.0f}"
    ret = td["retention"][0]["ratio"]
    M["RetFiveHundred"] = f"{100 * ret[at(0.5)]:.0f}"
    M["RetThousand"] = f"{100 * ret[at(1.0)]:.0f}"

    tech = {(t["tech"], t["mass_ratio"]): t for t in rt["techs"]}
    M["VFusion"] = _grp(tech[("fusion", 10)]["v_cruise_kms"])
    M["VFusionPctC"] = _f(100 * tech[("fusion", 10)]["v_cruise_c"], 1)
    M["TauFusionStatic"] = f"{tech[('fusion', 10)]['tau_star_static_yr']:.0f}"
    M["TauFusionMoving"] = f"{tech[('fusion', 10)].get('tau_star_moving_yr', float('nan')):.0f}"
    M["TauChemStatic"] = _grp(tech[("chemical", 10)]["tau_star_static_yr"])
    M["TauElecStatic"] = _grp(tech[("electric", 10)]["tau_star_static_yr"])
    M["TauAntiStatic"] = f"{tech[('antimatter', 10)]['tau_star_static_yr']:.0f}"
    M["MovingTkyr"] = f"{rt['r50_moving']['T_myr'] * 1000:.0f}" if rt["r50_moving"] else ""
    pil = {p["r_pc"]: p for p in rt["pilots"]}
    M["GreedyTwoFive"] = f"{100 * pil[2.5]['greedy_success']:.0f}"
    M["GreedyThree"] = f"{100 * pil[3.0]['greedy_success']:.0f}"
    M["GreedyFour"] = f"{100 * pil[4.0]['greedy_success']:.1f}"
    M["AstarRatioTwoFive"] = f"{100 * pil[2.5]['astar_expansion_ratio_median']:.0f}"
    M["AstarRatioFour"] = f"{100 * pil[4.0]['astar_expansion_ratio_median']:.0f}"
    M["OptStretchTwoFive"] = _f(pil[2.5]["optimal_stretch_median"], 2)
    M["NHostsFuel"] = _grp(fu["n_hosts"])
    M["NStarsFuel"] = _grp(fu["n_stars"])
    M["FuelRadius"] = f"{fu['radius_pc']}"
    M["FuelHop"] = _f(fu["hop_pc"], 0)
    M["TankFifty"] = _f(fu["need_quantiles"]["0.5"], 1)
    M["TankNinety"] = _f(fu["need_quantiles"]["0.9"], 1)
    M["RefuelAnyFifty"] = _f(fu["refuel_anywhere_r50"], 2)
    M["TankRatio"] = _f(fu["need_quantiles"]["0.5"] / fu["refuel_anywhere_r50"], 1)
    sa = rt["sail"]
    bysl = {x["tau_yr"]: x for x in sa["by_lifetime"]}
    M["SailVLaunch"] = f"{sa['v_launch_c']:g}"
    M["SailBrakeMedian"] = _grp(sa["brake_speed_kms_quantiles"]["0.5"])
    M["SailBrakeNinety"] = _grp(sa["brake_speed_kms_quantiles"]["0.9"])
    M["SailFracFast"] = f"{100 * sa['frac_targets_brake_above_0p01c']:.0f}"
    M["SailReachThousand"] = f"{100 * bysl[1000]['sail']['reach']:.0f}"
    M["SailReachThreeThousand"] = f"{100 * bysl[3000]['sail']['reach']:.0f}"
    if bysl[3000]["sail"]["t50_yr"]:
        M["SailTFifty"] = _grp(bysl[3000]["sail"]["t50_yr"])
        M["RocketTFifty"] = _grp(bysl[3000]["rocket_same_speed"]["t50_yr"])

    fl = {f["n_probes"]: f for f in ex["fleets"]}
    M["ExRadius"] = f"{ex['radius_pc']}"
    M["ExNStars"] = _grp(ex["n_stars"])
    imp = [f["improvement"] for f in ex["fleets"]]
    M["TwoOptMin"] = f"{100 * min(imp):.0f}"
    M["TwoOptMax"] = f"{100 * max(imp):.0f}"
    M["NNGapOne"] = f"{100 * fl[1]['nn_gap_to_bound']:.0f}"
    M["OptGapOne"] = f"{100 * fl[1]['opt_gap_to_bound']:.0f}"
    M["MakespanOneKyr"] = _f(fl[1]["opt_makespan_yr"] / 1000, 1)
    k8 = 8 if 8 in fl else max(fl)
    M["MakespanEightKyr"] = _f(fl[k8]["opt_makespan_yr"] / 1000, 1)
    M["FleetBig"] = f"{k8}"

    Path(args.macros).write_text(
        "% Auto-generated by src/figures/plot_routes.py -- do not edit by hand.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
    print(f"[figures] wrote 6 figures to {out} and {len(M)} macros to {args.macros}")


if __name__ == "__main__":
    main()
