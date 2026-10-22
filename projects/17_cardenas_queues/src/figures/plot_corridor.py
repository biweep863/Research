"""Figures + LaTeX macros for the Lázaro Cárdenas queueing paper.

Reads results/{inventory,analytic,validation,peak,montecarlo,interventions}.json,
writes seven PDFs to paper/figures and every quoted number to paper/macros.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from utils.plotting import COL_W, TEXT_W, apply_style  # noqa: E402

# Validated categorical palette, fixed order (blue, orange, aqua, yellow, magenta).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2 = "#0b0b0b", "#52514e"
CLASS_COLOR = {"major": C[0], "tertiary": C[1], "minor": C[2], "service": C[3], "retorno": C[4]}
DIR_COLOR = {"WB": C[0], "EB": C[1]}
DIR_NAME = {"WB": "westbound (to San Pedro)", "EB": "eastbound (to Garza Sada)"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()
    res = Path(args.results)
    L = {k: json.loads((res / f"{k}.json").read_text())
         for k in ("inventory", "analytic", "validation", "peak", "montecarlo", "interventions", "daily")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macros: list[str] = []

    def m(name, value, fmt="{:.0f}"):
        if value is None:
            value = "--"
        macros.append(f"\\newcommand{{\\{name}}}{{{fmt.format(value) if not isinstance(value, str) else value}}}")

    inv, ana, val, peak, mc, itv = (L[k] for k in ("inventory", "analytic", "validation", "peak",
                                                   "montecarlo", "interventions"))
    km0 = inv["segments"]["WB"][0]["c0"] / 1000 - 0.05

    # ------------------------------------------------------------------ #
    # Fig 1: inventory along the corridor
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W, TEXT_W * 0.48), sharex=True)
    for ax, d in zip(axes, ("WB", "EB")):
        ch = np.asarray(ana["dirs"][d]["chainage_m"]) / 1000
        ln = np.asarray(ana["dirs"][d]["lanes"])
        ax.step(ch, ln, where="mid", color=INK, lw=1.2, label="corridor lanes (express + lateral)")
        for cls, col in CLASS_COLOR.items():
            xs = [a["chainage"] / 1000 for a in inv["approaches_list"] if a["dir"] == d and a["dclass"] == cls]
            ys = [a["lanes"] for a in inv["approaches_list"] if a["dir"] == d and a["dclass"] == cls]
            if xs:
                ax.vlines(xs, 0, -np.asarray(ys), color=col, lw=1.4, label=f"{cls} feeder")
        for j in inv["junctions"]:
            ax.axvline(j["chainage_km"], color=INK2, lw=0.4, ls=":")
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_ylabel(f"{d}\nlanes  /  feeder lanes")
        ax.set_ylim(-4.5, 8)
        ax.set_yticks([-4, -2, 0, 2, 4, 6])
        ax.set_yticklabels(["4", "2", "0", "2", "4", "6"])
    for j in inv["junctions"]:
        axes[0].text(j["chainage_km"], 8.2, j["name"].replace("Avenida ", "Av. "), rotation=40, fontsize=5.5,
                     ha="left", va="bottom", color=INK2)
    axes[1].set_xlabel("chainage from the Garza Sada end (km)")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.95), ncol=3, frameon=False)
    fig.savefig(out / "inventory.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: steady-state utilisation
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.62))
    for d in ("WB", "EB"):
        a = ana["dirs"][d]
        ch = np.asarray(a["chainage_m"]) / 1000
        ax.plot(ch, a["rho"], color=DIR_COLOR[d], label=DIR_NAME[d])
        sc = a["signal_cells"]
        ax.plot(ch[sc], np.asarray(a["rho"])[sc], "o", ms=3.5, color=DIR_COLOR[d], mec="#fcfcfb", mew=0.8)
    ax.axhline(1, color=INK, lw=0.8, ls="--")
    ax.text(0.2, 1.02, r"$\rho = 1$", fontsize=7, color=INK)
    ax.set_xlabel("chainage (km)")
    ax.set_ylabel(r"utilisation $\rho$ = flow / capacity")
    ax.set_ylim(0, 1.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False, fontsize=6)
    fig.savefig(out / "utilisation.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: space-time density (CTM, base parameters)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.36), sharey=True)
    for ax, d in zip(axes, ("WB", "EB")):
        b = peak["base"][d]
        k = np.asarray(b["density"])
        ch = np.asarray(b["chainage_m"]) / 1000
        if d == "EB":                      # plot in map orientation (km increasing)
            k, ch = k[:, ::-1], ch[::-1]
        t = np.arange(len(k)) * b["record_dt_h"]
        im = ax.pcolormesh(ch, t, k, cmap=BLUES, vmin=0, vmax=150, shading="nearest", rasterized=True)
        ax.set_title(f"{DIR_NAME[d]}", fontsize=8)
        ax.set_xlabel("chainage (km)")
        ax.annotate("", xy=(0.95 if d == "WB" else 0.05, 1.02), xytext=(0.75 if d == "WB" else 0.25, 1.02),
                    xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
        ax.invert_yaxis()
    axes[0].set_ylabel("time (h)")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("density (veh/km/lane)")
    fig.savefig(out / "spacetime.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: validation (two panels, each its own single axis)
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, COL_W * 0.55))
    ax = axes[0]
    for i, c in enumerate(sorted({q["c"] for q in val["queues"]})):
        for svc, mk in (("exp", "o"), ("det", "s")):
            rows = [q for q in val["queues"] if q["c"] == c and q["service"] == svc]
            ax.loglog([q["theory_s"] for q in rows], [q["sim_s"] for q in rows], mk, ms=3.5, color=C[i],
                      mec="#fcfcfb", mew=0.6, label=f"c={c}" if svc == "exp" else None)
    lim = [0.02, 30]
    ax.plot(lim, lim, color=INK2, lw=0.6)
    ax.set_xlabel("theory $W_q$ (s)")
    ax.set_ylabel("simulated $W_q$ (s)")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_title("lanes as servers", fontsize=7)
    ax = axes[1]
    x = [s["x"] for s in val["signal"]]
    ax.plot(x, [s["sim_s"] for s in val["signal"]], "o-", ms=3.5, color=C[0], label="simulated")
    ax.plot(x, [s["webster_s"] for s in val["signal"]], "--", color=C[1], label="Webster")
    ax.set_xlabel("degree of saturation $x$")
    ax.set_ylabel("delay (s/veh)")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("fixed-time signal", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "validation.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: which predictor finds the bottleneck
    # ------------------------------------------------------------------ #
    preds = [("lane_ratio", "feeder lanes /\nlanes (local)"), ("lane_ratio_cum", "cumulative\nlane ratio"),
             ("feeder_cum", "feeder lanes\nalone"), ("inv_capacity", "capacity\nalone"), ("rho", r"$\rho$")]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.62))
    w = 0.38
    xs = np.arange(len(preds))
    for i, (scen, lab) in enumerate((("realistic", "OSM geometry"), ("gap_filled", "lateral gap filled"))):
        v = [mc["summary"][scen][f"{k}_hit"] * 100 for k, _ in preds]
        bars = ax.bar(xs + (i - 0.5) * w, v, w * 0.92, color=C[i], label=lab)
        for bar, hv in zip(bars, v):
            ax.text(bar.get_x() + bar.get_width() / 2, hv + 1.5, f"{hv:.0f}", ha="center", va="bottom",
                    fontsize=5.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([p[1] for p in preds], fontsize=6)
    ax.set_ylabel("hit rate (%)")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out / "predictors.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 6: delay vs demand scale and peak length
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, COL_W * 0.6), sharey=True)
    lens = sorted({g["peak_len_h"] for g in peak["grid"]})
    for ax, d in zip(axes, ("WB", "EB")):
        for i, Lh in enumerate(lens):
            rows = sorted([g for g in peak["grid"] if g["dir"] == d and g["peak_len_h"] == Lh],
                          key=lambda g: g["scale"])
            ax.plot([g["scale"] for g in rows], [g["delay_per_vehicle_s"] / 60 for g in rows], "o-", ms=3,
                    color=C[i], label=f"{Lh:g} h peak")
        ax.set_title(d, fontsize=7)
        ax.set_xlabel("demand scale")
    axes[0].set_ylabel("mean delay per vehicle (min)")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "demand_grid.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 7: interventions
    # ------------------------------------------------------------------ #
    names = [("metering", "ramp metering\n(ALINEA)"), ("consolidate", "consolidate\nminor accesses"),
             ("extra_lane", "+1 lane at\nbottleneck"), ("retime", "retime\nsignals")]
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))
    for i, (k, lab) in enumerate(names):
        rows = [r[f"{k}_change"] * 100 for r in itv["draws"]]
        q1, q3 = np.percentile(rows, [25, 75])
        med = np.median(rows)
        ax.hlines(i, q1, q3, color=C[0], lw=2.0)
        ax.plot(med, i, "o", ms=6, color=C[0], mec="#fcfcfb", mew=1.0)
        jitter = (np.random.default_rng(i).random(len(rows)) - 0.5) * 0.3
        ax.plot(np.clip(rows, -100, 150), i + 0.28 + jitter * 0.3, ".", ms=2.5, color=INK2, alpha=0.5)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[1] for n in names], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("change in total delay vs. baseline (%)")
    ax.set_xlim(-105, 155)
    fig.savefig(out / "interventions.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 8: a working day (two stacked panels, each its own single axis)
    # ------------------------------------------------------------------ #
    day = L["daily"]
    h0 = day["start_h"]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, COL_W * 0.95), sharex=True)
    hrs = np.arange(day["hours"]) + h0 + 0.5
    for d in ("WB", "EB"):
        b = day["base"][d]
        axes[0].plot(hrs, b["rho_h"], "o-", ms=3, color=DIR_COLOR[d], label=DIR_NAME[d])
        t = h0 + np.arange(len(b["queue_km_t"])) * b["record_dt_h"]
        axes[1].plot(t, b["queue_km_t"], color=DIR_COLOR[d])
    axes[0].axhline(1, color=INK, lw=0.8, ls="--")
    axes[0].set_ylabel(r"max utilisation $\rho$")
    axes[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2)
    axes[1].set_ylabel("queue length (km)")
    axes[1].set_xlabel("hour of day")
    axes[1].set_xticks(range(h0, h0 + day["hours"] + 1, 2))
    fig.savefig(out / "daily.pdf")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    lc = inv["llm_claims"]
    m("InvLengthKm", inv["length_km"], "{:.1f}")
    m("InvApproaches", inv["approaches"])
    m("InvApproachesNoService", inv["approaches_no_service"])
    m("InvServiceApproaches", inv["approaches_by_class"].get("service", 0))
    m("InvAccessesNoService", inv["accesses_no_service"])
    m("InvFeederLanes", inv["feeder_lanes"])
    m("InvFeederLanesNoService", inv["feeder_lanes_no_service"])
    m("InvRetornos", inv.get("retornos", 0))
    for k in ("major", "tertiary", "minor", "service", "retorno"):
        m(f"InvLanes{k.capitalize()}", inv["feeder_lanes_by_class"][k])
        m(f"InvN{k.capitalize()}", inv["approaches_by_class"].get(k, 0))
    m("InvMainlineLanes", inv["mainline"]["WB"]["total_median"])
    m("InvExpressLanes", inv["mainline"]["WB"]["express_median"])
    m("InvMainlineMin", min(inv["mainline"][d]["total_min"] for d in ("WB", "EB")))
    m("InvMainlineMax", max(inv["mainline"][d]["total_max"] for d in ("WB", "EB")))
    m("InvLaneRatio", inv["lane_ratio_all"], "{:.0f}")
    m("InvLaneRatioNoService", inv["lane_ratio_no_service"], "{:.0f}")
    m("InvTaggedPct", inv["approach_lanes_tagged_frac"] * 100)
    m("InvCorridorTaggedPct", inv["corridor_lanes_tag_coverage"] * 100)
    m("InvInternalLinks", inv["internal_links"])
    m("InvSignalsOSM", inv["signals_osm"])
    m("InvSignalsWB", inv["signals_model"]["WB"])
    m("InvSignalsEB", inv["signals_model"]["EB"])
    m("InvJunctions", len(inv["junctions"]))
    m("InvCrossing", inv["approaches_by_kind"].get("crossing", 0))
    m("InvTerminal", inv["approaches_by_kind"].get("terminal", 0))
    m("InvFetched", (inv.get("osm_fetched") or "synthetic")[:10])
    m("LLMAccLo", lc["accesses"][0]); m("LLMAccHi", lc["accesses"][1])
    m("LLMLanesLo", lc["feeder_lanes"][0]); m("LLMLanesHi", lc["feeder_lanes"][1])
    m("LLMMainLo", lc["mainline_lanes_per_dir"][0]); m("LLMMainHi", lc["mainline_lanes_per_dir"][1])

    th = ana["theta"]
    m("ThQMajor", th["q_major"]); m("ThQTertiary", th["q_tertiary"]); m("ThQMinor", th["q_minor"])
    m("ThQService", th["q_service"]); m("ThUp", th["upstream_vph"]); m("ThG", th["g_main"], "{:.2f}")
    m("ThTurnCross", th["turn_crossing"], "{:.2f}"); m("ThFriction", th["friction"] * 100, "{:.1f}")
    m("ThFifo", th["fifo"], "{:.1f}"); m("ThDrop", th["cap_drop"] * 100)
    for d in ("WB", "EB"):
        a = ana["dirs"][d]
        m(f"Ana{d}MaxRho", a["max_rho"], "{:.2f}")
        m(f"Ana{d}BottleneckKm", a["bottleneck_chainage_m"] / 1000, "{:.1f}")
        m(f"Ana{d}SideVph", round(a["side_demand_vph"], -2), "{:,.0f}")
        m(f"Ana{d}SideLanes", a["side_lanes"])
        m(f"Ana{d}PerLane", a["mean_vph_per_feeder_lane"])
        m(f"Ana{d}PerLanePct", a["mean_vph_per_feeder_lane"] / 1900 * 100)
        m(f"Ana{d}SatVph", round(a["side_at_saturation_vph"], -3), "{:,.0f}")
        m(f"Ana{d}PeakFlow", round(a["peak_flow_max_vph"], -2), "{:,.0f}")
        b = peak["base"][d]
        m(f"Peak{d}Delay", round(max(b["delay_total_vh"], 0), -1), "{:,.0f}")
        m(f"Peak{d}PerVeh", b["delay_per_vehicle_s"] / 60, "{:.1f}")
        m(f"Peak{d}QueueKm", b["max_congested_km"], "{:.1f}")
        m(f"Peak{d}HeadKm", b["bottleneck_km_ctm"], "{:.1f}")
        m(f"Peak{d}Fluid", round(b["fluid_bottleneck_delay_vh"], -1), "{:,.0f}")
        fb = peak["fifo_bounds"][d]
        m(f"Peak{d}FifoLo", round(fb["0.0"]["delay_total_vh"], -1), "{:,.0f}")
        m(f"Peak{d}FifoHi", round(fb["1.0"]["delay_total_vh"], -1), "{:,.0f}")
        g = peak["gap_filled"][d]
        m(f"Gap{d}Delay", round(g["delay_total_vh"], -1), "{:,.0f}")
        m(f"Gap{d}MaxRho", g["max_rho"], "{:.2f}")
        m(f"Gap{d}HeadKm", g["bottleneck_km_ctm"] if g["bottleneck_km_ctm"] else 0, "{:.1f}")
        # Demand at which congestion first appears (2 h peak).
        rows = sorted([x for x in peak["grid"] if x["dir"] == d and x["peak_len_h"] == 2.0], key=lambda x: x["scale"])
        first = next((x["scale"] for x in rows if x["delay_total_vh"] > 1), float("nan"))
        m(f"Grid{d}Onset", first, "{:.1f}")
        hi = rows[-1]
        m(f"Grid{d}TopPerVeh", hi["delay_per_vehicle_s"] / 60, "{:.0f}")
        m(f"Grid{d}TopScale", hi["scale"], "{:.1f}")

    v = val
    m("ValErr", v["mean_abs_rel_err"] * 100, "{:.1f}")
    m("ValErrMax", v["max_abs_rel_err"] * 100, "{:.0f}")
    m("ValSigErr", v["signal_mean_abs_rel_err_webster"] * 100, "{:.0f}")
    m("ValCtmFluidErr", v["ctm_fluid_mean_abs_rel_err"] * 100, "{:.2f}")
    m("ValN", len(v["queues"]))

    s, gfs, eq = mc["summary"]["realistic"], mc["summary"]["gap_filled"], mc["summary"]["equal_rates"]
    m("McDraws", mc["draws"])
    m("McRuns", s["n"]); m("McCongested", s["n_congested"]); m("McCongestedPct", s["n_congested"] / s["n"] * 100)
    for key, nm in (("lane_ratio", "LR"), ("lane_ratio_cum", "LRC"), ("feeder_cum", "FC"),
                    ("inv_capacity", "Cap"), ("rho", "Rho")):
        m(f"McHit{nm}", s[f"{key}_hit"] * 100)
        m(f"McGapHit{nm}", gfs[f"{key}_hit"] * 100)
        m(f"McEqHit{nm}", eq[f"{key}_hit"] * 100)
        m(f"McSp{nm}", s[f"{key}_rho_active"]["mean"], "{:.2f}")
    m("McGapCongested", gfs["n_congested"]); m("McEqCongested", eq["n_congested"])
    m("McAcc", s["rho_classifier_acc"] * 100)
    m("McAuc", s["rho_auc"], "{:.2f}")
    m("McDelayMedian", round(s["delay_vh_median"], -1), "{:,.0f}")
    m("McGapDelayMedian", round(gfs["delay_vh_median"], -1), "{:,.0f}")
    m("McFalsified", "no" if not mc["hypothesis_falsified"] else "yes")
    wb_share = max(mc["where"]["WB"]["share"]); eb_share = max(mc["where"]["EB"]["share"])
    m("McWhereWB", wb_share * 100); m("McWhereEB", eb_share * 100)

    m("ItvN", itv["n_congested_draws"])
    for k, nm in (("metering", "Met"), ("consolidate", "Con"), ("extra_lane", "Lane"), ("retime", "Ret")):
        sm = itv["summary"][k]
        m(f"Itv{nm}Median", sm["median"] * 100, "{:+.1f}" if abs(sm["median"]) < 0.01 else "{:+.0f}")
        m(f"Itv{nm}Better", sm["frac_better"] * 100)
        m(f"Itv{nm}Worse", sm["frac_worse"] * 100)
        for d in ("WB", "EB"):
            if d in itv["base"]:
                m(f"Itv{nm}Base{d}", itv["base"][d][f"{k}_change"] * 100, "{:+.0f}")

    for fc in peak.get("field_counts", []):
        m("FcName", fc["name"].replace("Avenida ", "Av. "))
        m("FcDir", fc["dir"])
        m("FcLo", round(fc["vph_lo"], -1)); m("FcHi", round(fc["vph_hi"], -1))
        m("FcKm", fc["chainage_m"] / 1000, "{:.2f}")
        m("FcAssumed", round(fc.get("assumed_vph", 0), -1), "{:,.0f}")
        break
    if "no_counts" in peak:
        for d in ("WB", "EB"):
            nc = peak["no_counts"][d]
            m(f"Nc{d}MaxRho", nc["max_rho"], "{:.2f}")
            m(f"Nc{d}Delay", round(nc["delay_total_vh"], -1), "{:,.0f}")
            m(f"Nc{d}QueueKm", nc["max_congested_km"], "{:.1f}")
    for d in ("WB", "EB"):
        recs = [r for r in mc["records"]["realistic"] if r["dir"] == d]
        m(f"McCong{d}", 100 * sum(r["scored"] for r in recs) / len(recs))
    for d in ("WB", "EB"):
        b, mcd = day["base"][d], day["mc"][d]
        m(f"Day{d}MaxRho", max(b["rho_h"]), "{:.2f}")
        m(f"Day{d}RhoSeven", b["rho_h"][0], "{:.2f}")
        m(f"Day{d}QueueKm", b["max_queue_km"], "{:.1f}")
        m(f"Day{d}CongH", b["congested_h"], "{:.1f}")
        m(f"Day{d}Delay", round(max(b["delay_vh"], 0), -1), "{:,.0f}")
        m(f"Day{d}Cap", round(b["cap_bottleneck"], -1), "{:,.0f}")
        hh = lambda x: "--" if x is None else f"{int(x)}:{int(round((x % 1) * 60)) % 60:02d}"  # noqa: E731
        m(f"Day{d}Start", hh(b["queue_start_h"]))
        m(f"Day{d}End", hh(b["queue_end_h"]))
        end_day = h0 + day["hours"]
        open_end = b["queue_end_h"] is not None and b["queue_end_h"] >= end_day - 0.05
        m(f"Day{d}Clears", f"is still there at {end_day}:00, the end of the simulated day" if open_end
          else (f"clears at {hh(b['queue_end_h'])}" if b["queue_end_h"] is not None else "never forms"))
        m(f"Day{d}PQueue", mcd["p_any_queue"] * 100)
        m(f"Day{d}CongHMed", mcd["congested_h"]["median"], "{:.1f}")
    m("DayDraws", day.get("draws", 0))
    for fc in peak.get("field_counts", []):
        m(f"Fc{fc['dir']}Lo", round(fc["vph_lo"], -1)); m(f"Fc{fc['dir']}Hi", round(fc["vph_hi"], -1))
        m(f"Fc{fc['dir']}Assumed", round(fc.get("assumed_vph", 0), -1), "{:,.0f}")
    m("InvApproachesAll", inv["approaches"])
    eb = itv["base"].get("EB")
    if eb:
        m("ItvBaseEBMain", round(eb["base_main_vh"], -1), "{:,.0f}")
        m("ItvBaseEBRamp", round(eb["base_ramp_vh"], -1), "{:,.0f}")
        m("ItvMetEBMain", round(eb["metering_main_vh"], -1), "{:,.0f}")
        m("ItvMetEBRamp", round(eb["metering_ramp_vh"], -1), "{:,.0f}")

    Path(args.macros).write_text("% Auto-generated by src/figures/plot_corridor.py -- do not edit.\n"
                                 + "\n".join(macros) + "\n")
    print(f"[figures] 8 figures -> {out}, {len(macros)} macros -> {args.macros}")


if __name__ == "__main__":
    main()
