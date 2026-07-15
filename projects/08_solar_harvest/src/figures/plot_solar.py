"""Figures + LaTeX macros for the solar-harvest paper.

Reads results/{resource,sbsp,optimize}.json; writes PDFs to paper/figures and
every quoted number to paper/macros.tex, so recomputing the figures rewrites
the prose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import col_fig, text_fig, apply_style


def _grp(x: float) -> str:
    """Group thousands with LaTeX-safe separators (works in text and math)."""
    return f"{x:,.0f}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()

    res = Path(args.results)
    rsrc = json.loads((res / "resource.json").read_text())
    sbsp = json.loads((res / "sbsp.json").read_text())
    opt = json.loads((res / "optimize.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macros: list[str] = []

    # ------------------------------------------------------------------ #
    # Fig 1: loss cascade waterfall + band surface GHI vs latitude
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.6))
    apply_style()
    labels = [lv["label"] for lv in rsrc["cascade"]]
    vals = [lv["irradiance"] for lv in rsrc["cascade"]]
    short = ["S$_0$", "TOA avg", "clouds", "ozone", "surface"]
    ax1.bar(range(len(vals)), vals, color=["#f4a300", "#f4c430", "#7fb3d5",
                                           "#5499c7", "#2e86c1"])
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels(short, rotation=0)
    ax1.set_ylabel("irradiance (W/m$^2$)")
    ax1.set_title("From solar constant to surface")
    for i, v in enumerate(vals):
        ax1.text(i, v + 25, f"{v:.0f}", ha="center", fontsize=6)

    lats = [b["lat"] for b in rsrc["band_ghi"]]
    ghis = [b["ghi"] for b in rsrc["band_ghi"]]
    ax2.plot(lats, ghis, "o-", color="#2e86c1")
    ax2.set_xlabel("|latitude| (deg)")
    ax2.set_ylabel("annual-mean surface GHI (W/m$^2$)")
    ax2.set_title("Best resource: the desert belt")
    ax2.axvspan(15, 35, color="#f4c430", alpha=0.18, label="subtropical desert")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "cascade_bands.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: deployable land energy vs demand
    # ------------------------------------------------------------------ #
    land = rsrc["land"]
    fig, ax = col_fig(height_ratio=0.85)
    bars = {
        "world\nprimary": land["world_primary_W"],
        "world\nelectricity": land["world_electricity_W"],
        "deployable\nsolar": land["deployable_power_W"],
    }
    xs = range(len(bars))
    ax.bar(xs, [v / 1e12 for v in bars.values()],
           color=["#909497", "#5d6d7e", "#28b463"])
    ax.set_yscale("log")
    ax.set_xticks(list(xs)); ax.set_xticklabels(list(bars.keys()))
    ax.set_ylabel("average power (TW)")
    ax.set_title("Potential dwarfs demand")
    for i, v in enumerate(bars.values()):
        ax.text(i, v / 1e12 * 1.15, f"{v/1e12:.0f}", ha="center", fontsize=7)
    fig.savefig(out / "land_vs_demand.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: SBSP chain + ground vs space per m^2
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5))
    apply_style()
    ch = sbsp["chain"]
    ax1.bar(range(len(ch)), [c["power"] for c in ch], color="#8e44ad")
    ax1.set_xticks(range(len(ch)))
    ax1.set_xticklabels(["collect", "PV", "DC-RF", "atm", "grid"], rotation=0)
    ax1.set_ylabel("W per m$^2$ collector")
    ax1.set_title("Orbital collection chain")
    for i, c in enumerate(ch):
        ax1.text(i, c["power"] + 25, f"{c['power']:.0f}", ha="center", fontsize=6)

    ax2.bar([0, 1], [sbsp["ground_per_m2_W"], sbsp["space_per_m2_W"]],
            color=["#28b463", "#8e44ad"])
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["ground\ndesert", "orbit\n(SBSP)"])
    ax2.set_ylabel("delivered W per m$^2$")
    ax2.set_title(f"Per-m$^2$: {sbsp['ratio']:.1f}x")
    for i, v in enumerate([sbsp["ground_per_m2_W"], sbsp["space_per_m2_W"]]):
        ax2.text(i, v + 6, f"{v:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sbsp.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: ecosystem optimisation -- retained value vs f_min
    # ------------------------------------------------------------------ #
    sw = opt["sweep"]
    fm = [s["f_min"] for s in sw]
    fig, ax = col_fig(height_ratio=0.8)
    ax.plot(fm, [100 * s["optimal_retained"] for s in sw], "o-",
            color="#28b463", label="optimal (temporal)")
    ax.plot(fm, [100 * s["flat_retained"] for s in sw], "s--",
            color="#e67e22", label="flat reservation")
    ax.set_xlabel("ecosystem reserve fraction $f_{\\min}$")
    ax.set_ylabel("harvest value retained (%)")
    ax.set_title("Scheduling around the ecosystem")
    ax2 = ax.twinx()
    ax2.plot(fm, [100 * s["temporal_gain"] for s in sw], "^:",
             color="#7d3c98", label="temporal gain")
    ax2.set_ylabel("temporal gain over flat (%)", color="#7d3c98")
    ax2.tick_params(axis="y", labelcolor="#7d3c98")
    ax2.grid(False)
    ax.legend(loc="lower left")
    fig.savefig(out / "optimize_sweep.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: headline accept-fraction heatmap (cells x hours) + demand
    # ------------------------------------------------------------------ #
    accept = np.array(opt["headline_accept"])
    dem = np.array(opt["demand_weights"])
    n_hours = accept.shape[1]
    hours = np.linspace(0, 24, n_hours, endpoint=False)
    fig, ax = col_fig(height_ratio=0.8)
    im = ax.imshow(accept, aspect="auto", cmap="YlGn", vmin=0, vmax=1,
                   extent=[0, 24, len(accept) - 0.5, -0.5])
    ax.set_yticks(range(len(opt["cells_lat"])))
    ax.set_yticklabels([f"{l}$^\\circ$" for l in opt["cells_lat"]])
    ax.set_xlabel("hour of day")
    ax.set_ylabel("band |lat|")
    ax.set_title(f"Harvest fraction (reserve $f_{{\\min}}={opt['f_min_headline']}$)")
    ax.grid(False)
    axd = ax.twinx()
    axd.plot(hours, dem, color="#c0392b", lw=1.0, label="demand")
    axd.set_ylabel("demand weight", color="#c0392b")
    axd.tick_params(axis="y", labelcolor="#c0392b")
    axd.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cb.set_label("harvest fraction $a$", fontsize=6)
    fig.savefig(out / "harvest_heatmap.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros: every number quoted in the paper
    # ------------------------------------------------------------------ #
    head = next(s for s in sw if abs(s["f_min"] - opt["f_min_headline"]) < 1e-9)
    tmax = max(sw, key=lambda s: s["temporal_gain"])
    best_band = max(rsrc["band_ghi"], key=lambda b: b["ghi"])
    m = macros.append
    m(("InterceptedTW", _grp(rsrc["intercepted_power_W"] / 1e12)))
    m(("ToaMean", f"{rsrc['toa_horizontal_mean_W_m2']:.0f}"))
    m(("SurfaceGHI", f"{rsrc['surface_ghi_mean_W_m2']:.0f}"))
    m(("OzoneFrac", "2"))
    m(("BestBandGHI", f"{best_band['ghi']:.0f}"))
    m(("BestBandLat", f"{best_band['lat']:.0f}"))
    m(("DeployableMkm", f"{rsrc['land']['deployable_area_m2']/1e12:.1f}"))
    m(("DeployableTW", f"{rsrc['land']['deployable_power_W']/1e12:.0f}"))
    m(("TimesElec", f"{rsrc['land']['frac_of_electricity']:.0f}"))
    m(("TimesPrimary", f"{rsrc['land']['frac_of_primary']:.1f}"))
    m(("AreaWorldKm", _grp(rsrc['land']['area_for_world_electricity_m2'] / 1e6)))
    m(("AreaWorldSide", f"{(rsrc['land']['area_for_world_electricity_m2']/1e6)**0.5:.0f}"))
    m(("MillionCities", _grp(rsrc['land']['n_million_cities'])))
    m(("GroundWm", f"{sbsp['ground_per_m2_W']:.0f}"))
    m(("SpaceWm", f"{sbsp['space_per_m2_W']:.0f}"))
    m(("SbspRatio", f"{sbsp['ratio']:.1f}"))
    m(("SpaceAreaKm", _grp(sbsp['space_area_for_world_m2'] / 1e6)))
    m(("SpaceCF", f"{100*sbsp['space_capacity_factor']:.0f}"))
    m(("GroundCF", f"{100*sbsp['ground_capacity_factor']:.0f}"))
    m(("FminHead", f"{opt['f_min_headline']}"))
    m(("TemporalGainHead", f"{100*head['temporal_gain']:.1f}"))
    m(("OptRetHead", f"{100*head['optimal_retained']:.1f}"))
    m(("FlatRetHead", f"{100*head['flat_retained']:.0f}"))
    m(("TemporalGainMax", f"{100*tmax['temporal_gain']:.1f}"))
    m(("TemporalGainMaxF", f"{tmax['f_min']}"))

    macro_lines = [f"\\newcommand{{\\{name}}}{{{val}}}" for name, val in macros]
    Path(args.macros).write_text("\n".join(macro_lines) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 5 figures")


if __name__ == "__main__":
    main()
