"""Figures + LaTeX macros for the Hodge-preferences paper.

Reads results/{spectrum,dynamics,noise,width,arena}.json, writes six PDFs to
paper/figures and every quoted number to paper/macros.tex.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from utils.plotting import apply_style

COL = {"grad": "#1565c0", "curl": "#c62828", "harm": "#ef6c00"}
MCOL = {"scalar": "#616161", "antisym": "#2e7d32", "gpm": "#6a1b9a"}
NAME = {"grad": "gradient (transitive)", "curl": "curl (local cycles)", "harm": "harmonic (global cycles)"}


def _arrow(ax, p, q, col, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=col, lw=lw,
                                 shrinkA=9, shrinkB=9))


def schematic(out):
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.9))
    pos = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1), 4: (0.5, 1.7)}
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (2, 4), (3, 4)]
    # Exact components: project a mixed flow with the real Hodge projectors.
    from src.hodge.graph import from_edges
    from src.hodge.decompose import projectors
    g = from_edges(5, edges)
    order = [tuple(e) for e in g.edges.tolist()]
    raw = {(0, 1): 1.0, (1, 2): 1.2, (2, 3): 0.3, (0, 3): -0.9, (2, 4): 1.4, (3, 4): -0.2}
    y = np.array([raw[e] for e in order]); P = projectors(g)
    flows = {c: {e: float(v) for e, v in zip(order, P[c] @ y)} for c in ("grad", "curl", "harm")}
    titles = {"grad": "gradient: a ranking", "curl": "curl: rock-paper-scissors", "harm": "harmonic: a cycle round a hole"}
    for ax, c in zip(axes, ("grad", "curl", "harm")):
        for (i, j) in edges:
            ax.plot(*zip(pos[i], pos[j]), color="#dddddd", lw=1, zorder=0)
        for (i, j), v in flows[c].items():
            if abs(v) < 0.05:
                continue
            a, b = (pos[i], pos[j]) if v > 0 else (pos[j], pos[i])
            _arrow(ax, a, b, COL[c], lw=0.6 + 2.0 * min(abs(v), 1.5))
        for k, p in pos.items():
            ax.scatter(*p, s=110, color="white", edgecolors="black", zorder=3)
            ax.text(*p, str(k), ha="center", va="center", fontsize=7, zorder=4)
        ax.set_title(titles[c], fontsize=8); ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.35, 2.0); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "schematic.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args(); apply_style()
    res = Path(args.results)
    spe = json.loads((res / "spectrum.json").read_text()); dyn = json.loads((res / "dynamics.json").read_text())
    noi = json.loads((res / "noise.json").read_text()); wid = json.loads((res / "width.json").read_text())
    are = json.loads((res / "arena.json").read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []; m = macros.append
    schematic(out)

    # ------------------------------------------------------------------ #
    # Fig 2: spectrum
    # ------------------------------------------------------------------ #
    rows = [r for r in spe["rows"] if r["depth"] == 1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.75, 2.3))
    ds = sorted({r["d"] for r in rows})
    for d, col in zip(ds, plt.cm.viridis(np.linspace(0.1, 0.85, len(ds)))):
        rr = [r for r in rows if r["d"] == d]
        a1.scatter([r["mean_degree"] for r in rr], [r["ratio_antisym"] for r in rr], color=col, s=14, label=f"d = {d}")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlabel("mean degree of the comparison graph")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for ax_ in (a1,):
        ax_.xaxis.set_minor_formatter(NullFormatter()); ax_.xaxis.set_major_formatter(ScalarFormatter())
    a1.set_ylabel("κ_grad / κ_cyc"); a1.set_title("Antisymmetric net: rate ratio"); a1.legend(fontsize=6)
    c = spe["canonical"]; k = min(80, len(c["eig"])); xx = np.arange(k)
    bottom = np.zeros(k)
    for comp in ("grad", "curl", "harm"):
        v = np.array(c[comp][:k]); a2.bar(xx, v, bottom=bottom, color=COL[comp], width=1.0, label=comp); bottom += v
    a2.set_xlabel("eigenvector rank (largest first)"); a2.set_ylabel("Hodge mass"); a2.set_title("Where the top modes live")
    a2.legend(fontsize=6, loc="center right"); a2.set_ylim(0, 1)
    emp = spe["empirical"]; ws = sorted({e["width"] for e in emp})
    for kind in ("antisym", "gpm"):
        mu = [np.mean([e["ratio"] for e in emp if e["model"] == kind and e["width"] == w]) for w in ws]
        sd = [np.std([e["ratio"] for e in emp if e["model"] == kind and e["width"] == w]) for w in ws]
        a3.errorbar(ws, mu, yerr=sd, marker="o", color=MCOL[kind], label=kind, capsize=2)
    ana = spe["analytic_canonical"]["antisym"]; dm = c["dims"]
    cyc = (ana["curl"] * dm["curl"] + (ana["harm"] * dm["harm"] if dm["harm"] else 0)) / (dm["curl"] + dm["harm"])
    a3.axhline(ana["grad"] / cyc, color=MCOL["antisym"], ls="--", lw=0.8, label="antisym, ∞ width")
    a3.set_xscale("log"); a3.set_xlabel("width"); a3.set_ylabel("κ_grad / κ_cyc"); a3.set_title("Finite width at init")
    a3.xaxis.set_minor_formatter(NullFormatter()); a3.set_xticks(ws); a3.set_xticklabels([str(w) for w in ws])
    a3.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "spectrum.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: dynamics
    # ------------------------------------------------------------------ #
    steps = np.array(dyn["steps"])
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.4))
    for ax, (kind, loss) in zip(axes[:2], (("antisym", "sq"), ("antisym", "bt"))):
        r = next(x for x in dyn["runs"] if x["model"] == kind and x["loss"] == loss and x["seed"] == 0)
        for comp in ("grad", "curl", "harm"):
            ax.semilogx(np.maximum(r["log"]["step"], 1), r["log"][f"fit_{comp}"], "-", color=COL[comp], label=comp)
            if r["linearised"] is not None:
                ax.semilogx(np.maximum(steps, 1), r["linearised"][comp], "--", color=COL[comp], lw=0.8)
        ax.set_ylim(-0.05, 1.05); ax.set_xlabel("gradient-descent step"); ax.set_ylabel("fraction of component fitted")
        ax.set_title({"sq": "Squared loss (dashed: NTK theory)", "bt": "Bradley-Terry loss on votes"}[loss], fontsize=8)
    axes[0].legend(fontsize=6)
    groups = [(k, l) for k in ("antisym", "gpm") for l in ("sq", "bt")]
    for gi, (kind, loss) in enumerate(groups):
        rr = [x for x in dyn["runs"] if x["model"] == kind and x["loss"] == loss]
        for ci, comp in enumerate(("curl", "harm")):
            vals = []
            for x in rr:
                tg = x["crossings"]["grad"]["0.5"]; tc = x["crossings"][comp]["0.5"]
                if tg and tc:
                    vals.append(tc / tg)
            if vals:
                axes[2].bar(gi + (ci - 0.5) * 0.38, np.mean(vals), 0.36, color=COL[comp], yerr=np.std(vals), capsize=2,
                            label=comp if gi == 0 else None)
    axes[2].set_yscale("log"); axes[2].set_xticks(range(len(groups)))
    axes[2].set_xticklabels([f"{k}\n{l}" for k, l in groups], fontsize=6.5)
    axes[2].axhline(1, color="grey", lw=0.6, ls=":"); axes[2].set_ylabel("t50(cycle) / t50(gradient)")
    axes[2].set_title("How much later cycles are fitted", fontsize=8); axes[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "dynamics.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: noise
    # ------------------------------------------------------------------ #
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.75, 2.4))
    fr = noi["fraction_rows"]
    a1.errorbar([r["theory"] for r in fr], [r["mean"] for r in fr], yerr=[r["std"] for r in fr], fmt="o", color=COL["grad"], ms=3, capsize=2)
    a1.plot([0, 0.7], [0, 0.7], "--", color="grey", lw=0.8)
    a1.set_xlabel("theory (|V|−1)/|E|"); a1.set_ylabel("measured fraction in gradient space")
    a1.set_title("i.i.d. noise is mostly cyclic", fontsize=8)
    agg = defaultdict(list)
    for r in noi["runs"]:
        agg[(r["model"], r["mode"], r["level"])].append(r)
    for kind, ls in (("antisym", "-"), ("gpm", "--")):
        for lev, col in zip((0.0, 0.2, 0.4), ("#2e7d32", "#ef6c00", "#c62828")):
            rr = agg[(kind, "flip", lev)]
            curve = np.mean([x["log"]["ho_acc"] for x in rr], 0)
            a2.semilogx(np.maximum(rr[0]["log"]["step"], 1), curve, ls, color=col, label=f"{kind}, flip {lev:.1f}")
    a2.set_xlabel("gradient-descent step"); a2.set_ylabel("held-out accuracy (clean truth)"); a2.set_title("Early stopping under vote flips", fontsize=8)
    a2.legend(fontsize=5.5)
    for kind in ("antisym", "gpm", "scalar"):
        levs = sorted({k[2] for k in agg if k[0] == kind and k[1] == "flip"})
        gain = [100 * np.mean([x["acc_best"] - x["acc_final"] for x in agg[(kind, "flip", l)]]) for l in levs]
        a3.plot(levs, gain, "o-", color=MCOL[kind], label=kind)
    a3.set_xlabel("fraction of votes flipped"); a3.set_ylabel("accuracy gain of best stop (pts)"); a3.set_title("What early stopping buys", fontsize=8)
    a3.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "noise.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: width / lazy-rich
    # ------------------------------------------------------------------ #
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.4))
    alphas = sorted({r["alpha"] for r in wid["rows"]}); widths = sorted({r["width"] for r in wid["rows"]})
    for a, col in zip(alphas, plt.cm.plasma(np.linspace(0.1, 0.8, len(alphas)))):
        ratio, drift = [], []
        for w in widths:
            rr = [r for r in wid["rows"] if r["alpha"] == a and r["width"] == w]
            ratio.append(np.mean([r["crossings"]["curl"]["0.9"] / r["crossings"]["grad"]["0.9"] for r in rr
                                  if r["crossings"]["curl"]["0.9"] and r["crossings"]["grad"]["0.9"]]))
            drift.append(np.mean([r["kernel"][-1]["drift"] for r in rr]))
        a1.semilogx(widths, ratio, "o-", color=col, label=f"α = {a:g}")
        a2.loglog(widths, drift, "o-", color=col, label=f"α = {a:g}")
    a1.axhline(1, color="grey", ls=":", lw=0.6); a1.set_xlabel("width"); a1.set_ylabel("t90(curl) / t90(gradient)")
    a1.set_title("Order survives feature learning", fontsize=8); a1.legend(fontsize=6)
    a2.set_xlabel("width"); a2.set_ylabel("‖Θ_T − Θ_0‖ / ‖Θ_0‖"); a2.set_title("Kernel drift (small α = rich)", fontsize=8)
    a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "width.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 6: arena
    # ------------------------------------------------------------------ #
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(6.75, 2.4))
    nq = are["null"]["grad"]
    a1.bar([0], [1 - are["fractions"]["grad"]], color=COL["curl"], width=0.5, label="observed")
    a1.errorbar([1], [1 - nq[1]], yerr=[[nq[2] - nq[1]], [nq[1] - nq[0]]], fmt="s", color="grey", capsize=3, label="transitive null (95 %)")
    a1.set_xticks([0, 1]); a1.set_xticklabels(["Arena votes", "BT null"]); a1.set_ylabel("cyclic energy fraction")
    a1.set_title("Are the cycles real?", fontsize=8); a1.legend(fontsize=6)
    r0 = next(x for x in are["runs"] if x["model"] == "antisym" and x["flip"] == 0.0 and x["seed"] == 0)
    for comp in ("grad", "curl", "harm"):
        if are["dims"][comp]:
            a2.semilogx(np.maximum(r0["fit_log"]["step"], 1), r0["fit_log"][f"fit_{comp}"], color=COL[comp], label=comp)
    a2.set_ylim(-0.05, 1.05); a2.set_xlabel("step"); a2.set_ylabel("fraction fitted"); a2.set_title("Antisym on Arena votes", fontsize=8)
    a2.legend(fontsize=6)
    for kind in ("scalar", "antisym", "gpm"):
        for flip, ls in ((0.0, "-"), (0.4, "--")):
            rr = [x for x in are["runs"] if x["model"] == kind and x["flip"] == flip]
            a3.semilogx(np.maximum(rr[0]["fit_log"]["step"], 1), np.mean([x["acc_curve"] for x in rr], 0), ls,
                        color=MCOL[kind], label=f"{kind}, flip {flip:.1f}")
    a3.set_xlabel("step"); a3.set_ylabel("held-out vote accuracy"); a3.set_title("Held-out votes", fontsize=8)
    a3.legend(fontsize=5.5)
    fig.tight_layout(); fig.savefig(out / "arena.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    ratios = [r["ratio_antisym"] for r in spe["rows"]]
    m(("SpeNGraphs", str(len(spe["rows"])))); m(("SpeRatioMin", f"{min(ratios):.1f}")); m(("SpeRatioMax", f"{max(ratios):.0f}"))
    m(("SpeRatioAllAboveOne", "all" if min(ratios) > 1 else "not all"))
    ch = [r["ratio_curl_harm"] for r in spe["rows"] if r["ratio_curl_harm"]]
    m(("SpeCurlHarmMin", f"{min(ch):.2f}")); m(("SpeCurlHarmMax", f"{max(ch):.2f}")); m(("SpeCurlHarmMean", f"{np.mean(ch):.2f}"))
    sc = max(abs(r["scalar"]["curl"]) for r in spe["rows"])
    m(("SpeScalarCycMax", f"{sc:.0e}".replace("e-", "\\times10^{-") + "}"))
    dense = [r for r in spe["rows"] if r["density"] == 1.0 and r["depth"] == 1]; sparse = [r for r in spe["rows"] if r["density"] == 0.15 and r["depth"] == 1]
    m(("SpeRatioDense", f"{np.mean([r['ratio_antisym'] for r in dense]):.0f}")); m(("SpeRatioSparse", f"{np.mean([r['ratio_antisym'] for r in sparse]):.1f}"))
    d2 = [r for r in spe["rows"] if r["depth"] == 2]; d1 = [r for r in spe["rows"] if r["depth"] == 1]
    m(("SpeRatioDepthOne", f"{np.mean([r['ratio_antisym'] for r in d1]):.1f}")); m(("SpeRatioDepthTwo", f"{np.mean([r['ratio_antisym'] for r in d2]):.1f}"))
    lo_d = [r for r in d1 if r["d"] == min(ds)]; hi_d = [r for r in d1 if r["d"] == max(ds)]
    m(("SpeRatioLowD", f"{np.mean([r['ratio_antisym'] for r in lo_d]):.1f}")); m(("SpeRatioHighD", f"{np.mean([r['ratio_antisym'] for r in hi_d]):.1f}"))
    m(("SpeLowD", str(min(ds)))); m(("SpeHighD", str(max(ds))))
    top10 = np.mean(c["grad"][:10]); m(("SpeTopTenGradMass", f"{100 * top10:.0f}"))
    m(("SpeCanonN", str(spe["empirical"] and 30))); m(("SpeCanonE", str(c["E"])))
    m(("SpeCanonGradDim", str(c["dims"]["grad"]))); m(("SpeCanonCurlDim", str(c["dims"]["curl"]))); m(("SpeCanonHarmDim", str(c["dims"]["harm"])))
    m(("SpeAnaRatioCanon", f"{ana['grad'] / cyc:.1f}"))
    for kind in ("antisym", "gpm"):
        for w in (min(ws), max(ws)):
            vals = [e["ratio"] for e in emp if e["model"] == kind and e["width"] == w]
            m((f"SpeEmp{kind.capitalize()}W{'Lo' if w == min(ws) else 'Hi'}", f"{np.mean(vals):.1f}"))
    errs = [e["rel_err_vs_analytic"] for e in emp if e["model"] == "antisym" and e["width"] == max(ws)]
    m(("SpeEmpErrHi", f"{100 * np.mean(errs):.1f}")); m(("SpeEmpWHi", str(max(ws)))); m(("SpeEmpWLo", str(min(ws))))
    errs_lo = [e["rel_err_vs_analytic"] for e in emp if e["model"] == "antisym" and e["width"] == min(ws)]
    m(("SpeEmpErrLo", f"{100 * np.mean(errs_lo):.0f}"))

    t = dyn["task"]; m(("TaskN", str(t["n_items"]))); m(("TaskD", str(t["feat_dim"]))); m(("TaskDens", f"{t['density']}"))
    fr0 = [x["fractions"] for x in dyn["runs"] if x["model"] == "antisym" and x["loss"] == "sq"]
    for comp in ("grad", "curl", "harm"):
        m((f"TaskFrac{comp.capitalize()}", f"{100 * np.mean([f[comp] for f in fr0]):.0f}"))
    def ratios_for(kind, loss, comp, lev="0.5"):
        v = []
        for x in dyn["runs"]:
            if x["model"] == kind and x["loss"] == loss:
                tg, tc = x["crossings"]["grad"][lev], x["crossings"][comp][lev]
                if tg and tc:
                    v.append(tc / tg)
        return v
    for kind in ("antisym", "gpm"):
        for loss in ("sq", "bt"):
            for comp in ("curl", "harm"):
                v = ratios_for(kind, loss, comp)
                m((f"Dyn{kind.capitalize()}{loss.capitalize()}{comp.capitalize()}", f"{np.mean(v):.0f}" if v else "--"))
    order_ok = sum(1 for x in dyn["runs"] if x["model"] != "scalar" and x["crossings"]["grad"]["0.5"] is not None and
                   all((x["crossings"][c]["0.5"] is None) or (x["crossings"][c]["0.5"] >= x["crossings"]["grad"]["0.5"]) for c in ("curl", "harm")))
    m(("DynOrderOk", str(order_ok))); m(("DynOrderTotal", str(sum(1 for x in dyn["runs"] if x["model"] != "scalar"))))
    sq_runs = [x for x in dyn["runs"] if x["model"] == "antisym" and x["loss"] == "sq"]
    dev = [max(abs(a - b) for comp in ("grad", "curl", "harm") for a, b in zip(x["log"][f"fit_{comp}"], x["linearised"][comp])) for x in sq_runs]
    m(("DynLinDev", f"{np.mean(dev):.2f}"))
    ho = defaultdict(list)
    for x in dyn["runs"]:
        if x["loss"] == "sq":
            ho[x["model"]].append(x["log"]["ho_acc"][-1])
    for k2, v in ho.items():
        m((f"DynHo{k2.capitalize()}", f"{np.mean(v):.3f}"))
    scal_final = [x["log"]["fit_curl"][-1] for x in dyn["runs"] if x["model"] == "scalar"]
    m(("DynScalarCurlFinal", f"{max(abs(v) for v in scal_final):.0e}".replace("e-", "\\times10^{-") + "}" if max(abs(v) for v in scal_final) > 0 else "0"))

    frd = noi["fraction_rows"]
    m(("NoiFracMaxDev", f"{max(abs(r['mean'] - r['theory']) for r in frd):.4f}")); m(("NoiFracCases", str(len(frd))))
    dense_r = max(frd, key=lambda r: r["E"]); m(("NoiFracDense", f"{dense_r['mean']:.3f}")); m(("NoiFracDenseE", str(dense_r["E"]))); m(("NoiFracDenseN", str(dense_r["n"])))
    def nm(kind, mode, lev, key):
        return np.mean([x[key] for x in agg[(kind, mode, lev)]])
    m(("NoiTheoryTask", f"{(noi['n'] - 1) / noi['E']:.3f}"))
    gl = sorted({k[2] for k in agg if k[1] == "gauss" and k[2] > 0}); fl = sorted({k[2] for k in agg if k[1] == "flip" and k[2] > 0})
    m(("NoiGaussGradFrac", f"{np.mean([nm('antisym', 'gauss', l, 'noise_grad_fraction') for l in gl]):.2f}"))
    m(("NoiFlipGradFrac", f"{np.mean([nm('antisym', 'flip', l, 'noise_grad_fraction') for l in fl]):.2f}"))
    for kind in ("antisym", "gpm", "scalar"):
        K = kind.capitalize()
        m((f"Noi{K}FinalFlipFour", f"{nm(kind, 'flip', 0.4, 'acc_final'):.3f}")); m((f"Noi{K}BestFlipFour", f"{nm(kind, 'flip', 0.4, 'acc_best'):.3f}"))
        m((f"Noi{K}FinalClean", f"{nm(kind, 'flip', 0.0, 'acc_final'):.3f}")); m((f"Noi{K}BestClean", f"{nm(kind, 'flip', 0.0, 'acc_best'):.3f}"))
        m((f"Noi{K}GainFlipFour", f"{100 * (nm(kind, 'flip', 0.4, 'acc_best') - nm(kind, 'flip', 0.4, 'acc_final')):.1f}"))
        m((f"Noi{K}GainGaussTwo", f"{100 * (nm(kind, 'gauss', 2.0, 'acc_best') - nm(kind, 'gauss', 2.0, 'acc_final')):.1f}"))
        m((f"Noi{K}AtTstarClean", f"{nm(kind, 'flip', 0.0, 'acc_at_t_star'):.3f}"))
        m((f"Noi{K}StepBestClean", f"{nm(kind, 'flip', 0.0, 'step_best'):.0f}")); m((f"Noi{K}StepBestFlipFour", f"{nm(kind, 'flip', 0.4, 'step_best'):.0f}"))

    wr = wid["rows"]
    m(("WidOrderOk", str(sum(r["order_holds"] for r in wr)))); m(("WidRuns", str(len(wr))))
    def wmean(a, key):
        rr = [r for r in wr if r["alpha"] == a]
        if key == "ratio":
            return np.mean([r["crossings"]["curl"]["0.9"] / r["crossings"]["grad"]["0.9"] for r in rr if r["crossings"]["curl"]["0.9"] and r["crossings"]["grad"]["0.9"]])
        if key == "drift":
            return np.mean([r["kernel"][-1]["drift"] for r in rr])
        if key == "kratio":
            return np.mean([r["kernel"][-1]["rates"]["grad"] / r["kernel"][-1]["rates"]["curl"] for r in rr])
        return np.mean([r["ho_acc_final"] for r in rr])
    amin, amax = min(alphas), max(alphas)
    m(("WidAlphaRich", f"{amin:g}")); m(("WidAlphaLazy", f"{amax:g}"))
    m(("WidRatioRich", f"{wmean(amin, 'ratio'):.0f}")); m(("WidRatioLazy", f"{wmean(amax, 'ratio'):.0f}")); m(("WidRatioOne", f"{wmean(1.0, 'ratio'):.0f}"))
    m(("WidDriftRich", f"{wmean(amin, 'drift'):.1f}")); m(("WidDriftLazy", f"{wmean(amax, 'drift'):.3f}"))
    m(("WidKRatioRich", f"{wmean(amin, 'kratio'):.0f}")); m(("WidKRatioLazy", f"{wmean(amax, 'kratio'):.0f}"))
    m(("WidAccRich", f"{wmean(amin, 'acc'):.3f}")); m(("WidAccLazy", f"{wmean(amax, 'acc'):.3f}"))
    m(("WidMin", str(min(widths)))); m(("WidMax", str(max(widths))))

    m(("AreItems", str(are["items"]))); m(("AreEdges", str(are["edges"]))); m(("AreVotes", f"{are['votes']:,}".replace(",", "{,}")))
    m(("AreMinVotes", str(are["min_votes"])))
    for comp in ("grad", "curl", "harm"):
        m((f"AreDim{comp.capitalize()}", str(are["dims"][comp]))); m((f"AreFrac{comp.capitalize()}", f"{100 * are['fractions'][comp]:.1f}"))
    m(("AreCyc", f"{100 * (1 - are['fractions']['grad']):.1f}"))
    m(("AreNullCyc", f"{100 * (1 - nq[1]):.1f}")); m(("AreNullCycLo", f"{100 * (1 - nq[2]):.1f}")); m(("AreNullCycHi", f"{100 * (1 - nq[0]):.1f}"))
    m(("AreP", f"{are['p_null_cyclic_ge_observed']:.2f}"))
    m(("AreHoN", f"{are['n_heldout']:,}".replace(",", "{,}")))
    m(("AreHodgeLL", f"{are['hodgerank_heldout']['logloss']:.4f}")); m(("AreHodgeAcc", f"{are['hodgerank_heldout']['acc']:.3f}"))
    ra = defaultdict(list)
    for x in are["runs"]:
        ra[(x["model"], x["flip"])].append(x)
    for kind in ("scalar", "antisym", "gpm"):
        K = kind.capitalize()
        for flip, tag in ((0.0, "Clean"), (0.4, "FlipFour")):
            rr = ra[(kind, flip)]
            m((f"Are{K}LL{tag}", f"{np.mean([x['heldout_logloss'] for x in rr]):.4f}"))
            m((f"Are{K}Acc{tag}", f"{np.mean([x['heldout_acc'] for x in rr]):.3f}"))
            m((f"Are{K}Best{tag}", f"{np.mean([x['acc_best'] for x in rr]):.3f}"))
        m((f"Are{K}Ratio", f"{np.mean([x['ratio'] for x in ra[(kind, 0.0)]]):.0f}"))
    ta = [x["crossings"] for x in ra[("antisym", 0.0)]]
    m(("AreAntisymTGrad", f"{np.mean([c_['grad']['0.9'] for c_ in ta]):.0f}"))
    m(("AreAntisymTCurl", f"{np.mean([c_['curl']['0.9'] for c_ in ta if c_['curl']['0.9']]):.0f}"))
    m(("AreTopModel", are["top5"][0].replace("_", "\\_")))

    Path(args.macros).write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 6 figures")


if __name__ == "__main__":
    main()
