"""Figures + macros for the topological-attention project.

Primary test (pre-registered, doc/05_decisions.md): a paired Welch/t-test
per layer on H0 total persistence (truthful - hallucinated, paired by
pair_idx), with Holm correction across layers. This supersedes the
pilot's unpaired Welch test, which is kept as a secondary/legacy check
for continuity with the original paper.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sst
from utils.plotting import col_fig, apply_style
from utils.stats import bootstrap_ci


DIGIT_WORDS = {"0":"Zero","1":"One","2":"Two","3":"Three","4":"Four","5":"Five",
               "6":"Six","7":"Seven","8":"Eight","9":"Nine","_":"","-":""}
def latex_safe(s):
    return "".join(DIGIT_WORDS.get(c, c) for c in s)


def holm_correct(pvals):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        raw = pvals[i] * (m - rank)
        running_max = max(running_max, raw)
        adj[i] = min(running_max, 1.0)
    return adj


def paired_series(rows, layer, field="total_persistence_H0"):
    """Return per-pair (truthful, hallucinated) arrays for a layer, aligned by
    pair_idx, restricted to pairs present in both kinds and with a value."""
    t = {r["pair_idx"]: r[field] for r in rows if r["layer"] == layer and r["kind"] == "truthful" and field in r}
    h = {r["pair_idx"]: r[field] for r in rows if r["layer"] == layer and r["kind"] == "hallucinated" and field in r}
    common = sorted(set(t) & set(h))
    return np.array([t[i] for i in common]), np.array([h[i] for i in common])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    ap.add_argument("--config", default=None)  # accepted for Makefile symmetry; unused
    args = ap.parse_args()
    apply_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []

    for path in sorted(Path(args.results).glob("tda_*.json")):
        short = path.stem.replace("tda_", "")
        short_tex = latex_safe(short)
        rows = json.loads(path.read_text())
        layers = sorted({r["layer"] for r in rows})
        has_tokens = all("n_tokens" in r for r in rows)
        has_nosink = any("total_persistence_H0_nosink" in r for r in rows)

        # ---- per-layer mean total persistence H0 / H1, by kind
        kinds = ["truthful", "hallucinated"]
        per_layer_h1 = {k: [] for k in kinds}
        per_layer_h0 = {k: [] for k in kinds}
        for l in layers:
            for k in kinds:
                vals_h1 = [r["total_persistence_H1"] for r in rows
                           if r["layer"] == l and r["kind"] == k]
                vals_h0 = [r["total_persistence_H0"] for r in rows
                           if r["layer"] == l and r["kind"] == k]
                per_layer_h1[k].append(np.mean(vals_h1) if vals_h1 else 0.0)
                per_layer_h0[k].append(np.mean(vals_h0) if vals_h0 else 0.0)

        fig, ax = col_fig(height_ratio=0.7)
        ax.plot(layers, per_layer_h1["truthful"], marker="o", label="truthful H₁")
        ax.plot(layers, per_layer_h1["hallucinated"], marker="s", label="hallucinated H₁")
        ax.set_xlabel("layer"); ax.set_ylabel("total persistence (H1)")
        ax.set_title(f"H1 total persistence per layer ({short})")
        ax.legend()
        fig.savefig(out / f"h1_per_layer_{short_tex}.pdf"); plt.close(fig)

        fig, ax = col_fig(height_ratio=0.7)
        ax.plot(layers, per_layer_h0["truthful"], marker="o", label="truthful H₀")
        ax.plot(layers, per_layer_h0["hallucinated"], marker="s", label="hallucinated H₀")
        ax.set_xlabel("layer"); ax.set_ylabel("total persistence (H0)")
        ax.set_title(f"H0 total persistence per layer ({short})")
        ax.legend()
        fig.savefig(out / f"h0_per_layer_{short_tex}.pdf"); plt.close(fig)

        # ---- legacy: unpaired Welch on H0 (kept for continuity with the pilot)
        layer_p, layer_t = [], []
        for l in layers:
            t1 = [r["total_persistence_H0"] for r in rows
                  if r["layer"] == l and r["kind"] == "truthful"]
            t2 = [r["total_persistence_H0"] for r in rows
                  if r["layer"] == l and r["kind"] == "hallucinated"]
            if len(t1) >= 3 and len(t2) >= 3:
                t, p = sst.ttest_ind(t1, t2, equal_var=False)
            else:
                t, p = 0.0, 1.0
            layer_t.append(t); layer_p.append(p)
        layer_p_holm = holm_correct(np.array(layer_p))

        fig, ax = col_fig(height_ratio=0.7)
        ax.bar(layers, [-np.log10(max(p, 1e-300)) for p in layer_p])
        ax.axhline(-np.log10(0.05), ls="--", color="red", lw=0.7, label="p=0.05")
        ax.set_xlabel("layer"); ax.set_ylabel("-log10 p (Welch t-test on H0)")
        ax.set_title(f"Per-layer significance ({short})")
        ax.legend(fontsize=6)
        fig.savefig(out / f"pvals_{short_tex}.pdf"); plt.close(fig)

        # ---- mean Betti curve at the most-discriminative layer (by legacy test)
        best_layer = layers[int(np.argmin(layer_p))]
        thr = rows[0]["thresholds"]
        def avg_betti(k, key):
            curves = np.array([r[key] for r in rows if r["layer"] == best_layer and r["kind"] == k])
            return curves.mean(axis=0) if curves.size else np.zeros_like(thr)
        fig, ax = col_fig(height_ratio=0.7)
        ax.plot(thr, avg_betti("truthful", "betti_0"), label="truthful B₀")
        ax.plot(thr, avg_betti("hallucinated", "betti_0"), label="hallucinated B₀")
        if "betti_1" in rows[0]:
            ax.plot(thr, avg_betti("truthful", "betti_1"), label="truthful B₁", ls="--")
            ax.plot(thr, avg_betti("hallucinated", "betti_1"), label="hallucinated B₁", ls="--")
        ax.set_xlabel("filtration distance")
        ax.set_ylabel("Betti number")
        ax.set_title(f"Betti curves at layer {best_layer} ({short})")
        ax.legend(fontsize=6)
        fig.savefig(out / f"betti_curves_{short_tex}.pdf"); plt.close(fig)

        min_p = float(min(layer_p))
        n_sig = sum(1 for p in layer_p if p < 0.05)
        n_sig_holm = int(np.sum(layer_p_holm < 0.05))
        row_macros = [
            (f"{short_tex}MinPval", f"{min_p:.3g}"),
            (f"{short_tex}MinPvalLayer", f"{int(best_layer)}"),
            (f"{short_tex}NumSigLayers", f"{n_sig}"),
            (f"{short_tex}NumSigLayersHolm", f"{n_sig_holm}"),
            (f"{short_tex}TruthHZeroMean", f"{np.mean(per_layer_h0['truthful']):.3f}"),
            (f"{short_tex}HallHZeroMean", f"{np.mean(per_layer_h0['hallucinated']):.3f}"),
            (f"{short_tex}TruthHOneMean", f"{np.mean(per_layer_h1['truthful']):.3f}"),
            (f"{short_tex}HallHOneMean", f"{np.mean(per_layer_h1['hallucinated']):.3f}"),
            (f"{short_tex}NumLayers", f"{len(layers)}"),
            (f"{short_tex}NumPairs", f"{len({r['pair_idx'] for r in rows})}"),
        ]

        if "total_persistence_H2" in rows[0]:
            h2_t = [r["total_persistence_H2"] for r in rows if r["kind"] == "truthful"]
            h2_h = [r["total_persistence_H2"] for r in rows if r["kind"] == "hallucinated"]
            row_macros += [
                (f"{short_tex}TruthHTwoMean", f"{np.mean(h2_t):.3f}"),
                (f"{short_tex}HallHTwoMean", f"{np.mean(h2_h):.3f}"),
            ]

        # ---- primary test: paired, per layer, Holm across layers
        paired_p, paired_dz, wilcoxon_p, lennorm_diffs = [], [], [], []
        for l in layers:
            t, h = paired_series(rows, l)
            if len(t) >= 3:
                diff = t - h
                sd = diff.std(ddof=1)
                dz = float(diff.mean() / sd) if sd > 0 else 0.0
                tt, pp = sst.ttest_rel(t, h)
                try:
                    _, wp = sst.wilcoxon(diff)
                except ValueError:
                    wp = 1.0
            else:
                dz, pp, wp = 0.0, 1.0, 1.0
            paired_p.append(pp); paired_dz.append(dz); wilcoxon_p.append(wp)

            if has_tokens:
                tn, hn = paired_series(rows, l, "n_tokens")
                t2, h2 = paired_series(rows, l)
                if len(t2) >= 3:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ln_diff = t2 / np.maximum(tn, 1) - h2 / np.maximum(hn, 1)
                    lennorm_diffs.append((l, ln_diff))

        paired_p_holm = holm_correct(np.array(paired_p))
        min_paired_p = float(min(paired_p)) if paired_p else 1.0
        min_paired_layer = layers[int(np.argmin(paired_p))] if paired_p else -1
        n_sig_paired_holm = int(np.sum(paired_p_holm < 0.05))
        min_wilcoxon_p = float(min(wilcoxon_p)) if wilcoxon_p else 1.0

        row_macros += [
            (f"{short_tex}PairedMinPval", f"{min_paired_p:.3g}"),
            (f"{short_tex}PairedMinPvalLayer", f"{int(min_paired_layer)}"),
            (f"{short_tex}PairedNumSigHolm", f"{n_sig_paired_holm}"),
            (f"{short_tex}WilcoxonMinPval", f"{min_wilcoxon_p:.3g}"),
            (f"{short_tex}MeanDz", f"{np.mean(paired_dz):.3f}"),
            (f"{short_tex}MaxDz", f"{np.max(np.abs(paired_dz)):.3f}"),
        ]

        if lennorm_diffs:
            lennorm_p = []
            for l, d in lennorm_diffs:
                if len(d) >= 3 and d.std(ddof=1) > 0:
                    _, p = sst.ttest_1samp(d, 0.0)
                else:
                    p = 1.0
                lennorm_p.append(p)
            row_macros.append((f"{short_tex}LenNormMinPval", f"{min(lennorm_p):.3g}"))

        if has_nosink:
            nosink_p = []
            for l in layers:
                t, h = paired_series(rows, l, "total_persistence_H0_nosink")
                if len(t) >= 3:
                    diff = t - h
                    if diff.std(ddof=1) > 0:
                        _, p = sst.ttest_1samp(diff, 0.0)
                    else:
                        p = 1.0
                else:
                    p = 1.0
                nosink_p.append(p)
            row_macros.append((f"{short_tex}NoSinkPairedMinPval", f"{min(nosink_p):.3g}"))

        # ---- dz-per-layer figure with bootstrap CI
        dz_lo, dz_hi = [], []
        for l in layers:
            t, h = paired_series(rows, l)
            if len(t) >= 3:
                diff = t - h
                boots = []
                rng = np.random.default_rng(0)
                for _ in range(1000):
                    s = rng.choice(diff, size=len(diff), replace=True)
                    sd = s.std(ddof=1)
                    boots.append(s.mean() / sd if sd > 0 else 0.0)
                dz_lo.append(np.quantile(boots, 0.025))
                dz_hi.append(np.quantile(boots, 0.975))
            else:
                dz_lo.append(0.0); dz_hi.append(0.0)

        fig, ax = col_fig(height_ratio=0.7)
        ax.errorbar(layers, paired_dz,
                     yerr=[np.array(paired_dz) - np.array(dz_lo), np.array(dz_hi) - np.array(paired_dz)],
                     fmt="o-", capsize=2)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("layer"); ax.set_ylabel("paired effect size (dz)")
        ax.set_title(f"Paired effect size per layer, 95% CI ({short})")
        fig.savefig(out / f"paired_dz_{short_tex}.pdf"); plt.close(fig)

        macros += row_macros

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for n, v in macros:
            f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    print(f"[plot_tda] wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
