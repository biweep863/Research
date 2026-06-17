"""Figures + macros for the topological-attention project."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sst
from utils.plotting import col_fig, apply_style


DIGIT_WORDS = {"0":"Zero","1":"One","2":"Two","3":"Three","4":"Four","5":"Five",
               "6":"Six","7":"Seven","8":"Eight","9":"Nine","_":"","-":""}
def latex_safe(s):
    return "".join(DIGIT_WORDS.get(c, c) for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []

    for path in sorted(Path(args.results).glob("tda_*.json")):
        short = path.stem.replace("tda_", "")
        short_tex = latex_safe(short)
        rows = json.loads(path.read_text())
        layers = sorted({r["layer"] for r in rows})

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

        # ---- significance test on H0 (H1 trivial at these sizes)
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

        fig, ax = col_fig(height_ratio=0.7)
        ax.bar(layers, [-np.log10(max(p, 1e-300)) for p in layer_p])
        ax.axhline(-np.log10(0.05), ls="--", color="red", lw=0.7, label="p=0.05")
        ax.set_xlabel("layer"); ax.set_ylabel("-log10 p (Welch t-test on H0)")
        ax.set_title(f"Per-layer significance ({short})")
        ax.legend(fontsize=6)
        fig.savefig(out / f"pvals_{short_tex}.pdf"); plt.close(fig)

        # ---- mean Betti curve at the most-discriminative layer
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

        # ---- macros (H0 is the primary signal at these sizes)
        min_p = float(min(layer_p))
        n_sig = sum(1 for p in layer_p if p < 0.05)
        macros += [
            (f"{short_tex}MinPval", f"{min_p:.3g}"),
            (f"{short_tex}MinPvalLayer", f"{int(best_layer)}"),
            (f"{short_tex}NumSigLayers", f"{n_sig}"),
            (f"{short_tex}TruthHZeroMean", f"{np.mean(per_layer_h0['truthful']):.3f}"),
            (f"{short_tex}HallHZeroMean", f"{np.mean(per_layer_h0['hallucinated']):.3f}"),
            (f"{short_tex}TruthHOneMean", f"{np.mean(per_layer_h1['truthful']):.3f}"),
            (f"{short_tex}HallHOneMean", f"{np.mean(per_layer_h1['hallucinated']):.3f}"),
            (f"{short_tex}NumLayers", f"{len(layers)}"),
            (f"{short_tex}NumPairs", f"{len({r['pair_idx'] for r in rows})}"),
        ]

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for n, v in macros:
            f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    print(f"[plot_tda] wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
