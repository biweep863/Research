"""Figures + macros for Kolmogorov-attention project."""
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

    for path in sorted(Path(args.results).glob("complexity_*.json")):
        short = path.stem.replace("complexity_", "")
        short_tex = latex_safe(short)
        rows = json.loads(path.read_text())
        layers = sorted({r["layer"] for r in rows})

        # ratio used as Kolmogorov-density proxy
        r_per_layer = {}
        slopes, intercepts, r2s = [], [], []
        for l in layers:
            r_l = [r for r in rows if r["layer"] == l]
            x = np.array([r["compression_ratio"] for r in r_l])
            y_ent = np.array([r["mean_entropy"] for r in r_l])
            res = sst.linregress(x, y_ent)
            slopes.append(res.slope)
            intercepts.append(res.intercept)
            r2s.append(res.rvalue ** 2)
            r_per_layer[l] = (x, y_ent)

        # Figure: per-layer R² of entropy ~ compression_ratio
        fig, ax = col_fig(height_ratio=0.7)
        ax.plot(layers, r2s, marker="o")
        ax.set_xlabel("layer"); ax.set_ylabel("$R^2$")
        ax.set_title(f"Predictive R² per layer ({short})")
        fig.savefig(out / f"r2_per_layer_{short_tex}.pdf"); plt.close(fig)

        # Figure: scatter at deepest layer
        x, y = r_per_layer[layers[-1]]
        fig, ax = col_fig(height_ratio=0.8)
        families = sorted({r["family"] for r in rows})
        cmap = plt.get_cmap("tab10")
        for i, fam in enumerate(families):
            idx = [k for k, r in enumerate(rows)
                   if r["layer"] == layers[-1] and r["family"] == fam]
            xs = np.array([rows[k]["compression_ratio"] for k in idx])
            ys = np.array([rows[k]["mean_entropy"] for k in idx])
            ax.scatter(xs, ys, label=fam, c=[cmap(i)], s=10, alpha=0.7)
        ax.set_xlabel("compression ratio (Kolmogorov proxy)")
        ax.set_ylabel("mean attention entropy (deepest layer)")
        ax.set_title(f"Density vs complexity ({short}, layer {layers[-1]})")
        ax.legend(fontsize=5, ncol=2)
        fig.savefig(out / f"scatter_deep_{short_tex}.pdf"); plt.close(fig)

        # Figure: per-family density curves
        fig, ax = col_fig(height_ratio=0.8)
        for i, fam in enumerate(families):
            avg = []
            for l in layers:
                vals = [r["mean_entropy"] for r in rows if r["layer"] == l and r["family"] == fam]
                avg.append(np.mean(vals) if vals else np.nan)
            ax.plot(layers, avg, label=fam, color=cmap(i))
        ax.set_xlabel("layer"); ax.set_ylabel("mean attention entropy")
        ax.set_title(f"Attention entropy per family ({short})")
        ax.legend(fontsize=5, ncol=2)
        fig.savefig(out / f"entropy_per_family_{short_tex}.pdf"); plt.close(fig)

        macros += [
            (f"{short_tex}MaxRsq", f"{max(r2s):.3f}"),
            (f"{short_tex}MaxRsqLayer", f"{int(layers[int(np.argmax(r2s))])}"),
            (f"{short_tex}DeepRsq", f"{r2s[-1]:.3f}"),
            (f"{short_tex}DeepSlope", f"{slopes[-1]:.3f}"),
        ]

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for n, v in macros:
            f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    print(f"[plot_complexity] wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
