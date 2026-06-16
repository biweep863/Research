"""Figures + macros for quantization project."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt
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

    for path in sorted(Path(args.results).glob("math_eval_*.json")):
        short = path.stem.replace("math_eval_", "")
        short_tex = latex_safe(short)
        data = json.loads(path.read_text())
        rows = data["rows"]
        bits = [r["bits"] for r in rows]
        accs = [r["acc"] for r in rows]
        mses = [r["logit_mse"] for r in rows]
        kls  = [r["logit_kl"] for r in rows]

        fig, ax = col_fig(height_ratio=0.7)
        ax.plot(bits, accs, marker="o")
        ax.axhline(data["baseline_acc"], ls="--", color="gray", lw=0.8,
                   label=f"FP32 baseline {data['baseline_acc']:.2f}")
        ax.set_xlabel("bit-width")
        ax.set_ylabel("accuracy")
        ax.set_title(f"Math accuracy vs bit-width ({short})")
        ax.invert_xaxis()
        ax.legend()
        fig.savefig(out / f"acc_vs_bits_{short_tex}.pdf"); plt.close(fig)

        fig, ax = col_fig(height_ratio=0.7)
        ax.plot(bits, mses, marker="o", label="logit MSE")
        ax.plot(bits, kls, marker="s", label="logit KL")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("bit-width")
        ax.set_ylabel("distance to FP32")
        ax.set_title(f"Logit divergence vs bit-width ({short})")
        ax.legend()
        fig.savefig(out / f"mse_vs_bits_{short_tex}.pdf"); plt.close(fig)

        macros += [
            (f"{short_tex}BaselineAcc", f"{data['baseline_acc']:.3f}"),
            (f"{short_tex}AccBitsFour", f"{[r['acc'] for r in rows if r['bits']==4][0]:.3f}"),
            (f"{short_tex}AccBitsTwo",  f"{[r['acc'] for r in rows if r['bits']==2][0]:.3f}"),
            (f"{short_tex}MSEBitsFour", f"{[r['logit_mse'] for r in rows if r['bits']==4][0]:.4f}"),
            (f"{short_tex}MSEBitsTwo",  f"{[r['logit_mse'] for r in rows if r['bits']==2][0]:.4f}"),
        ]

    for path in sorted(Path(args.results).glob("sensitivity_*.json")):
        short = path.stem.replace("sensitivity_", "")
        short_tex = latex_safe(short)
        data = json.loads(path.read_text())
        rows = data["rows"]
        labels = [r["module"] for r in rows]
        dacc = [r["delta_acc"] for r in rows]
        mses = [r["logit_mse"] for r in rows]

        fig, ax = col_fig(height_ratio=0.7)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, dacc, width=0.4, label="ΔAcc")
        ax.bar(x + 0.2, [m / max(mses) for m in mses], width=0.4, label="normalised MSE")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(f"Submodule sensitivity @ {data['bits']}-bit ({short})")
        ax.legend()
        fig.savefig(out / f"sensitivity_{short_tex}.pdf"); plt.close(fig)

        for r in rows:
            macros.append((f"{short_tex}Sens{r['module'].title()}DeltaAcc", f"{r['delta_acc']:.3f}"))
            macros.append((f"{short_tex}Sens{r['module'].title()}MSE", f"{r['logit_mse']:.4f}"))

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for name, val in macros:
            f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    print(f"[plot_quantization] wrote {len(macros)} macros to {args.macros}")


if __name__ == "__main__":
    main()
