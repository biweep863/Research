"""Figures + LaTeX macros for the mechanistic-logic project."""
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


def plot_probe(res, short_tex, out_dir, macros):
    layers = list(range(res["layers"]))
    fig, ax = col_fig(height_ratio=0.7)
    if "probe_english" in res:
        ax.plot(layers, res["probe_english"], marker="o", label="English")
    if "probe_conlang" in res:
        ax.plot(layers, res["probe_conlang"], marker="s", label="Conlang")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("probe accuracy")
    ax.set_title("Logical-validity probe per layer")
    ax.legend()
    fig.savefig(out_dir / f"probe_{short_tex}.pdf"); plt.close(fig)

    fig, ax = col_fig(height_ratio=0.7)
    ax.plot(layers, res["cross_en2cl"], marker="o", label="train en → test conlang")
    ax.plot(layers, res["cross_cl2en"], marker="s", label="train conlang → test en")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("cross-surface accuracy")
    ax.set_title("Cross-surface probe transfer")
    ax.legend()
    fig.savefig(out_dir / f"cross_{short_tex}.pdf"); plt.close(fig)

    # macros
    en = res.get("probe_english", [0])
    cl = res.get("probe_conlang", [0])
    macros += [
        (f"{short_tex}ProbeEnPeak", f"{max(en):.3f}"),
        (f"{short_tex}ProbeClPeak", f"{max(cl):.3f}"),
        (f"{short_tex}CrossEnCl", f"{max(res['cross_en2cl']):.3f}"),
        (f"{short_tex}CrossClEn", f"{max(res['cross_cl2en']):.3f}"),
        (f"{short_tex}ProbeEnPeakLayer", f"{int(np.argmax(en))}"),
        (f"{short_tex}ProbeClPeakLayer", f"{int(np.argmax(cl))}"),
    ]


def plot_ablation(res, short_tex, out_dir, macros):
    rows = res["rows"]
    if not rows:
        return
    dacc = np.array([r["delta_acc"] for r in rows])
    dlm = np.array([r["delta_lm"] for r in rows])
    # scatter Δ-logic vs Δ-LM
    fig, ax = col_fig(height_ratio=0.85)
    sc = ax.scatter(dlm, dacc, c=[r["layer"] for r in rows], cmap="viridis", s=18)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("ΔLM cross-entropy")
    ax.set_ylabel("ΔLogic accuracy")
    ax.set_title("Per-head ablation: logic vs LM impact")
    fig.colorbar(sc, ax=ax, label="layer")
    fig.savefig(out_dir / f"ablate_scatter_{short_tex}.pdf"); plt.close(fig)

    # bar of mean |Δacc| per layer
    layers = sorted({r["layer"] for r in rows})
    mean_abs = [np.mean([abs(r["delta_acc"]) for r in rows if r["layer"] == l]) for l in layers]
    fig, ax = col_fig(height_ratio=0.7)
    ax.bar(layers, mean_abs)
    ax.set_xlabel("layer"); ax.set_ylabel("mean |ΔAcc| over heads")
    ax.set_title("Layer-wise sensitivity to head ablation")
    fig.savefig(out_dir / f"ablate_bar_{short_tex}.pdf"); plt.close(fig)

    # critical heads: most negative Δacc with small Δlm
    crit = sorted(rows, key=lambda r: r["delta_acc"])[:3]
    macros += [
        (f"{short_tex}BaselineAcc", f"{res['baseline_acc']:.3f}"),
        (f"{short_tex}BaselineLM", f"{res['baseline_lm']:.3f}"),
        (f"{short_tex}WorstHeadLayer", f"{crit[0]['layer']}"),
        (f"{short_tex}WorstHeadIndex", f"{crit[0]['head']}"),
        (f"{short_tex}WorstHeadDeltaAcc", f"{crit[0]['delta_acc']:.3f}"),
        (f"{short_tex}WorstHeadDeltaLM", f"{crit[0]['delta_lm']:.3f}"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()

    apply_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []

    for path in sorted(Path(args.results).glob("probe_*.json")):
        short = path.stem.replace("probe_", "")
        short_tex = latex_safe(short)
        plot_probe(json.loads(path.read_text()), short_tex, out, macros)

    for path in sorted(Path(args.results).glob("ablate_*.json")):
        short = path.stem.replace("ablate_", "")
        short_tex = latex_safe(short)
        plot_ablation(json.loads(path.read_text()), short_tex, out, macros)

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for name, val in macros:
            f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    print(f"[plot_logic] wrote {len(macros)} macros to {args.macros}")


if __name__ == "__main__":
    main()
