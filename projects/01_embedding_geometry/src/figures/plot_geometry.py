"""Generate figures and macros from the geometry sweep results."""
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


def to_grid(rows, key):
    ctxs = sorted({r["ctx"] for r in rows})
    layers = sorted({r["layer"] for r in rows})
    M = np.full((len(layers), len(ctxs)), np.nan)
    for r in rows:
        i = layers.index(r["layer"]); j = ctxs.index(r["ctx"])
        M[i, j] = r[key]
    return M, ctxs, layers


def heat(M, ctxs, layers, title, out, cmap):
    fig, ax = col_fig(height_ratio=0.85)
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xticks(range(len(ctxs))); ax.set_xticklabels(ctxs, rotation=30)
    ax.set_yticks(range(0, len(layers), max(1, len(layers)//6)))
    ax.set_yticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers)//6))])
    ax.set_xlabel("context length")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out)
    plt.close(fig)


def lines(M, ctxs, layers, title, ylabel, out, picks=(2, 5, 8, 11)):
    fig, ax = col_fig(height_ratio=0.7)
    for l in picks:
        if l >= len(layers):
            continue
        ax.plot(ctxs, M[l, :], marker="o", label=f"layer {layers[l]}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()

    apply_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []

    digit_words = {"0":"Zero","1":"One","2":"Two","3":"Three","4":"Four","5":"Five",
                   "6":"Six","7":"Seven","8":"Eight","9":"Nine","_":"","-":""}
    def latex_safe(s):
        return "".join(digit_words.get(c, c) for c in s)

    for json_path in sorted(Path(args.results).glob("sweep_*.json")):
        short = json_path.stem.replace("sweep_", "")
        short_tex = latex_safe(short)
        rows = json.loads(json_path.read_text())

        ID, ctxs, layers = to_grid(rows, "twonn")
        AN, _, _ = to_grid(rows, "anisotropy")
        CU, _, _ = to_grid(rows, "curvature")
        ER, _, _ = to_grid(rows, "eff_rank")

        heat(ID, ctxs, layers, f"Intrinsic dim ({short})", out / f"id_heat_{short}.pdf", "viridis")
        heat(AN, ctxs, layers, f"Anisotropy ({short})", out / f"aniso_heat_{short}.pdf", "magma")
        heat(CU, ctxs, layers, f"Curvature proxy ({short})", out / f"curv_heat_{short}.pdf", "cividis")
        lines(ID, ctxs, layers, f"ID vs context length ({short})", "intrinsic dim",
              out / f"id_lines_{short}.pdf")
        lines(AN, ctxs, layers, f"Anisotropy vs context length ({short})", "mean cosine",
              out / f"aniso_lines_{short}.pdf")

        # numeric macros
        mid = len(layers) // 2
        last_ctx = -1
        macros += [
            (f"{short_tex}IDmidShort", f"{ID[mid, 0]:.2f}"),
            (f"{short_tex}IDmidLong", f"{ID[mid, last_ctx]:.2f}"),
            (f"{short_tex}AnisoMidShort", f"{AN[mid, 0]:.3f}"),
            (f"{short_tex}AnisoMidLong", f"{AN[mid, last_ctx]:.3f}"),
            (f"{short_tex}EffRankDrop", f"{(ER[mid, 0] - ER[mid, last_ctx]):.2f}"),
            (f"{short_tex}MaxCtx", f"{ctxs[-1]}"),
            (f"{short_tex}NumLayers", f"{len(layers)-1}"),
        ]

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for name, val in macros:
            f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    print(f"[plot_geometry] wrote {len(macros)} macros to {args.macros}")


if __name__ == "__main__":
    main()
