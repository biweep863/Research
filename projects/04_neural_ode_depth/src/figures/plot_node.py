"""Figures + LaTeX macros for the Neural ODE depth project."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt
from utils.plotting import col_fig, apply_style


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()

    apply_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    macros = []

    data = json.loads(Path(args.results, "sweep.json").read_text())
    rows = data["rows"]

    # Pareto: latency vs accuracy
    fig, ax = col_fig(height_ratio=0.8)
    for k, marker, color in [("ode", "o", "C0"), ("discrete", "s", "C3")]:
        sel = [r for r in rows if r["kind"] == k]
        if not sel:
            continue
        xs = [r["latency_s"] for r in sel]
        ys = [r["acc"] for r in sel]
        ax.scatter(xs, ys, marker=marker, c=color, label=k, s=40)
        for r in sel:
            ax.annotate(r["name"], (r["latency_s"], r["acc"]),
                        fontsize=5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("latency (s, eval batch)")
    ax.set_ylabel("accuracy")
    ax.set_title("Latency vs accuracy: ODE vs discrete depth")
    ax.legend()
    fig.savefig(out / "pareto.pdf"); plt.close(fig)

    # Accuracy vs NFE
    fig, ax = col_fig(height_ratio=0.8)
    for k, marker in [("ode", "o"), ("discrete", "s")]:
        sel = [r for r in rows if r["kind"] == k]
        xs = [r["nfe"] for r in sel]
        ys = [r["acc"] for r in sel]
        ax.scatter(xs, ys, marker=marker, label=k, s=40)
    ax.set_xlabel("function evaluations (NFE)")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs compute (NFE)")
    ax.legend()
    fig.savefig(out / "acc_vs_nfe.pdf"); plt.close(fig)

    # Training loss
    losses = data["train_losses"]
    if losses:
        fig, ax = col_fig(height_ratio=0.7)
        ax.plot([l["step"] for l in losses], [l["loss"] for l in losses])
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.set_title("NODE-Transformer training loss")
        fig.savefig(out / "train_loss.pdf"); plt.close(fig)

    # Headline: best discrete vs best ODE matching accuracy
    best_disc = max((r for r in rows if r["kind"] == "discrete"), key=lambda r: r["acc"])
    same_acc_ode = [r for r in rows if r["kind"] == "ode" and r["acc"] >= 0.95 * best_disc["acc"]]
    fastest_ode = min(same_acc_ode, key=lambda r: r["latency_s"]) if same_acc_ode else best_disc
    speedup = best_disc["latency_s"] / max(1e-9, fastest_ode["latency_s"])
    esc = lambda s: str(s).replace("_", r"\_")
    # also compute loss-stability: how does loss change across discrete depths vs ODE?
    ode_losses = [r["loss"] for r in rows if r["kind"] == "ode"]
    disc_losses = [r["loss"] for r in rows if r["kind"] == "discrete"]
    macros += [
        ("BestDiscreteName", esc(best_disc["name"])),
        ("BestDiscreteAcc", f"{best_disc['acc']:.3f}"),
        ("BestDiscreteLatency", f"{best_disc['latency_s']:.3f}"),
        ("FastestEquivOdeName", esc(fastest_ode["name"])),
        ("FastestEquivOdeAcc", f"{fastest_ode['acc']:.3f}"),
        ("FastestEquivOdeLatency", f"{fastest_ode['latency_s']:.3f}"),
        ("OdeSpeedup", f"{speedup:.2f}"),
        ("OdeLossRange", f"{max(ode_losses) - min(ode_losses):.3f}"),
        ("DiscreteLossRange", f"{max(disc_losses) - min(disc_losses):.3f}"),
    ]
    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for name, val in macros:
            f.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
    print(f"[plot_node] wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
