"""Figures + LaTeX macros for the optimal-stopping paper.

Reads results/montecarlo.json and results/agents.json; writes PDFs to
paper/figures and numeric constants to paper/macros.tex. Every number quoted
in the paper is emitted here, so recomputing the figures rewrites the prose.
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

from utils.plotting import col_fig, apply_style


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()

    res = Path(args.results)
    mc = json.loads((res / "montecarlo.json").read_text())
    ag = json.loads((res / "agents.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    macros = []

    ooe = mc["one_over_e"]

    # ------------------------------------------------------------------ #
    # Fig 1: secretary success vs skip fraction (MC + closed form + 1/e)
    # ------------------------------------------------------------------ #
    big = max(mc["secretary_curves"], key=lambda c: c["n"])
    fig, ax = col_fig(height_ratio=0.75)
    ax.plot(big["grid_x"], big["analytic_success"], "-", label="closed form", color="C0")
    ax.plot(big["grid_x"], big["mc_success"], "o", ms=2.5, label="Monte Carlo", color="C1")
    ax.axvline(ooe, ls="--", lw=0.8, color="k", label="$1/e\\approx0.368$")
    ax.set_xlabel("skip fraction $x=r/n$")
    ax.set_ylabel("P(select the best)")
    ax.set_title(f"Secretary problem ($n={big['n']}$)")
    ax.legend(loc="lower center")
    fig.savefig(out / "secretary_curve.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 2: finite-n optimum approaching 1/e
    # ------------------------------------------------------------------ #
    ns = [c["n"] for c in mc["secretary_curves"]]
    xstars = [c["x_star"] for c in mc["secretary_curves"]]
    pstars = [c["p_star"] for c in mc["secretary_curves"]]
    fig, ax = col_fig(height_ratio=0.75)
    ax.plot(ns, xstars, "o-", label="optimal $x^*$")
    ax.plot(ns, pstars, "s-", label="optimal $P^*$")
    ax.axhline(ooe, ls="--", lw=0.8, color="k", label="$1/e$")
    ax.set_xscale("log")
    ax.set_xlabel("pool size $n$")
    ax.set_ylabel("value")
    ax.set_title("Finite-$n$ optimum $\\to 1/e$")
    ax.legend()
    fig.savefig(out / "secretary_finite_n.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 3: parking expected cost vs occupancy p (optimal / agent / naive)
    # ------------------------------------------------------------------ #
    park = mc["parking"]["by_p"]
    ps = [r["p"] for r in park]
    opt_c = [r["optimal_cost"] for r in park]
    naive_c = [r["naive37_cost"] for r in park]
    qa = {r["p"]: r for r in ag["parking"]}
    q_c = [qa[p]["qlearn_cost"]["mean"] for p in ps]
    q_lo = [qa[p]["qlearn_cost"]["ci_lo"] for p in ps]
    q_hi = [qa[p]["qlearn_cost"]["ci_hi"] for p in ps]
    fig, ax = col_fig(height_ratio=0.75)
    ax.plot(ps, opt_c, "o-", label="optimal (DP)", color="C0")
    ax.errorbar(ps, q_c, yerr=[np.array(q_c) - q_lo, np.array(q_hi) - np.array(q_c)],
                fmt="s-", ms=3, capsize=2, label="Q-learning", color="C2")
    ax.plot(ps, naive_c, "^--", label="naive 37% rule", color="C3")
    ax.set_xlabel("occupancy $p$ (prob. a spot is free)")
    ax.set_ylabel("expected distance cost")
    ax.set_title("Parking: cost vs. occupancy")
    ax.legend()
    fig.savefig(out / "parking_cost.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 4: agents recover the secretary threshold
    # ------------------------------------------------------------------ #
    sa = ag["secretary"]
    ns_a = [r["n"] for r in sa]
    q_x = [r["qlearn_x"]["mean"] for r in sa]
    re_x = [r["reinforce_x"]["mean"] for r in sa]
    fig, ax = col_fig(height_ratio=0.75)
    ax.plot(ns_a, q_x, "o-", label="Q-learning cutoff")
    ax.plot(ns_a, re_x, "s-", label="REINFORCE cutoff")
    ax.axhline(ooe, ls="--", lw=0.8, color="k", label="$1/e$")
    ax.set_xlabel("pool size $n$")
    ax.set_ylabel("learned skip fraction")
    ax.set_title("Agents rediscover the 37% rule")
    ax.legend()
    fig.savefig(out / "agents_secretary.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Fig 5: observation cost shrinks the optimal cutoff below 1/e
    # ------------------------------------------------------------------ #
    sc = mc["secretary_cost"]
    costs = [r["cost"] for r in sc]
    cx = [r["x_star"] for r in sc]
    fig, ax = col_fig(height_ratio=0.75)
    ax.plot(costs, cx, "o-")
    ax.axhline(ooe, ls="--", lw=0.8, color="k", label="$1/e$ (cost-free)")
    ax.set_xlabel("per-observation cost $c$")
    ax.set_ylabel("optimal skip fraction $x^*$")
    ax.set_title(f"Search cost pulls the cutoff below 37% ($n={sc[0]['n']}$)")
    ax.legend()
    fig.savefig(out / "secretary_cost.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ #
    # Macros
    # ------------------------------------------------------------------ #
    sec_by_n = {c["n"]: c for c in mc["secretary_curves"]}
    n_hi = max(sec_by_n)
    n_lo = min(sec_by_n)
    park_by_p = {r["p"]: r for r in park}
    p_hi = max(park_by_p)
    p_lo = min(park_by_p)
    sa_by_n = {r["n"]: r for r in sa}
    n_mid = max(sa_by_n)          # largest trained pool: closest to asymptotic

    macros += [
        ("OneOverE", f"{ooe:.3f}"),
        ("SecNhi", f"{n_hi}"),
        ("SecXstarHi", f"{sec_by_n[n_hi]['x_star']:.3f}"),
        ("SecPstarHi", f"{sec_by_n[n_hi]['p_star']:.3f}"),
        ("SecNlo", f"{n_lo}"),
        ("SecXstarLo", f"{sec_by_n[n_lo]['x_star']:.3f}"),
        ("SecPstarLo", f"{sec_by_n[n_lo]['p_star']:.3f}"),
        ("ParkPhi", f"{p_hi:.2f}"),
        ("ParkNaiveRegretHi", _pct(park_by_p[p_hi]["naive37_regret"])),
        ("ParkPlo", f"{p_lo:.2f}"),
        ("ParkNaiveRegretLo", _pct(park_by_p[p_lo]["naive37_regret"])),
        ("AgentSecN", f"{n_mid}"),
        ("AgentQx", f"{sa_by_n[n_mid]['qlearn_x']['mean']:.3f}"),
        ("AgentQregret", _pct(sa_by_n[n_mid]["qlearn_regret"]["mean"])),
        ("AgentRex", f"{sa_by_n[n_mid]['reinforce_x']['mean']:.3f}"),
        ("AgentReRegret", _pct(sa_by_n[n_mid]["reinforce_regret"]["mean"])),
        ("ParkQregretHi", _pct(qa[p_hi]["qlearn_regret"]["mean"])),
        ("ParkQregretLo", _pct(qa[p_lo]["qlearn_regret"]["mean"])),
        ("CostMax", f"{costs[-1]:.4f}"),
        ("CostMaxXstar", f"{cx[-1]:.3f}"),
    ]

    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    with open(args.macros, "w") as f:
        for n, v in macros:
            f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
    print(f"[plot_stopping] wrote 5 figures + {len(macros)} macros")


if __name__ == "__main__":
    main()
