"""Figures + LaTeX macros for the drift/selection project."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import col_fig, apply_style

DIGIT_WORD = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
              "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def _fix(name: str) -> str:
    return re.sub(r"(\d+)", lambda m: "".join(DIGIT_WORD[c] for c in m.group(1)), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--macros", default="paper/macros.tex")
    args = ap.parse_args()
    apply_style()
    R = Path(args.results)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thr = json.loads((R / "threshold_sweep.json").read_text())
    eff = json.loads((R / "effective_size.json").read_text())
    macros = []; m = macros.append

    # ---- Fig 1: success probability vs N, one curve per K
    fig, ax = col_fig(height_ratio=0.75)
    for row in thr["rows"]:
        ax.semilogx(row["n_grid"], row["p_success"], "o-", ms=3, label=f"K={row['K']}")
        if np.isfinite(row["n_c"]):
            ax.axvline(row["n_c"], ls=":", lw=0.6, color=ax.lines[-1].get_color())
    ax.axhline(0.5, color="grey", lw=0.6, ls="--")
    ax.set_xlabel("population size N"); ax.set_ylabel("P(reach global optimum)")
    ax.set_title("Success probability vs population size")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(out / "success_vs_n.pdf"); plt.close(fig)

    # ---- Fig 2: N_c(K), log-linear vs linear fit
    valid = [r for r in thr["rows"] if np.isfinite(r["n_c"])]
    Ks = np.array([r["K"] for r in valid], dtype=float)
    Ncs = np.array([r["n_c"] for r in valid])
    lo = np.array([r["n_c_ci"][0] for r in valid])
    hi = np.array([r["n_c_ci"][1] for r in valid])
    fig, ax = col_fig(height_ratio=0.75)
    yerr = np.abs(np.vstack([Ncs - lo, hi - Ncs]))
    ax.errorbar(Ks, Ncs, yerr=yerr, fmt="o", capsize=2, color="#1565c0", label="$N_c$ (fitted)")
    if thr.get("fit"):
        kk = np.linspace(Ks.min(), Ks.max(), 50)
        f = thr["fit"]
        ax.plot(kk, f["linear"]["intercept"] + f["linear"]["slope"] * kk, "--", color="#2e7d32",
               label=f"linear (AIC {f['linear']['aic']:.1f})")
        ax.plot(kk, np.exp(f["log_linear"]["intercept"] + f["log_linear"]["slope"] * kk), "-",
               color="#ef6c00", label=f"log-linear (AIC {f['log_linear']['aic']:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("landscape ruggedness K"); ax.set_ylabel("critical population size $N_c$")
    ax.set_title("Does $N_c$ scale log-linearly with K?")
    ax.legend(fontsize=6)
    fig.savefig(out / "nc_vs_k.pdf"); plt.close(fig)

    # ---- Fig 3: s(K) decay and Kimura-implied N_c vs the directly-fitted N_c
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.4))
    eK = np.array([r["K"] for r in eff["rows"]], dtype=float)
    es = np.array([r["s"] for r in eff["rows"]])
    a1.semilogy(eK, np.maximum(es, 1e-6), "o-", color="#6a1b9a")
    a1.set_xlabel("K"); a1.set_ylabel("selection coefficient s(K)")
    a1.set_title("Mean beneficial-mutation effect size")

    nck = np.array([r["n_c_kimura"] for r in eff["rows"]])
    a2.semilogy(eK, nck, "s--", color="#6a1b9a", label="$N_c$ (Kimura, from s(K))")
    if len(valid):
        a2.semilogy(Ks, Ncs, "o-", color="#1565c0", label="$N_c$ (fitted from GA runs)")
    a2.set_xlabel("K"); a2.set_ylabel("$N_c$"); a2.set_title("Two independent estimates of $N_c$")
    a2.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "kimura_check.pdf"); plt.close(fig)

    # ---- macros
    m(("SweepL", str(thr["L"]))); m(("SweepReplicates", str(thr["replicates"])))
    m(("SweepNumK", str(len(thr["rows"]))))
    for row in thr["rows"]:
        k = row["K"]
        m((f"NcK{k}", f"{row['n_c']:.1f}" if np.isfinite(row["n_c"]) else "--"))
        m((f"PSuccessMaxK{k}", f"{max(row['p_success']):.2f}"))
    if thr.get("fit"):
        f = thr["fit"]
        m(("FitLinearAic", f"{f['linear']['aic']:.1f}"))
        m(("FitLogLinearAic", f"{f['log_linear']['aic']:.1f}"))
        m(("FitLogLinearSlope", f"{f['log_linear']['slope']:.3f}"))
        m(("FitPreferred", "log-linear" if f["log_linear_preferred"] else "linear"))
    if eff.get("fit"):
        m(("SSlope", f"{eff['fit']['s_log_linear_slope']:.3f}"))
        m(("SFitIntercept", f"{eff['fit']['s_log_linear_intercept']:.2f}"))
    for row in eff["rows"]:
        k = row["K"]
        m((f"SK{k}", f"{row['s']:.4f}")); m((f"NcKimuraK{k}", f"{row['n_c_kimura']:.1f}"))

    macros = [(_fix(k), v) for k, v in macros]
    assert all(not re.search(r"\d", k) for k, _ in macros), [k for k, _ in macros if re.search(r"\d", k)]
    Path(args.macros).parent.mkdir(parents=True, exist_ok=True)
    Path(args.macros).write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")
    print(f"[figures] wrote {len(macros)} macros and 3 figures")


if __name__ == "__main__":
    main()
