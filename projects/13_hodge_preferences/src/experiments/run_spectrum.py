"""Experiment 1 -- Hodge structure of the edge NTK.

Analytic infinite-width edge NTKs of the scalar and antisymmetric models over
a grid of graph size, density, feature dimension and depth: component
dimensions, mean Hodge rates kappa_c, the ratio kappa_grad / kappa_cyc, and
the spectral mass of the top eigenvectors per component.  Then empirical
NTKs at initialisation for the three models at several widths, compared with
the analytic kernel.  Writes results/spectrum.json.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import dump, load_cfg
from src.hodge.decompose import dims, projectors
from src.hodge.graph import random_graph
from src.kernels.edge_ntk import antisym_edge_ntk, hodge_rates, scalar_edge_ntk, spectral_mass
from src.kernels.empirical import empirical_edge_ntk
from src.models.nets import build


def cyc_rate(r, dm):
    num = sum(r[c] * dm[c] for c in ("curl", "harm") if dm[c] > 0)
    den = sum(dm[c] for c in ("curl", "harm"))
    return num / den if den else float("nan")


def main():
    cfg, out = load_cfg()
    sp = cfg["spectrum"]; seed = cfg.get("seed", 0)
    rows = []
    for n in sp["n_items"]:
        for dens in sp["density"]:
            for d in sp["feat_dim"]:
                for depth in sp["depth"]:
                    for r in range(sp["reps"]):
                        rng = np.random.default_rng(seed + 7919 * r + 31 * n + d)
                        X = rng.normal(size=(n, d)); X -= X.mean(0)
                        g = random_graph(n, dens, rng); P = projectors(g); dm = dims(g, P)
                        rs = hodge_rates(scalar_edge_ntk(g, X, depth), P)
                        ra = hodge_rates(antisym_edge_ntk(g, X, depth), P)
                        rows.append({"n": n, "density": dens, "d": d, "depth": depth, "rep": r, "E": g.n_edges,
                                     "mean_degree": 2 * g.n_edges / n, "dims": dm, "scalar": rs, "antisym": ra,
                                     "ratio_antisym": ra["grad"] / cyc_rate(ra, dm),
                                     "ratio_curl_harm": (ra["curl"] / ra["harm"]) if dm["harm"] and dm["curl"] else None})
                print(f"[spectrum] n={n} dens={dens:.2f} d={d:2d}  E={rows[-1]['E']:4d} dims={rows[-1]['dims']}  "
                      f"antisym grad/cyc={rows[-1]['ratio_antisym']:.1f}  scalar cyc={rows[-1]['scalar']['curl']:.1e}")

    # Spectral mass of the top eigenvectors for one canonical graph.
    e = sp["empirical"]; rng = np.random.default_rng(seed + 1)
    X = rng.normal(size=(e["n_items"], e["feat_dim"])); X -= X.mean(0)
    g = random_graph(e["n_items"], e["density"], rng); P = projectors(g)
    Ta = antisym_edge_ntk(g, X, 1)
    sm = spectral_mass(Ta, P)
    canon = {"dims": dims(g, P), "E": g.n_edges, "eig": sm["eig"].tolist(),
             **{c: sm[c].tolist() for c in ("grad", "curl", "harm")}}

    # Empirical kernels at init vs analytic.
    Xi = torch.tensor(X[g.edges[:, 0]], dtype=torch.float32); Xj = torch.tensor(X[g.edges[:, 1]], dtype=torch.float32)
    Ts = scalar_edge_ntk(g, X, 1)
    emp = []
    for w in e["widths"]:
        for r in range(e["reps"]):
            for kind in ("scalar", "antisym", "gpm"):
                m = build(kind, e["feat_dim"], w, 1, seed=seed + r, **({"k": cfg["net"]["gpm_k"]} if kind == "gpm" else {}))
                Te = empirical_edge_ntk(m, Xi, Xj)
                rr = hodge_rates(Te, P)
                ref = Ts if kind == "scalar" else (Ta if kind == "antisym" else None)
                emp.append({"width": w, "rep": r, "model": kind, "rates": rr,
                            "ratio": rr["grad"] / cyc_rate(rr, canon["dims"]),
                            "rel_err_vs_analytic": float(np.linalg.norm(Te - ref) / np.linalg.norm(ref)) if ref is not None else None})
            print(f"[spectrum] empirical width={w} rep={r}: " + ", ".join(
                f"{x['model']} ratio={x['ratio']:.1f}" + (f" err={x['rel_err_vs_analytic']:.3f}" if x['rel_err_vs_analytic'] is not None else "")
                for x in emp[-3:]))
    dump(out, "spectrum.json", {"rows": rows, "canonical": canon, "empirical": emp,
                                "analytic_canonical": {"scalar": hodge_rates(Ts, P), "antisym": hodge_rates(Ta, P)}})


if __name__ == "__main__":
    main()
