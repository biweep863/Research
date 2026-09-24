"""Experiment 4 -- does the order survive finite width and feature learning?

Antisymmetric model on the shared task, squared loss, sweeping width and
the output scale alpha (small alpha = rich / feature learning, large = lazy).
Logs the per-component fit, the relative kernel drift |Theta_t - Theta_0| /
|Theta_0| and the Hodge rates of the evolving empirical kernel.
Writes results/width.json.
"""
from __future__ import annotations

import numpy as np
import torch

from .common import dump, load_cfg
from .shared import edge_ntk, log_steps, summarize_fit, task_from_cfg
from src.kernels.edge_ntk import hodge_rates
from src.kernels.empirical import empirical_edge_ntk
from src.models.nets import build
from src.train import fit


def main():
    cfg, out = load_cfg()
    wc, net = cfg["width"], cfg["net"]; seed = cfg.get("seed", 0)
    steps = log_steps(wc["steps"], wc["n_log"])
    kern_steps = set(int(x) for x in steps[:: max(1, len(steps) // 8)]) | {int(steps[-1])}
    rows = []
    for s in range(wc["seeds"]):
        T = task_from_cfg(cfg, seed + s)
        i, j = T.g_train.edges[:, 0], T.g_train.edges[:, 1]
        Xi = torch.tensor(T.X[i], dtype=torch.float32); Xj = torch.tensor(T.X[j], dtype=torch.float32)
        E = T.g_train.n_edges
        Th = edge_ntk(wc["model"], T, net)
        eta = cfg["dynamics"]["lr_frac"] * E / float(np.linalg.eigvalsh(Th).max())
        ho = (T.X[T.heldout[:, 0]], T.X[T.heldout[:, 1]], T.y_heldout)
        for w in wc["widths"]:
            for a in wc["alphas"]:
                m = build(wc["model"], T.X.shape[1], w, net["depth"], alpha=a, seed=seed + s)
                K0 = {}

                def tk(model, t):
                    if t not in kern_steps:
                        return None
                    K = empirical_edge_ntk(model, Xi, Xj) / a ** 2
                    if "0" not in K0:
                        K0["0"] = K
                    r = hodge_rates(K, T.P)
                    return {"t": t, "drift": float(np.linalg.norm(K - K0["0"]) / np.linalg.norm(K0["0"])), "rates": r}

                log = fit(m, T.X[i], T.X[j], T.P, T.y_train, lr=eta, steps=wc["steps"], log_steps=steps,
                          alpha=a, heldout=ho, track_kernel=tk)
                kl = [k for k in log["kernel"] if k is not None]
                cr = summarize_fit(log)
                g90, c90 = cr["grad"]["0.9"], cr["curl"]["0.9"]
                rows.append({"seed": s, "width": w, "alpha": a, "crossings": cr, "kernel": kl,
                             "final_fit": {c: log[f"fit_{c}"][-1] for c in ("grad", "curl", "harm")},
                             "ho_acc_final": log["ho_acc"][-1], "order_holds": (g90 is not None) and (c90 is None or g90 <= c90),
                             "fit_log": {k: log[k] for k in ("step", "fit_grad", "fit_curl", "fit_harm")}})
                print(f"[width] seed={s} w={w:5d} alpha={a:5.2f}  t90 grad={g90} curl={c90}  "
                      f"drift={kl[-1]['drift']:.3f}  ratio_end={kl[-1]['rates']['grad'] / max(kl[-1]['rates']['curl'], 1e-12):.1f}  "
                      f"ho_acc={log['ho_acc'][-1]:.3f}")
    dump(out, "width.json", {"rows": rows, "steps": steps.tolist()})


if __name__ == "__main__":
    main()
