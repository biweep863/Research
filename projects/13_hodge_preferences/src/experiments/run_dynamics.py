"""Experiment 2 -- which Hodge component does gradient descent fit first?

On the shared synthetic task, train scalar, antisymmetric and GPM models with
the squared loss (to the clean flow) and the Bradley-Terry loss (to votes
drawn from the clean flow) and log the fitted fraction of each component on a
log-spaced grid of steps; compare with the linearised NTK prediction and with
the target-specific Rayleigh quotients.  Writes results/dynamics.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from .shared import edge_ntk, first_crossing, log_steps, summarize_fit, task_from_cfg
from src.data.synthetic import votes
from src.hodge.decompose import COMPONENTS
from src.kernels.edge_ntk import linearised_fit, rayleigh
from src.models.nets import build
from src.train import fit


def main():
    cfg, out = load_cfg()
    dy, net = cfg["dynamics"], cfg["net"]; seed = cfg.get("seed", 0)
    steps = log_steps(dy["steps"], dy["n_log"])
    runs = []
    for s in range(dy["seeds"]):
        T = task_from_cfg(cfg, seed + s)
        i, j = T.g_train.edges[:, 0], T.g_train.edges[:, 1]
        E = T.g_train.n_edges
        ho = (T.X[T.heldout[:, 0]], T.X[T.heldout[:, 1]], T.y_heldout)
        for kind in dy["models"]:
            Th = edge_ntk(kind, T, net, seed + s)
            lam = float(np.linalg.eigvalsh(Th).max())
            eta = dy["lr_frac"] * E / lam
            ray = {c: rayleigh(Th, T.P[c] @ T.y_train) for c in COMPONENTS}
            lin = linearised_fit(Th, T.y_train, T.P, eta, steps)
            for loss in dy["losses"]:
                m = build(kind, T.X.shape[1], net["width"], net["depth"], seed=seed + s,
                          **({"k": net["gpm_k"]} if kind == "gpm" else {}))
                if loss == "sq":
                    log = fit(m, T.X[i], T.X[j], T.P, T.y_train, lr=eta, steps=dy["steps"], log_steps=steps, heldout=ho)
                else:
                    w, n = votes(T.y_train, dy["n_votes"], np.random.default_rng(seed + s + 99))
                    # BT curvature at g=0 is p(1-p)=1/4, so the linearised rate is eta/4: use 4x.
                    log = fit(m, T.X[i], T.X[j], T.P, T.y_train, loss="bt", wins=w, tot=n,
                              lr=4 * eta * dy["bt_lr"], steps=dy["steps"], log_steps=steps, heldout=ho)
                runs.append({"seed": s, "model": kind, "loss": loss, "eta": eta, "lambda_max": lam, "rayleigh": ray,
                             "fractions": T.fractions, "log": log, "crossings": summarize_fit(log),
                             "linearised": lin if loss == "sq" else None,
                             "lin_crossings": {c: {str(l): first_crossing(steps, lin[c], l) for l in (0.5, 0.9)} for c in COMPONENTS}})
                cr = runs[-1]["crossings"]
                print(f"[dynamics] seed={s} {kind:7s} {loss}  t90 grad={cr['grad']['0.9']} curl={cr['curl']['0.9']} "
                      f"harm={cr['harm']['0.9']}  final ho_acc={log['ho_acc'][-1]:.3f}  "
                      f"rayleigh g/c/h={ray['grad']:.1f}/{ray['curl']:.2f}/{ray['harm']:.2f}")
    dump(out, "dynamics.json", {"steps": steps.tolist(), "runs": runs, "task": cfg["task"]})


if __name__ == "__main__":
    main()
