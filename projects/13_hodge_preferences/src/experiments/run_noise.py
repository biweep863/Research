"""Experiment 3 -- random noise lives in cycle space; early stopping filters it.

(a) Fraction of i.i.d. Gaussian edge noise that lands in gradient space vs
    the prediction (|V|-1)/|E|, over graph sizes and densities.
(b) Train on a noisy flow (Gaussian sigma grid, squared loss) and on noisy
    votes (flip grid, Bradley-Terry loss); log the fit to the *clean* flow per
    component, clean training-edge error, and held-out sign accuracy against
    the clean truth.  The best early-stopping step is compared with the
    NTK-predicted step t* at which the gradient component is 95 % fitted
    under the linearised dynamics.  Baselines: the fully trained model and
    HodgeRank on the noisy flow (a feature-free transitive fit).
Writes results/noise.json.
"""
from __future__ import annotations

import numpy as np

from .common import dump, load_cfg
from .shared import edge_ntk, first_crossing, log_steps, task_from_cfg
from src.data.synthetic import gaussian_noise, votes
from src.hodge.decompose import dims, hodgerank, projectors
from src.hodge.graph import random_graph
from src.kernels.edge_ntk import linearised_fit
from src.models.nets import build
from src.train import fit


def main():
    cfg, out = load_cfg()
    nz, net = cfg["noise"], cfg["net"]; seed = cfg.get("seed", 0)
    fc = nz["fraction_check"]
    frac_rows = []
    for n in fc["n_items"]:
        for dens in fc["density"]:
            rng = np.random.default_rng(seed + n + int(100 * dens))
            g = random_graph(n, dens, rng); P = projectors(g); dm = dims(g, P)
            z = rng.normal(size=(fc["trials"], g.n_edges))
            fr = np.sum((z @ P["grad"]) * z, 1) / np.sum(z * z, 1)
            frac_rows.append({"n": n, "density": dens, "E": g.n_edges, "dims": dm, "theory": (n - 1) / g.n_edges,
                              "mean": float(fr.mean()), "std": float(fr.std())})
            print(f"[noise] n={n:3d} dens={dens:.1f} E={g.n_edges:5d}  grad fraction {fr.mean():.4f} ± {fr.std():.4f} "
                  f"(theory {(n - 1) / g.n_edges:.4f})")

    steps = log_steps(nz["steps"], nz["n_log"])
    runs = []
    for s in range(nz["seeds"]):
        T = task_from_cfg(cfg, seed + s)
        i, j = T.g_train.edges[:, 0], T.g_train.edges[:, 1]
        E = T.g_train.n_edges
        ho = (T.X[T.heldout[:, 0]], T.X[T.heldout[:, 1]], T.y_heldout)
        for kind in nz["models"]:
            Th = edge_ntk(kind, T, net, seed + s)
            eta = cfg["dynamics"]["lr_frac"] * E / float(np.linalg.eigvalsh(Th).max())
            lin = linearised_fit(Th, T.y_train, T.P, eta, np.arange(0, nz["steps"] + 1))
            t_star = first_crossing(range(nz["steps"] + 1), lin["grad"], 0.95)
            settings = [("gauss", sg) for sg in nz["sigma_grid"]] + [("flip", f) for f in nz["flip_grid"]]
            for mode, level in settings:
                rng = np.random.default_rng(seed + s + int(1000 * level) + (7 if mode == "flip" else 0))
                m = build(kind, T.X.shape[1], net["width"], net["depth"], seed=seed + s,
                          **({"k": net["gpm_k"]} if kind == "gpm" else {}))
                if mode == "gauss":
                    y_obs = gaussian_noise(T.y_train, level, rng)
                    log = fit(m, T.X[i], T.X[j], T.P, T.y_train, y_obs=y_obs, lr=eta, steps=nz["steps"],
                              log_steps=steps, heldout=ho)
                    hr = hodgerank(T.g_train, y_obs)
                else:
                    w, n = votes(T.y_train, nz["n_votes"], rng, flip=level)
                    log = fit(m, T.X[i], T.X[j], T.P, T.y_train, loss="bt", wins=w, tot=n, lr=4 * eta,
                              steps=nz["steps"], log_steps=steps, heldout=ho)
                    y_obs = np.log((w + 0.5) / (n - w + 0.5))
                    hr = hodgerank(T.g_train, y_obs)
                acc = np.array(log["ho_acc"])
                k_best = int(np.argmax(acc))
                k_star = int(np.argmin(np.abs(np.array(log["step"]) - (t_star or 0))))
                y_obs_noise = y_obs - T.y_train
                runs.append({"seed": s, "model": kind, "mode": mode, "level": level, "t_star": t_star,
                             "noise_grad_fraction": float(np.sum((T.P["grad"] @ y_obs_noise) ** 2) / max(np.sum(y_obs_noise ** 2), 1e-12)),
                             "acc_final": float(acc[-1]), "acc_best": float(acc[k_best]), "step_best": int(log["step"][k_best]),
                             "acc_at_t_star": float(acc[k_star]),
                             "hodgerank_train_sign_acc": float(np.mean(np.sign(hr[j] - hr[i]) == np.sign(T.y_train))),
                             "log": {k: log[k] for k in ("step", "ho_acc", "ho_mse", "fit_grad", "fit_curl", "fit_harm")}})
                r = runs[-1]
                print(f"[noise] seed={s} {kind:7s} {mode:5s} {level:.2f}  acc final={r['acc_final']:.3f} best={r['acc_best']:.3f}"
                      f"@{r['step_best']}  at t*={r['acc_at_t_star']:.3f}@{t_star}  noise∈grad={r['noise_grad_fraction']:.3f}")
    dump(out, "noise.json", {"fraction_rows": frac_rows, "steps": steps.tolist(), "runs": runs,
                             "E": int(E), "n": cfg["task"]["n_items"]})


if __name__ == "__main__":
    main()
