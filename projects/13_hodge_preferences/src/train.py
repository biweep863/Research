"""Full-batch training of a preference model with per-component tracking.

Losses: 'sq' = mean squared error to a target flow (the setting of the NTK
theory), 'bt' = Bradley-Terry binary cross-entropy on vote counts
(sigmoid(g_ij) = P(j beats i)).  At every logged step we record, for each
Hodge component c of the *training* graph, the fitted fraction
1 - |P_c (y - g)|^2 / |P_c y|^2 of a reference flow y (the clean truth), and
optionally held-out sign accuracy / MSE against the clean truth.
"""
from __future__ import annotations

import numpy as np
import torch

from .hodge.decompose import COMPONENTS


def fit(model, Xi, Xj, P: dict, y_ref: np.ndarray, *, loss: str = "sq", y_obs: np.ndarray | None = None,
        wins: np.ndarray | None = None, tot: np.ndarray | None = None, lr: float = 1.0, steps: int = 1000,
        log_every: int = 10, log_steps=None, alpha: float = 1.0, heldout: tuple | None = None, optimizer: str = "gd",
        track_kernel=None) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    lr_eff = lr / alpha ** 2
    opt = torch.optim.SGD(params, lr=lr_eff) if optimizer == "gd" else torch.optim.Adam(params, lr=lr_eff)
    Xi_t, Xj_t = torch.as_tensor(Xi, dtype=torch.float32), torch.as_tensor(Xj, dtype=torch.float32)
    y_obs_t = torch.as_tensor(y_ref if y_obs is None else y_obs, dtype=torch.float32)
    if loss == "bt":
        w_t, n_t = torch.as_tensor(wins, dtype=torch.float32), torch.as_tensor(tot, dtype=torch.float32)
    yc = {c: P[c] @ y_ref for c in COMPONENTS}
    yn = {c: float(yc[c] @ yc[c]) for c in COMPONENTS}
    log_set = set(int(x) for x in log_steps) if log_steps is not None else None
    log = {"step": [], "loss": [], **{f"fit_{c}": [] for c in COMPONENTS}, "ho_acc": [], "ho_mse": [], "kernel": []}
    for t in range(steps + 1):
        g = model(Xi_t, Xj_t)
        if loss == "sq":
            L = 0.5 * torch.mean((g - y_obs_t) ** 2)
        else:
            L = torch.mean(n_t * torch.nn.functional.softplus(g) - w_t * g) / torch.mean(n_t)
        do_log = (t in log_set) if log_set is not None else (t % log_every == 0)
        if do_log or t == steps:
            gn = g.detach().double().numpy()
            r = y_ref - gn
            log["step"].append(t); log["loss"].append(float(L.detach()))
            for c in COMPONENTS:
                log[f"fit_{c}"].append(1 - float(np.sum((P[c] @ r) ** 2)) / yn[c] if yn[c] > 0 else float("nan"))
            if heldout is not None:
                hi, hj, hy = heldout
                with torch.no_grad():
                    hg = model(torch.as_tensor(hi, dtype=torch.float32), torch.as_tensor(hj, dtype=torch.float32)).double().numpy()
                log["ho_acc"].append(float(np.mean(np.sign(hg) == np.sign(hy))))
                log["ho_mse"].append(float(np.mean((hg - hy) ** 2)))
            if track_kernel is not None:
                log["kernel"].append(track_kernel(model, t))
        if t == steps:
            break
        opt.zero_grad(); L.backward(); opt.step()
    return log


def time_to(log: dict, comp: str, level: float) -> float:
    for s, v in zip(log["step"], log[f"fit_{comp}"]):
        if v >= level:
            return float(s)
    return float("inf")
