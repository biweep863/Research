"""Helpers shared by the experiment entry points."""
from __future__ import annotations

import numpy as np
import torch

from src.data.synthetic import make_task
from src.hodge.decompose import COMPONENTS
from src.kernels.edge_ntk import antisym_edge_ntk, scalar_edge_ntk
from src.kernels.empirical import empirical_edge_ntk
from src.models.nets import build


def log_steps(steps: int, n: int) -> np.ndarray:
    return np.unique(np.concatenate([[0], np.round(np.geomspace(1, steps, n)).astype(int)]))


def task_from_cfg(cfg, seed):
    t = cfg["task"]
    return make_task(t["n_items"], t["feat_dim"], t["density"], t["beta_g"], t["beta_c"],
                     np.random.default_rng(seed), t["heldout_frac"])


def edge_ntk(kind, T, net_cfg, seed=0):
    """Kernel used to set the step size and the linearised prediction:
    analytic for scalar/antisym, empirical at init for GPM."""
    if kind == "scalar":
        return scalar_edge_ntk(T.g_train, T.X, net_cfg["depth"])
    if kind == "antisym":
        return antisym_edge_ntk(T.g_train, T.X, net_cfg["depth"])
    m = build("gpm", T.X.shape[1], net_cfg["width"], net_cfg["depth"], seed=seed, k=net_cfg.get("gpm_k", 4))
    i, j = T.g_train.edges[:, 0], T.g_train.edges[:, 1]
    return empirical_edge_ntk(m, torch.tensor(T.X[i], dtype=torch.float32), torch.tensor(T.X[j], dtype=torch.float32))


def first_crossing(steps, values, level):
    for s, v in zip(steps, values):
        if v >= level:
            return float(s)
    return None


def summarize_fit(log, levels=(0.5, 0.9)):
    return {c: {str(l): first_crossing(log["step"], log[f"fit_{c}"], l) for l in levels} for c in COMPONENTS}
