"""Empirical edge NTK of a torch preference model: J J^T over edges."""
from __future__ import annotations

import numpy as np
import torch
from torch.func import functional_call, jacrev, vmap


def empirical_edge_ntk(model: torch.nn.Module, Xi: torch.Tensor, Xj: torch.Tensor, chunk: int = 256) -> np.ndarray:
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_buffers()}
    frozen = {k: v.detach() for k, v in model.named_parameters() if not v.requires_grad}

    def f(p, xi, xj):
        out = functional_call(model, {**p, **frozen, **buffers}, (xi[None], xj[None]))
        return out.reshape(())

    jac_fn = vmap(jacrev(f), in_dims=(None, 0, 0))
    rows = []
    for s in range(0, len(Xi), chunk):
        J = jac_fn(params, Xi[s:s + chunk], Xj[s:s + chunk])
        rows.append(torch.cat([J[k].reshape(J[k].shape[0], -1) for k in params], 1))
    J = torch.cat(rows, 0)
    return (J @ J.T).double().numpy()
