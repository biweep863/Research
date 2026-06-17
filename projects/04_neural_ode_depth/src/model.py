"""Tiny continuous-depth transformer.

Architecture:
    embed → ODE-block(d_model, n_heads) on t∈[0,1] → unembed.

The ODE vector field is one transformer block (self-attention + FFN) shared
across all "depths"; integration over t∈[0,1] replaces stacking. Discrete
baselines simply call the same block in a loop without `odeint`.

This file deliberately keeps the model tiny so the local config trains in
~1-2 minutes on CPU/MPS.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


def causal_mask(T: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


class Block(nn.Module):
    """Pre-norm transformer block; output has the *delta* convention so that
    integrating it gives a residual update suitable for an ODE field."""
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        h = d_model * ffn_mult
        self.ffn = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Linear(h, d_model))

    def forward(self, x):
        T = x.size(1)
        m = causal_mask(T, x.device)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=m,
                         need_weights=False)
        # residual is added *outside* in the ODE field (we return only delta)
        f = self.ffn(self.ln2(x + a)) + a
        return f


class ODEFunc(nn.Module):
    def __init__(self, block: Block):
        super().__init__()
        self.block = block
        self.nfe = 0  # number-of-function-evaluations counter

    def forward(self, t, x):
        self.nfe += 1
        return self.block(x)


class NODETransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, ffn_mult: int = 4,
                 max_len: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.block = Block(d_model, n_heads, ffn_mult)
        self.func = ODEFunc(self.block)
        self.head = nn.Linear(d_model, vocab_size)

    def _embed(self, idx):
        B, T = idx.shape
        p = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        return self.tok(idx) + self.pos(p)

    def forward_continuous(self, idx, solver: str, atol: float = 1e-3, rtol: float = 1e-3,
                           step_size: float | None = None):
        h0 = self._embed(idx)
        t = torch.tensor([0.0, 1.0], device=h0.device)
        self.func.nfe = 0
        kw = dict(method=solver, atol=atol, rtol=rtol)
        if step_size is not None and solver in ("euler", "rk4"):
            kw = dict(method=solver, options={"step_size": step_size})
        out = odeint(self.func, h0, t, **kw)
        h = out[-1]
        return self.head(h), self.func.nfe

    def forward_discrete(self, idx, depth: int):
        h = self._embed(idx)
        nfe = 0
        for _ in range(depth):
            h = h + self.block(h)
            nfe += 1
        return self.head(h), nfe
