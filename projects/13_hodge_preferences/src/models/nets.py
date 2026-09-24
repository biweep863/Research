"""Preference networks in the NTK parameterisation.

Every hidden layer computes  sigma_w/sqrt(fan_in) * W x + sigma_b * b  with
W, b ~ N(0, 1), matching the closed-form kernel in kernels/ntk_relu.py.
The output is  alpha * (net_theta(x) - net_theta0(x)): it starts at zero and
the scale alpha moves the model between the lazy (large alpha, kernel) and the
rich (small alpha, feature-learning) regimes (Chizat, Oyallon & Bach 2019);
the learning rate is divided by alpha^2 so the kernel-regime time scale is
the same for every alpha.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class NTKLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, sigma_w2: float = 2.0, sigma_b2: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in))
        self.b = nn.Parameter(torch.randn(d_out))
        self.sw, self.sb = math.sqrt(sigma_w2 / d_in), math.sqrt(sigma_b2)

    def forward(self, x):
        return x @ self.W.T * self.sw + self.b * self.sb


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int = 1, d_out: int = 1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [NTKLinear(d, width), nn.ReLU()]
            d = width
        # Readout without the factor 2 of the ReLU layers (sigma_w^2 = 1 on the
        # last layer would change the kernel's scale only; we keep 2 so that the
        # closed form matches exactly).
        layers.append(NTKLinear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Centered(nn.Module):
    """alpha * (m(x) - m0(x)) with m0 a frozen copy of m at initialisation."""

    def __init__(self, m: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.m, self.m0, self.alpha = m, copy.deepcopy(m), alpha
        for p in self.m0.parameters():
            p.requires_grad_(False)

    def forward(self, *a):
        return self.alpha * (self.m(*a) - self.m0(*a))


class ScalarBT(nn.Module):
    """Scalar reward: g_ij = f(x_j) - f(x_i)."""

    def __init__(self, d: int, width: int, depth: int = 1):
        super().__init__()
        self.f = MLP(d, width, depth)

    def forward(self, xi, xj):
        return (self.f(xj) - self.f(xi)).squeeze(-1)


class Antisym(nn.Module):
    """g_ij = h([x_i, x_j]) - h([x_j, x_i]): exactly antisymmetric."""

    def __init__(self, d: int, width: int, depth: int = 1):
        super().__init__()
        self.h = MLP(2 * d, width, depth)

    def forward(self, xi, xj):
        return (self.h(torch.cat([xi, xj], -1)) - self.h(torch.cat([xj, xi], -1))).squeeze(-1)


class GPM(nn.Module):
    """General Preference Model (Zhang et al. 2025): g_ij = e(x_i)^T R e(x_j)
    with R block-diagonal skew [[0, 1], [-1, 0]]; e an MLP to R^{2k}."""

    def __init__(self, d: int, width: int, depth: int = 1, k: int = 4):
        super().__init__()
        self.e = MLP(d, width, depth, d_out=2 * k)
        self.k = k

    def forward(self, xi, xj):
        a, b = self.e(xi), self.e(xj)
        a1, a2 = a[..., 0::2], a[..., 1::2]
        b1, b2 = b[..., 0::2], b[..., 1::2]
        return (a1 * b2 - a2 * b1).sum(-1) / math.sqrt(self.k)


def build(kind: str, d: int, width: int, depth: int = 1, alpha: float = 1.0, seed: int = 0, **kw) -> nn.Module:
    torch.manual_seed(seed)
    m = {"scalar": ScalarBT, "antisym": Antisym, "gpm": GPM}[kind](d, width, depth, **kw)
    return Centered(m, alpha) if kind != "gpm" else _GPMCentered(m, alpha)


class _GPMCentered(nn.Module):
    """GPM is bilinear in the embedding, so centring the output would kill its
    first-order term at init; we centre by subtracting the initial *flow* but
    keep the live model's gradient (standard 'centred' trick)."""

    def __init__(self, m: nn.Module, alpha: float = 1.0):
        super().__init__()
        self.m, self.m0, self.alpha = m, copy.deepcopy(m), alpha
        for p in self.m0.parameters():
            p.requires_grad_(False)

    def forward(self, xi, xj):
        return self.alpha * (self.m(xi, xj) - self.m0(xi, xj))
