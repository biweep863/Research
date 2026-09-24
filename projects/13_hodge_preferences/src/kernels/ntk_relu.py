"""Closed-form infinite-width NTK of a fully connected ReLU network.

NTK parameterisation with weight variance sigma_w^2 = 2 and bias variance
sigma_b^2 (Jacot et al. 2018; Arora et al. 2019).  For inputs x, x' the
recursion over L hidden layers is

  Sigma^0 = sigma_w^2 x.x'/d + sigma_b^2   (the first layer also carries sigma_w)
  Sigma^l = sigma_w^2 * k1(Sigma^{l-1}) + sigma_b^2          (arc-cosine, degree 1)
  Sdot^l  = sigma_w^2 * k0(Sigma^{l-1})                       (arc-cosine, degree 0)
  Theta^l = Sigma^l + Theta^{l-1} * Sdot^l,   Theta^0 = Sigma^0

and the output layer adds one more linear readout (Theta = Theta^L), where
k0, k1 are the ReLU arc-cosine kernels of Cho & Saul (2009).
"""
from __future__ import annotations

import numpy as np


def _k0_k1(S: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    norm = np.sqrt(np.outer(dx, dy))
    c = np.clip(S / np.maximum(norm, 1e-300), -1.0, 1.0)
    th = np.arccos(c)
    k0 = (np.pi - th) / (2 * np.pi)
    k1 = norm * (np.sin(th) + (np.pi - th) * c) / (2 * np.pi)
    return k0, k1


def relu_ntk(X: np.ndarray, Y: np.ndarray | None = None, depth: int = 1,
             sigma_w2: float = 2.0, sigma_b2: float = 0.1) -> np.ndarray:
    """NTK matrix between the rows of X and Y for `depth` hidden layers."""
    Y = X if Y is None else Y
    d = X.shape[1]
    S = sigma_w2 * X @ Y.T / d + sigma_b2
    dx = sigma_w2 * np.sum(X * X, 1) / d + sigma_b2
    dy = sigma_w2 * np.sum(Y * Y, 1) / d + sigma_b2
    T = S.copy()
    for _ in range(depth):
        k0, k1 = _k0_k1(S, dx, dy)
        S_new = sigma_w2 * k1 + sigma_b2
        Sd = sigma_w2 * k0
        T = S_new + T * Sd
        S = S_new
        dx = sigma_w2 * dx / 2 + sigma_b2          # k1(x,x) = |x|^2 / 2
        dy = sigma_w2 * dy / 2 + sigma_b2
    return T
