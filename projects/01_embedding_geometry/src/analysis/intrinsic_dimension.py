"""Intrinsic dimension estimators.

TwoNN (Facco et al. 2017): if r1, r2 are the distances to the 1st and 2nd nearest
neighbours, then mu = r2/r1 follows a Pareto(d) distribution. The MLE for d is
    d_hat = N / sum_i log(mu_i)
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn(x: np.ndarray, fraction: float = 0.9, seed: int = 0) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 10:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(x)
    d, _ = nbrs.kneighbors(x)
    r1 = d[:, 1]
    r2 = d[:, 2]
    mask = (r1 > 0) & (r2 > 0)
    r1, r2 = r1[mask], r2[mask]
    mu = r2 / r1
    mu = np.sort(mu)
    # use the first `fraction` to avoid outliers in the upper tail
    k = max(10, int(fraction * len(mu)))
    mu = mu[:k]
    logs = np.log(mu)
    s = logs.sum()
    if s <= 0:
        return float("nan")
    return float(k / s)


def effective_rank(x: np.ndarray) -> float:
    """Roy & Vetterli effective rank: exp(H(p)) where p_i = sigma_i / sum sigma."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))
