"""Discrete local-PCA curvature proxy.

For each point, take its k nearest neighbours, center them, and run PCA. The
ratio (sum of trailing eigenvalues) / (sum of leading eigenvalues) is a coarse
proxy for sectional curvature: flat manifolds have most variance in a few
directions (low ratio); high-curvature regions spread variance across many
directions (high ratio).
"""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors


def local_pca_curvature(x: np.ndarray, k: int = 20, leading: int = 3, max_points: int = 512,
                        seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < k + 2:
        return float("nan")
    idx = rng.choice(n, size=min(max_points, n), replace=False)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, knn = nbrs.kneighbors(x[idx])
    ratios = []
    for row in knn:
        nbh = x[row]
        nbh = nbh - nbh.mean(axis=0, keepdims=True)
        s = np.linalg.svd(nbh, compute_uv=False)
        s = s ** 2
        if s.sum() <= 0:
            continue
        ratios.append(s[leading:].sum() / (s[:leading].sum() + 1e-12))
    return float(np.mean(ratios)) if ratios else float("nan")
