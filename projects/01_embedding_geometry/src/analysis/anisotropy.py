"""Average pairwise cosine similarity (anisotropy) on a random subsample."""
from __future__ import annotations
import numpy as np


def anisotropy(x: np.ndarray, n_pairs: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 4:
        return float("nan")
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    xn = x / norms
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    sims = np.einsum("ij,ij->i", xn[i[keep]], xn[j[keep]])
    return float(sims.mean())
