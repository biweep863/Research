"""Persistent homology of weighted attention graphs.

We treat each layer's attention matrix as a weighted undirected graph
(symmetrising by max). The distance between two tokens is 1 - max(A_ij, A_ji).
We compute persistent homology over the Vietoris-Rips filtration of this
distance, summarise it with Betti numbers across a fixed grid of thresholds,
and report total persistence in H0 and H1.
"""
from __future__ import annotations
import numpy as np
from ripser import ripser


def attention_distance(A: np.ndarray) -> np.ndarray:
    """A: (n_heads, Q, K). Returns (Q, Q) symmetric distance matrix obtained
    by averaging over heads and symmetrising."""
    H, Q, K = A.shape
    assert Q == K, "expected self-attention"
    G = A.mean(axis=0)
    G = np.maximum(G, G.T)              # symmetrise by max
    np.fill_diagonal(G, 1.0)
    D = 1.0 - G
    np.fill_diagonal(D, 0.0)
    return D


def persistence(D: np.ndarray, max_dim: int = 1) -> dict:
    """Compute persistent homology of a distance matrix and return total
    persistence per dimension."""
    res = ripser(D, distance_matrix=True, maxdim=max_dim)
    dgms = res["dgms"]
    out = {}
    for k, d in enumerate(dgms):
        if d.size == 0:
            out[f"total_persistence_H{k}"] = 0.0
            out[f"n_features_H{k}"] = 0
            continue
        # ignore infinite-persistence H0 bar (always one of them)
        finite = d[np.isfinite(d[:, 1])]
        if finite.size == 0:
            out[f"total_persistence_H{k}"] = 0.0
            out[f"n_features_H{k}"] = 0
            continue
        bars = finite[:, 1] - finite[:, 0]
        out[f"total_persistence_H{k}"] = float(bars.sum())
        out[f"n_features_H{k}"] = int(finite.shape[0])
    return out


def betti_curve(D: np.ndarray, thresholds: np.ndarray, max_dim: int = 1) -> dict:
    """Betti numbers at each filtration threshold. Cheap approximation
    suitable for rapid visualisation."""
    res = ripser(D, distance_matrix=True, maxdim=max_dim)
    dgms = res["dgms"]
    out = {}
    for k, dgm in enumerate(dgms):
        if dgm.size == 0:
            out[f"betti_{k}"] = np.zeros_like(thresholds)
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        deaths = np.where(np.isinf(deaths), thresholds[-1] + 1, deaths)
        b = np.zeros_like(thresholds, dtype=int)
        for i, t in enumerate(thresholds):
            b[i] = int(np.sum((births <= t) & (deaths > t)))
        out[f"betti_{k}"] = b.tolist()
    return out
