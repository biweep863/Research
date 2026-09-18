"""Group the clips flagged as novel into candidate new species.

Agglomerative clustering (average linkage, cosine) on the embeddings of the
flagged clips; the number of clusters is chosen by the silhouette score over
``k_range``.  Quality vs the true unknown-species labels: adjusted Rand index
(Hubert & Arabie 1985) and |k_hat - k_true|.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score


def cluster_novel(z: np.ndarray, k_range=(2, 10), link: str = "average") -> tuple[np.ndarray, int, float]:
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    best = (-1.0, None, 0)
    for k in range(k_range[0], min(k_range[1], len(z) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage=link).fit_predict(z)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(z, lab, metric="cosine")
        if s > best[0]:
            best = (s, lab, k)
    if best[1] is None:
        return np.zeros(len(z), dtype=int), 1, 0.0
    return best[1], best[2], best[0]


def ari(true: np.ndarray, pred: np.ndarray) -> float:
    return float(adjusted_rand_score(true, pred))
