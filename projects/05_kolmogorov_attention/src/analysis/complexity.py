"""Kolmogorov-complexity proxies and attention-density metrics."""
from __future__ import annotations
import gzip
import zlib
import numpy as np


def gzip_complexity(text: str) -> int:
    return len(gzip.compress(text.encode("utf-8")))


def zlib_complexity(text: str) -> int:
    return len(zlib.compress(text.encode("utf-8")))


def compression_ratio(text: str) -> float:
    raw = len(text.encode("utf-8"))
    return gzip_complexity(text) / max(1, raw)


def normalised_compressed_distance(a: str, b: str) -> float:
    """NCD(a,b) = (C(ab) - min(C(a),C(b))) / max(C(a),C(b))."""
    ca, cb, cab = gzip_complexity(a), gzip_complexity(b), gzip_complexity(a + b)
    return (cab - min(ca, cb)) / max(ca, cb)


def attention_entropy(attn_row: np.ndarray) -> float:
    """Shannon entropy of an attention distribution (one row)."""
    p = attn_row[attn_row > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def attention_gini(attn_row: np.ndarray) -> float:
    """Gini coefficient of attention concentration. 0 = uniform; 1 = on one key."""
    v = np.sort(attn_row.flatten())
    n = v.size
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def attention_sparsity(attn_row: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of attention mass concentrated in fewer than `threshold` of keys."""
    v = np.sort(attn_row.flatten())[::-1]
    cum = np.cumsum(v)
    n = v.size
    target = 0.9 * v.sum()
    k = int(np.searchsorted(cum, target) + 1)
    return k / max(1, n)


def density_summary(attn_layer: np.ndarray) -> dict:
    """Aggregate density metrics on a (heads, queries, keys) tensor."""
    heads, Q, K = attn_layer.shape
    entropies, ginis, sparsities = [], [], []
    for h in range(heads):
        for q in range(Q):
            row = attn_layer[h, q]
            entropies.append(attention_entropy(row))
            ginis.append(attention_gini(row))
            sparsities.append(attention_sparsity(row))
    return {
        "mean_entropy":   float(np.mean(entropies)),
        "mean_gini":      float(np.mean(ginis)),
        "mean_sparsity":  float(np.mean(sparsities)),
    }
