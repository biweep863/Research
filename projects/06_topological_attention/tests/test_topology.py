import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.analysis.topology import attention_distance, persistence, betti_curve


def _toy_attention(n_heads=2, n=10, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(n), size=(n_heads, n))
    return A  # shape (n_heads, n, n)


def test_attention_distance_symmetric_zero_diag():
    A = _toy_attention()
    D = attention_distance(A)
    assert D.shape == (A.shape[1], A.shape[1])
    assert np.allclose(D, D.T)
    assert np.all(np.diag(D) == 0)


def test_persistence_runs():
    A = _toy_attention(n=12)
    D = attention_distance(A)
    out = persistence(D, max_dim=1)
    assert "total_persistence_H0" in out and "total_persistence_H1" in out


def test_betti_curve_runs():
    A = _toy_attention(n=12)
    D = attention_distance(A)
    thr = np.linspace(0, 1, 10)
    out = betti_curve(D, thr, max_dim=1)
    assert "betti_0" in out and len(out["betti_0"]) == 10
