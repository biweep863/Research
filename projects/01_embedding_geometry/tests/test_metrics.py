import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.analysis.intrinsic_dimension import twonn, effective_rank
from src.analysis.anisotropy import anisotropy
from src.analysis.curvature import local_pca_curvature


def test_twonn_on_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 5))
    d = twonn(x)
    assert 3 < d < 8, f"expected ~5, got {d}"


def test_anisotropy_random():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 64))
    a = anisotropy(x, n_pairs=500)
    assert -0.1 < a < 0.1


def test_curvature_finite():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 16))
    c = local_pca_curvature(x, k=10, leading=3, max_points=100)
    assert np.isfinite(c) and c >= 0


def test_eff_rank():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 10))
    er = effective_rank(x)
    assert 1 < er <= 10
