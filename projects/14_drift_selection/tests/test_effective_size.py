import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from src.landscape.nk import NKLandscape
from src.analysis.effective_size import (critical_n_kimura, dfe_selection_coefficient,
                                          fit_threshold, kimura_fixation_prob, temporal_ne)


def test_kimura_neutral_limit():
    assert abs(kimura_fixation_prob(0.0, 50) - 1 / 50) < 1e-9


def test_kimura_strong_selection_limit():
    s = 0.0001
    p = kimura_fixation_prob(s, 100_000)  # Ns=10: strong selection, s itself small
    assert abs(p - 2 * s) < 1e-6


def test_critical_n_kimura_matches_definition():
    s = 0.05
    nc = critical_n_kimura(s)
    assert abs(nc - 1 / s) < 1e-9


def test_dfe_selection_coefficient_nonnegative():
    lm = NKLandscape(L=10, K=2, seed=0)
    s = dfe_selection_coefficient(lm, n_samples=500, seed=0, device="cpu")
    assert s >= 0


def test_temporal_ne_neutral_drift_matches_order_of_magnitude():
    torch.manual_seed(0)
    N = 100
    G = 200
    p = torch.zeros(G + 1)
    p[0] = 0.5
    for g in range(G):
        k = torch.distributions.Binomial(N, p[g].clamp(1e-6, 1 - 1e-6)).sample()
        p[g + 1] = k / N
    ne = temporal_ne(p.unsqueeze(1))
    assert 20 < ne.item() < 500


def test_fit_threshold_recovers_known_crossing():
    N_grid = np.array([4, 8, 16, 32, 64, 128, 256])
    true_nc = 30.0
    p_success = 1 / (1 + (true_nc / N_grid))
    nc, lo, hi = fit_threshold(N_grid, p_success, n_reps=500, n_boot=100)
    assert 15 < nc < 60
