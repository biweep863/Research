import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.landscape.nk import NKLandscape
from src.ga.simple_ga import run_ga


def test_run_ga_shapes():
    lm = NKLandscape(L=8, K=1, seed=0)
    _, opt = lm.brute_force_optimum()
    res = run_ga(lm, N=20, R=4, generations=15, p_c=0.7, p_m=0.02,
                 optimum_fitness=opt, success_tol=1e-6, seed=0, device="cpu")
    assert res.best_ever.shape == (4,)
    assert res.gen_to_success.shape == (4,)
    assert res.freq_history.shape == (16, 4)


def test_large_population_easy_landscape_succeeds_often():
    lm = NKLandscape(L=8, K=0, seed=0)
    _, opt = lm.brute_force_optimum()
    res = run_ga(lm, N=200, R=20, generations=60, p_c=0.7, p_m=0.02,
                 optimum_fitness=opt, success_tol=1e-6, seed=1, device="cpu")
    assert (res.gen_to_success >= 0).float().mean() > 0.7


def test_tiny_population_succeeds_less_often_than_large():
    lm = NKLandscape(L=10, K=4, seed=2)
    _, opt = lm.brute_force_optimum()
    small = run_ga(lm, N=4, R=30, generations=40, p_c=0.7, p_m=0.02,
                   optimum_fitness=opt, success_tol=1e-6, seed=3, device="cpu")
    large = run_ga(lm, N=200, R=30, generations=40, p_c=0.7, p_m=0.02,
                   optimum_fitness=opt, success_tol=1e-6, seed=3, device="cpu")
    p_small = (small.gen_to_success >= 0).float().mean().item()
    p_large = (large.gen_to_success >= 0).float().mean().item()
    assert p_large >= p_small


def test_reproducible_with_same_seed():
    lm = NKLandscape(L=8, K=1, seed=0)
    _, opt = lm.brute_force_optimum()
    a = run_ga(lm, N=20, R=4, generations=10, p_c=0.7, p_m=0.02,
              optimum_fitness=opt, success_tol=1e-6, seed=7, device="cpu")
    b = run_ga(lm, N=20, R=4, generations=10, p_c=0.7, p_m=0.02,
              optimum_fitness=opt, success_tol=1e-6, seed=7, device="cpu")
    assert torch.equal(a.best_ever, b.best_ever)
