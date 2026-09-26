import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from src.landscape.nk import NKLandscape


def test_fitness_range():
    lm = NKLandscape(L=8, K=2, seed=0)
    genomes = torch.randint(0, 2, (100, 8))
    fit = lm.fitness(genomes)
    assert fit.shape == (100,)
    assert (fit >= 0).all() and (fit < 1).all()


def test_k_zero_is_additive_and_reproducible():
    lm1 = NKLandscape(L=6, K=0, seed=1)
    lm2 = NKLandscape(L=6, K=0, seed=1)
    genomes = torch.randint(0, 2, (10, 6))
    assert torch.allclose(lm1.fitness(genomes), lm2.fitness(genomes))


def test_different_seeds_give_different_landscapes():
    lm1 = NKLandscape(L=6, K=2, seed=1)
    lm2 = NKLandscape(L=6, K=2, seed=2)
    assert not torch.allclose(lm1.table, lm2.table)


def test_brute_force_matches_exhaustive_check():
    lm = NKLandscape(L=8, K=1, seed=0)
    best_genome, best_fit = lm.brute_force_optimum()
    ints = torch.arange(1 << 8)
    bits = ((ints.unsqueeze(1) >> torch.arange(7, -1, -1)) & 1)
    all_fit = lm.fitness(bits)
    assert abs(best_fit - all_fit.max().item()) < 1e-6
    assert torch.equal(best_genome, bits[all_fit.argmax()])


def test_batched_shape_supports_replicate_axis():
    lm = NKLandscape(L=10, K=3, seed=0)
    genomes = torch.randint(0, 2, (5, 20, 10))  # (R, N, L)
    fit = lm.fitness(genomes)
    assert fit.shape == (5, 20)
