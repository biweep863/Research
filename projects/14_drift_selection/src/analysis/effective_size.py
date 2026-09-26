"""Population-genetics quantities borrowed to analyse the GA runs:

- `temporal_ne`: the standard temporal (F-statistic) estimator of effective
  population size from allele-frequency trajectories (Krimbas & Tsakas 1971;
  Waples 1989): Ne = p(1-p) / (2 * Var(delta_p)), averaged over generations
  and tracked loci.
- `dfe_selection_coefficient`: the mean relative fitness advantage of
  single-bit-flip mutants over their parent, sampled from random genomes on
  a landscape -- a proxy for the population-genetics selection coefficient s.
- `kimura_fixation_prob` / `critical_n_kimura`: Kimura's (1962) diffusion
  fixation probability for a mutation of advantage s in a haploid
  Wright-Fisher population of size N, and the population size at which
  N*s = 1 (the conventional drift/selection crossover).
- `fit_threshold`: logistic fit of P(success) vs log(N) to extract the
  population size at which success probability crosses 0.5, with a
  bootstrap CI.
"""
from __future__ import annotations

import numpy as np
import torch


def temporal_ne(freq_history: torch.Tensor) -> torch.Tensor:
    """freq_history: (G+1, R). Returns (R,) Ne estimate per replicate."""
    p = freq_history[:-1]
    dp = freq_history[1:] - freq_history[:-1]
    var_dp = dp.var(dim=0, unbiased=True)
    p_bar = p.mean(dim=0)
    denom = 2 * var_dp
    ne = p_bar * (1 - p_bar) / denom
    return torch.where(denom > 1e-12, ne, torch.full_like(ne, float("nan")))


def dfe_selection_coefficient(landscape, n_samples: int, seed: int, device: str) -> float:
    """Mean relative fitness gain of beneficial single-bit-flip mutants,
    sampled from random genomes -- a landscape-level proxy for s."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    genomes = (torch.rand(n_samples, landscape.L, generator=gen) < 0.5).to(device).long()
    base_fit = landscape.fitness(genomes)
    loci = torch.randint(0, landscape.L, (n_samples,), generator=gen).to(device)
    mutants = genomes.clone()
    mutants[torch.arange(n_samples, device=device), loci] ^= 1
    mut_fit = landscape.fitness(mutants)
    delta = (mut_fit - base_fit) / base_fit.clamp_min(1e-9)
    beneficial = delta[delta > 0]
    if beneficial.numel() == 0:
        return 0.0
    return float(beneficial.mean().item())


def kimura_fixation_prob(s: float, N: int) -> float:
    """Haploid Wright-Fisher diffusion approximation (Kimura 1962)."""
    if abs(s) < 1e-12:
        return 1.0 / N
    num = 1.0 - np.exp(-2.0 * s)
    den = 1.0 - np.exp(-2.0 * N * s)
    if abs(den) < 1e-300:
        return 2.0 * s
    return num / den


def critical_n_kimura(s: float) -> float:
    """Population size at which N*s = 1, the conventional drift/selection
    crossover: below it, u(s,N) is close to the neutral value 1/N; above
    it, selection dominates and u(s,N) approaches 2s."""
    if s <= 0:
        return float("inf")
    return 1.0 / s


def _fit_logistic_crossing(x: np.ndarray, p_success: np.ndarray, n_reps: int):
    """One point per N with (successes, fails) as sample weights -- avoids
    materialising per-trial rows for an aggregate binomial fit."""
    from sklearn.linear_model import LogisticRegression

    successes = p_success * n_reps
    fails = n_reps - successes
    X = np.concatenate([x, x])
    y = np.concatenate([np.ones(len(x)), np.zeros(len(x))])
    w = np.concatenate([successes, fails])
    if successes.sum() <= 0 or fails.sum() <= 0:
        return float("nan")
    clf = LogisticRegression().fit(X, y, sample_weight=w)
    a, b = clf.intercept_[0], clf.coef_[0, 0]
    return -a / b if b != 0 else float("nan")


def fit_threshold(N_grid: np.ndarray, p_success: np.ndarray, n_reps: int = 200,
                   n_boot: int = 500, seed: int = 0):
    """Logistic regression of p_success on log2(N); returns (N_c, ci_lo, ci_hi).
    N_c is the population size where the fitted curve crosses 0.5. n_reps is
    the number of Bernoulli trials p_success[i] was estimated from, used as
    the binomial weight and for a Gaussian bootstrap over trial noise."""
    x = np.log2(N_grid).reshape(-1, 1)
    log2_nc = _fit_logistic_crossing(x, p_success, n_reps)
    if np.isnan(log2_nc):
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        p_boot = rng.binomial(n_reps, np.clip(p_success, 0, 1)) / n_reps
        lb = _fit_logistic_crossing(x, p_boot, n_reps)
        if not np.isnan(lb):
            boots.append(lb)
    if not boots:
        return float(2 ** log2_nc), float("nan"), float("nan")
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(2 ** log2_nc), float(2 ** lo), float(2 ** hi)
