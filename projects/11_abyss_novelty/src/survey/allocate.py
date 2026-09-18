"""Where to spend the sensor hours.

Sites belong to habitats with different (unknown) richness; a satellite-derived
prior labels each site's habitat with accuracy ``prior_quality``.  Expected new
species from ``h`` hours at a site is the expected species-accumulation
E[S(h)] = sum_i (1 - exp(-lambda_i h)) for that habitat's community.  A greedy
allocation spends the budget in increments where the marginal expected gain
(under the prior) is largest; compared with a uniform split.
"""
from __future__ import annotations

import numpy as np


def expected_species(abund: np.ndarray, hours: float, rate: float) -> float:
    lam = rate * hours * abund * len(abund)
    return float(np.sum(1.0 - np.exp(-lam)))


def greedy_allocate(site_priors: np.ndarray, communities: dict, budget_h: float, rate: float,
                    step_h: float = 5.0) -> np.ndarray:
    """``site_priors[s]`` = dict-like mapping habitat -> probability.  Returns hours per site."""
    n = len(site_priors)
    hours = np.zeros(n)
    steps = int(budget_h / step_h)
    for _ in range(steps):
        gains = []
        for s in range(n):
            g = 0.0
            for hab, p in site_priors[s].items():
                g += p * (expected_species(communities[hab], hours[s] + step_h, rate)
                          - expected_species(communities[hab], hours[s], rate))
            gains.append(g)
        hours[int(np.argmax(gains))] += step_h
    return hours


def realised_species(hours: np.ndarray, true_habitat: list, communities: dict, rate: float,
                     rng: np.random.Generator) -> int:
    """Simulate the survey: union of species detected over all sites (species
    identities are shared within a habitat, distinct across habitats)."""
    seen = set()
    for s, h in enumerate(hours):
        if h <= 0:
            continue
        ab = communities[true_habitat[s]]
        lam = rate * h * ab * len(ab)
        det = rng.poisson(lam) > 0
        seen.update((true_habitat[s], i) for i in np.flatnonzero(det))
    return len(seen)
