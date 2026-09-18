"""A deep-sea community and a sampling process.

Abundances follow Fisher's log-series (Fisher, Corbet & Williams 1943), the
classic model for species-abundance distributions with many rare species.
A sensor deployed for ``hours`` detects individuals as a Poisson process whose
rate for species i is  rate * n_i * range_scale, so common species are found
early and rare ones only with long effort.  ``sample`` returns the detection
counts per species (the "abundance vector" the estimators consume).
"""
from __future__ import annotations

import numpy as np


def logseries_abundances(n_species: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Relative abundances of ``n_species`` species drawn from a log-series
    shape: p_k ~ x^k / k with x close to 1; sorted, normalised."""
    x = 1.0 - 1.0 / max(alpha, 1.01)
    k = np.arange(1, n_species + 1)
    p = x ** k / k
    p = p / p.sum()
    return rng.permutation(p)


def sample(abund: np.ndarray, hours: float, rate: float, range_scale: float, rng: np.random.Generator) -> np.ndarray:
    lam = rate * range_scale * hours * abund * len(abund)
    return rng.poisson(lam)
