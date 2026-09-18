"""How many species have we *not* seen?

* Chao1 (Chao 1984): S_hat = S_obs + f1^2 / (2 f2)   (f1 singletons, f2 doubletons),
  bias-corrected form f1 (f1 - 1) / (2 (f2 + 1)) when f2 = 0; variance from Chao 1987.
* Good-Turing coverage (Good 1953): C = 1 - f1 / n is the probability that the
  next detection is a species already seen; 1 - C is the discovery probability.
"""
from __future__ import annotations

import numpy as np


def chao1(counts: np.ndarray) -> tuple[float, float]:
    c = np.asarray(counts)
    s_obs = int((c > 0).sum())
    f1 = int((c == 1).sum())
    f2 = int((c == 2).sum())
    if f2 > 0:
        s_hat = s_obs + f1 * f1 / (2.0 * f2)
        var = f2 * (0.5 * (f1 / f2) ** 2 + (f1 / f2) ** 3 + 0.25 * (f1 / f2) ** 4)
    else:
        s_hat = s_obs + f1 * (f1 - 1) / 2.0
        var = f1 * (f1 - 1) / 2.0 + f1 * (2 * f1 - 1) ** 2 / 4.0 - f1 ** 4 / (4.0 * max(s_hat, 1))
    return float(s_hat), float(np.sqrt(max(var, 0.0)))


def good_turing_discovery(counts: np.ndarray) -> float:
    c = np.asarray(counts)
    n = c.sum()
    f1 = (c == 1).sum()
    return float(f1 / n) if n > 0 else 1.0
