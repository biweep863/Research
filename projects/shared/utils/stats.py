"""Bootstrap CIs and basic significance tests."""
from __future__ import annotations
import numpy as np


def bootstrap_ci(x, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = rng.choice(x, size=n, replace=True).mean()
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(x.mean()), float(lo), float(hi)


def welch_t(a, b):
    from scipy import stats
    a, b = np.asarray(a), np.asarray(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)
