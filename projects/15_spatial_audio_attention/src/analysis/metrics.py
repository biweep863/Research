from __future__ import annotations

import numpy as np

from .tracker import wrap_deg


def angular_error_deg(est_deg, true_deg):
    return np.abs(wrap_deg(np.asarray(est_deg) - np.asarray(true_deg)))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019)."""
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((target @ target + 1e-12) / (noise @ noise + 1e-12)))


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)
