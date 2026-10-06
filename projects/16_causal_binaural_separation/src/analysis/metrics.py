"""Signal-quality metrics. Pure functions on numpy arrays, no model I/O."""

from __future__ import annotations

import numpy as np

try:  # PESQ needs a C compiler to install on Windows; optional by design (doc/05).
    from pesq import pesq as _pesq

    HAS_PESQ = True
except ImportError:  # pragma: no cover
    HAS_PESQ = False


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al. 2019). Same as project 15."""
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((target @ target + 1e-12) / (noise @ noise + 1e-12)))


def si_sdr_binaural(est: np.ndarray, ref: np.ndarray) -> float:
    """Mean SI-SDR over the two ears. est/ref: (2, T)."""
    return float(np.mean([si_sdr(est[e], ref[e]) for e in range(est.shape[0])]))


def si_sdr_improvement(est: np.ndarray, mix: np.ndarray, ref: np.ndarray) -> float:
    return si_sdr_binaural(est, ref) - si_sdr_binaural(mix, ref)


def estoi(est: np.ndarray, ref: np.ndarray, fs: int) -> float:
    """Extended STOI (Jensen & Taal 2016), averaged over ears."""
    from pystoi import stoi

    return float(np.mean([stoi(ref[e], est[e], fs, extended=True) for e in range(est.shape[0])]))


def pesq_wb(est: np.ndarray, ref: np.ndarray, fs: int) -> float:
    """Wide-band PESQ averaged over ears, or NaN when the package is missing."""
    if not HAS_PESQ or fs != 16000:
        return float("nan")
    vals = []
    for e in range(est.shape[0]):
        try:
            vals.append(_pesq(fs, ref[e], est[e], "wb"))
        except Exception:  # silent / too-short segments raise inside the C code
            vals.append(np.nan)
    return float(np.nanmean(vals))
