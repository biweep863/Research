"""Spatially selective filtering: delay-and-sum and MVDR toward the tracked speaker.

Output is mono and meant to feed DeepFilterNet3 exactly like ch0 does today
in home2 ``noise_cancellation.py`` - DF removes stationary/diffuse noise, the
beamformer removes what DF cannot: a competing talker from another direction.
"""

from __future__ import annotations

import numpy as np

from .geometry import SPEED_OF_SOUND, steering_vectors


def delay_and_sum_weights(
    mics: np.ndarray, az_rad: float, freqs: np.ndarray, c: float = SPEED_OF_SOUND
) -> np.ndarray:
    """(F, M) weights with w^H a = 1."""
    a = steering_vectors(mics, np.array([az_rad]), freqs, c)[0]
    return a / mics.shape[0]


def mvdr_weights(
    Rn: np.ndarray,
    mics: np.ndarray,
    az_rad: float,
    freqs: np.ndarray,
    c: float = SPEED_OF_SOUND,
    loading: float = 1e-2,
) -> np.ndarray:
    """(F, M) MVDR weights w = Rn^-1 a / (a^H Rn^-1 a), diagonal loading relative
    to the per-bin trace (robust to steering/calibration error)."""
    a = steering_vectors(mics, np.array([az_rad]), freqs, c)[0]  # (F, M)
    M = mics.shape[0]
    tr = np.real(np.einsum("fii->f", Rn)) / M
    Rl = Rn + (loading * tr + 1e-10)[:, None, None] * np.eye(M)[None]
    Ria = np.linalg.solve(Rl, a[..., None])[..., 0]  # (F, M)
    denom = np.einsum("fm,fm->f", a.conj(), Ria)
    return Ria / (denom[:, None] + 1e-12)


def apply_weights(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(F, M) weights, (M, F, N) STFT -> (F, N) beam output y = w^H x."""
    return np.einsum("fm,mfn->fn", W.conj(), X)
