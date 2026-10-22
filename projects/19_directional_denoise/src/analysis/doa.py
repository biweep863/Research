# Vendored from projects/15_spatial_audio_attention/src/analysis/doa.py (kept identical).
"""Direction of arrival: SRP-PHAT on an azimuth grid, GCC-PHAT baseline.

The ReSpeaker aperture is 6.4 cm, so the largest inter-mic delay is ~0.19 ms
(~3 samples at 16 kHz). Integer-lag GCC peaks quantise the angle badly, which
is why the main estimator steers in the frequency domain (SRP-PHAT, no lag
quantisation) and the GCC baseline upsamples its correlation.
"""

from __future__ import annotations

import numpy as np

from .geometry import SPEED_OF_SOUND, far_field_delays, mic_pairs


def cross_spectra(X: np.ndarray) -> np.ndarray:
    """(M, F) or (M, F, N) STFT -> (F, M, M) cross-spectral matrix E[X X^H]."""
    if X.ndim == 2:
        X = X[:, :, None]
    return np.einsum("ifn,jfn->fij", X, X.conj()) / X.shape[-1]


def band_mask(freqs: np.ndarray, band) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def harmonic_weights(
    freqs: np.ndarray, f0: float, rel_width: float = 0.04, floor: float = 0.1
) -> np.ndarray:
    """Comb of Gaussians at k*F0: emphasises bins that carry the voice harmonics.

    This is the frequency-domain use of the VAD pitch: noise between harmonics
    is down-weighted in the SRP sum. Returns ones if no pitch.
    """
    if not f0 or f0 <= 0:
        return np.ones_like(freqs)
    k = np.maximum(np.round(freqs / f0), 1.0)
    dist = freqs - k * f0
    sigma = rel_width * k * f0 + 1e-9
    return floor + (1 - floor) * np.exp(-0.5 * (dist / sigma) ** 2)


def snr_weights(Rx: np.ndarray, Rn: np.ndarray | None, floor: float = 0.0) -> np.ndarray | None:
    """Per-bin Wiener-like speech-presence weight 1 - tr(Rn)/tr(Rx), in [floor, 1].

    PHAT whitens every bin to unit magnitude, so bins that contain only noise
    (e.g. between voice harmonics) vote as loudly as speech bins. The silence
    noise model tells us which bins actually carry the talker.
    """
    if Rn is None:
        return None
    px = np.real(np.einsum("fii->f", Rx))
    pn = np.real(np.einsum("fii->f", Rn))
    return np.clip(1.0 - pn / np.maximum(px, 1e-20), floor, 1.0)


def srp_phat(
    R: np.ndarray,
    freqs: np.ndarray,
    mics: np.ndarray,
    grid_rad: np.ndarray,
    band=(200.0, 3800.0),
    c: float = SPEED_OF_SOUND,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Steered response power with PHAT weighting over an azimuth grid.

    R: (F, M, M) cross-spectra. Returns P (n_az,) in [-1, 1]; 1 = all pairs
    perfectly phase-aligned at that azimuth.
    """
    sel = band_mask(freqs, band)
    f = freqs[sel]
    w = np.ones_like(f) if weights is None else weights[sel]
    tau = far_field_delays(mics, grid_rad, c)  # (n_az, M)
    P = np.zeros(len(grid_rad))
    pairs = mic_pairs(mics.shape[0])
    for i, j in pairs:
        g = R[sel, i, j]
        g = w * g / (np.abs(g) + 1e-12)
        dtau = tau[:, i] - tau[:, j]  # (n_az,)
        P += np.real(np.exp(2j * np.pi * np.outer(dtau, f)) @ g)
    return P / (len(pairs) * w.sum() + 1e-12)


def peak_azimuth(P: np.ndarray, grid_rad: np.ndarray) -> tuple[float, float]:
    """(azimuth_rad, confidence) from an SRP map on a uniform circular grid.

    Peak refined by parabolic interpolation (with wrap-around). Confidence is
    how far the peak stands above the map mean, normalised to [0, 1].
    """
    n = len(P)
    k = int(np.argmax(P))
    a, b, cc = P[(k - 1) % n], P[k], P[(k + 1) % n]
    denom = a - 2 * b + cc
    shift = 0.5 * (a - cc) / denom if denom < 0 else 0.0
    step = grid_rad[1] - grid_rad[0]
    az = np.mod(grid_rad[k] + shift * step, 2 * np.pi)
    mean = float(P.mean())
    conf = float(np.clip((b - mean) / (1.0 - mean + 1e-12), 0.0, 1.0))
    return float(az), conf


def gcc_phat_tdoa(
    R_ij: np.ndarray, nfft: int, fs: int, max_tau: float, interp: int = 16
) -> float:
    """TDOA tau_i - tau_j (s) from one pair's cross-spectrum (F,), upsampled GCC."""
    g = R_ij / (np.abs(R_ij) + 1e-12)
    n = nfft * interp
    cc = np.fft.irfft(g, n=n)
    max_shift = int(np.ceil(max_tau * fs * interp)) + 1
    cc = np.concatenate([cc[-max_shift:], cc[: max_shift + 1]])
    shift = int(np.argmax(cc)) - max_shift
    return shift / float(fs * interp)


def doa_from_tdoas(
    tdoas: dict[tuple[int, int], float], mics: np.ndarray, c: float = SPEED_OF_SOUND
) -> float:
    """Least-squares azimuth from pairwise TDOAs: (p_i - p_j) . u = -c tau_ij."""
    A = np.array([mics[i] - mics[j] for (i, j) in tdoas])
    b = -c * np.array(list(tdoas.values()))
    u, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.mod(np.arctan2(u[1], u[0]), 2 * np.pi))


def gcc_phat_doa(
    R: np.ndarray,
    freqs: np.ndarray,
    mics: np.ndarray,
    nfft: int,
    fs: int,
    band=(200.0, 3800.0),
    c: float = SPEED_OF_SOUND,
) -> float:
    """Baseline: per-pair upsampled GCC-PHAT, then LS fit of the azimuth."""
    sel = band_mask(freqs, band).astype(float)
    tdoas = {}
    for i, j in mic_pairs(mics.shape[0]):
        max_tau = np.linalg.norm(mics[i] - mics[j]) / c
        tdoas[(i, j)] = gcc_phat_tdoa(R[:, i, j] * sel, nfft, fs, max_tau)
    return doa_from_tdoas(tdoas, mics, c)
