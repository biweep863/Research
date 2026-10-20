"""Multichannel front ends compared before DeepFilterNet3.

Every method sees only what a deployed system sees: the mixture and its first
``lead`` frames, where the talker is silent (project 15's silence model). The
``oracle`` uses the true components and is an upper bound only.

All of them are linear in the input, so they return weights and the speech and
noise components can be passed through the *same* weights separately - that is
what makes the output SNR exact (``apply``). Weights are (F, M) for fixed
filters or (F, M, N) for the adaptive GSC. Outputs are referenced to mic 0
(firmware channel 1): a distortionless filter returns mic 0's speech.
"""

from __future__ import annotations

import numpy as np

from .doa import peak_azimuth, srp_phat
from .geometry import steering_vectors

REF = 0


# ------------------------------------------------------------------ helpers
def covariance(X: np.ndarray, frames=None) -> np.ndarray:
    """(M, F, N) STFT -> (F, M, M) spatial covariance over the selected frames."""
    if frames is not None:
        X = X[..., frames]
    return np.einsum("ifn,jfn->fij", X, X.conj()) / max(X.shape[-1], 1)


def _load(R: np.ndarray, loading: float) -> np.ndarray:
    M = R.shape[-1]
    tr = np.real(np.einsum("fii->f", R)) / M
    return R + (loading * tr + 1e-12)[:, None, None] * np.eye(M)[None]


def gevd_rtf(Ry: np.ndarray, Rn: np.ndarray, loading: float = 1e-3) -> np.ndarray:
    """Relative transfer function by covariance whitening (Markovich-Golan 2009).

    The principal eigenvector of Rn^-1/2 Ry Rn^-1/2, de-whitened, is the
    direction in which the mixture exceeds the noise most: the talker's
    acoustic transfer function (reverberation included), up to a scale that the
    normalisation to the reference mic removes. Needs no geometry and no DOA.
    """
    Rn = _load(Rn, loading)
    w, V = np.linalg.eigh(Rn)
    Rn_h = np.einsum("fij,fj,fkj->fik", V, np.sqrt(np.maximum(w, 1e-20)), V.conj())
    Rn_ih = np.einsum("fij,fj,fkj->fik", V, 1 / np.sqrt(np.maximum(w, 1e-20)), V.conj())
    C = Rn_ih @ Ry @ Rn_ih
    _, U = np.linalg.eigh(C)
    h = np.einsum("fij,fj->fi", Rn_h, U[..., -1])
    return h / (h[:, REF : REF + 1] + 1e-12)


def mvdr(Rn: np.ndarray, h: np.ndarray, loading: float = 1e-3) -> np.ndarray:
    """w = Rn^-1 h / (h^H Rn^-1 h); with h[ref] = 1 the output is the ref-mic speech."""
    Ria = np.linalg.solve(_load(Rn, loading), h[..., None])[..., 0]
    return Ria / (np.einsum("fm,fm->f", h.conj(), Ria)[:, None] + 1e-12)


def apply(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(F, M) or (F, M, N) weights, (M, F, N) STFT -> (F, N) output w^H x."""
    if W.ndim == 2:
        return np.einsum("fm,mfn->fn", W.conj(), X)
    return np.einsum("fmn,mfn->fn", W.conj(), X)


# ------------------------------------------------------------------ methods
def ref_mic(M: int, F: int) -> np.ndarray:
    W = np.zeros((F, M), complex)
    W[:, REF] = 1
    return W


def mic_mean(M: int, F: int) -> np.ndarray:
    return np.full((F, M), 1.0 / M, complex)


def mute_loudest(Rn: np.ndarray) -> np.ndarray:
    """The user's idea, literally: switch off the mic that hears the most noise
    (silence model, whole band) and average the other three, no alignment."""
    pn = np.real(np.einsum("fii->i", Rn))
    keep = np.ones(Rn.shape[-1], bool)
    keep[np.argmax(pn)] = False
    W = np.zeros(Rn.shape[:2], complex)
    W[:, keep] = 1.0 / keep.sum()
    return W


def mute_soft(Rn: np.ndarray) -> np.ndarray:
    """Per-band version: weight each mic by its inverse noise power, real
    weights (no phase alignment) - 'listen less to the noisier mic'."""
    pn = np.real(np.einsum("fii->fi", Rn))
    w = 1 / np.maximum(pn, 1e-20)
    return (w / w.sum(1, keepdims=True)).astype(complex)


def doa_from_mixture(X: np.ndarray, Rn: np.ndarray, lead: int, freqs, mics, band=(300.0, 4000.0)):
    """Project 15's estimator: SRP-PHAT on (mixture - silence model) covariance."""
    grid = np.deg2rad(np.arange(0, 360, 1.0))
    R = covariance(X, slice(lead, None)) - Rn
    P = srp_phat(R, freqs, mics, grid, band=band)
    az, conf = peak_azimuth(P, grid)
    return az, conf


def farfield_h(mics, az, freqs) -> np.ndarray:
    a = steering_vectors(mics, np.array([az]), freqs)[0]
    return a / a[:, REF : REF + 1]


def delay_and_sum(h: np.ndarray) -> np.ndarray:
    return h / h.shape[1]  # w^H h = 1 for |h_m| = 1


def gsc(X: np.ndarray, h: np.ndarray, lead: int, mu: float = 0.05, ratio_db: float = 3.0) -> np.ndarray:
    """Griffiths-Jim GSC with an NLMS canceller, time-varying weights (F, M, N).

    Fixed beam w0 = h/|h|^2 (distortionless), blocking matrix B = the M-1
    directions orthogonal to h (speech-free noise references), and per-bin
    complex NLMS that subtracts what of the references is in the fixed beam.
    It adapts only when the fixed beam's power is within ``ratio_db`` of its
    noise-only level (the lead): a speech-presence gate, so the canceller does
    not learn to cancel the talker through RTF errors (signal leakage).
    """
    M, F, N = X.shape
    w0 = h / np.sum(np.abs(h) ** 2, axis=1, keepdims=True)
    # orthonormal basis of the complement of h, per bin
    hn = h / np.linalg.norm(h, axis=1, keepdims=True)
    P = np.eye(M)[None] - np.einsum("fi,fj->fij", hn, hn.conj())
    U, _, _ = np.linalg.svd(P)
    B = U[:, :, : M - 1]  # (F, M, M-1)
    yb = np.einsum("fm,mfn->fn", w0.conj(), X)
    u = np.einsum("fmk,mfn->fkn", B.conj(), X)  # (F, M-1, N)
    p_noise = np.mean(np.abs(yb[:, :lead]) ** 2, axis=1) + 1e-12
    g = np.zeros((F, M - 1), complex)
    W = np.empty((F, M, N), complex)
    thr = 10 ** (ratio_db / 10)
    pow_u = np.mean(np.abs(u[..., :lead]) ** 2, axis=(1, 2)) + 1e-12
    for n in range(N):
        W[:, :, n] = w0 - np.einsum("fmk,fk->fm", B, g)
        e = yb[:, n] - np.einsum("fk,fk->f", g.conj(), u[:, :, n])
        adapt = (n < lead) | (np.abs(yb[:, n]) ** 2 < thr * p_noise)
        norm = np.sum(np.abs(u[:, :, n]) ** 2, axis=1) + 0.1 * pow_u
        g += (adapt * mu / norm)[:, None] * u[:, :, n] * e.conj()[:, None]
    return W


def oracle_mvdr(S: np.ndarray, Nn: np.ndarray, active_frames) -> np.ndarray:
    Rn = covariance(Nn)
    h = gevd_rtf(covariance(S, active_frames) + Rn, Rn)
    return mvdr(Rn, h)


def all_weights(X, S, Nn, lead, active_frames, freqs, mics, cfg) -> tuple[dict, dict]:
    """Weights of every method for one scene + diagnostics (estimated DOA)."""
    M, F, _ = X.shape
    Rn = covariance(X, slice(0, lead))
    Ry = covariance(X, slice(lead, None))
    az, conf = doa_from_mixture(X, Rn, lead, freqs, mics)
    h_ff = farfield_h(mics, az, freqs)
    h_rtf = gevd_rtf(Ry, Rn)
    W = {
        "ref": ref_mic(M, F),
        "mean": mic_mean(M, F),
        "mute": mute_loudest(Rn),
        "mute_soft": mute_soft(Rn),
        "ds": delay_and_sum(h_ff),
        "mvdr_doa": mvdr(Rn, h_ff, cfg["loading_ff"]),
        "mvdr_rtf": mvdr(Rn, h_rtf, cfg["loading"]),
        "gsc": gsc(X, h_rtf, lead, cfg["gsc_mu"], cfg["gsc_gate_db"]),
        "oracle": oracle_mvdr(S, Nn, active_frames),
    }
    return W, {"doa_deg": float(np.rad2deg(az)), "doa_conf": conf}
