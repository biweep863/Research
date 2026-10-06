"""Interaural cue metrics (ITD, ILD, IPD) and their errors between an estimate
and the clean binaural target.

Conventions (used everywhere in this project):
- Signals are (2, T) with row 0 = left ear, row 1 = right ear.
- ITD > 0 means the right ear lags the left ear (source on the left), in seconds.
- ILD > 0 means the left ear is louder, in dB.
"""

from __future__ import annotations

import numpy as np
import scipy.signal

# Perceptual just-noticeable differences used as pass/fail lines (doc/02).
ILD_JND_DB = 1.0
ITD_JND_S = 20e-6


def itd_gcc_phat(
    x: np.ndarray,
    fs: int,
    band_hz: tuple[float, float] = (100.0, 1500.0),
    max_itd_s: float = 1e-3,
    upsample: int = 32,
) -> float:
    """Broadband ITD by band-limited GCC-PHAT with a sub-sample peak.

    Integer-lag peak from the plain IFFT, then the band-limited correlation is
    evaluated directly on a 1/``upsample``-sample grid within +/-1 sample of
    it (same answer as a zero-padded IFFT, ~30x cheaper).
    """
    left, right = x[0], x[1]
    n = 1 << int(np.ceil(np.log2(2 * len(left))))
    L = np.fft.rfft(left, n)
    R = np.fft.rfft(right, n)
    G = np.conj(L) * R
    G /= np.abs(G) + 1e-12
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= band_hz[0]) & (f <= band_hz[1])
    G[~band] = 0.0
    cc = np.fft.irfft(G, n)
    max_lag = int(np.ceil(max_itd_s * fs))
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    k0 = lags[np.argmax(np.concatenate([cc[: max_lag + 1], cc[-max_lag:]]))]
    tau = (k0 + np.linspace(-1, 1, 2 * upsample + 1)) / fs
    tau = tau[np.abs(tau) <= max_itd_s + 1e-12]
    fine = np.real(np.exp(2j * np.pi * np.outer(tau, f[band])) @ G[band])
    return float(tau[np.argmax(fine)])


def ild_broadband(x: np.ndarray) -> float:
    return float(10 * np.log10((np.sum(x[0] ** 2) + 1e-12) / (np.sum(x[1] ** 2) + 1e-12)))


def _tf(x: np.ndarray, fs: int, nfft: int = 512, hop: int = 128):
    f, _, X = scipy.signal.stft(x, fs=fs, nperseg=nfft, noverlap=nfft - hop, axis=-1)
    return f, X  # X: (2, F, N)


def tf_cue_errors(
    est: np.ndarray,
    ref: np.ndarray,
    fs: int,
    dyn_range_db: float = 20.0,
    ipd_max_hz: float = 1500.0,
) -> dict:
    """Energy-weighted per-bin |dILD| (dB) and |dIPD| (rad) over target-active bins.

    Active bins: target (left+right) energy within ``dyn_range_db`` of its max.
    IPD is only evaluated below ``ipd_max_hz`` where it is perceptually used.
    """
    f, E = _tf(est, fs)
    _, S = _tf(ref, fs)
    pw = np.abs(S[0]) ** 2 + np.abs(S[1]) ** 2
    active = pw > pw.max() * 10 ** (-dyn_range_db / 10)
    w = pw * active
    eps = 1e-10
    ild_e = 20 * np.log10((np.abs(E[0]) + eps) / (np.abs(E[1]) + eps))
    ild_s = 20 * np.log10((np.abs(S[0]) + eps) / (np.abs(S[1]) + eps))
    d_ild = float(np.sum(w * np.abs(ild_e - ild_s)) / (w.sum() + eps))
    ipd_e = E[0] * np.conj(E[1])
    ipd_s = S[0] * np.conj(S[1])
    d_ipd_bin = np.abs(np.angle(ipd_e * np.conj(ipd_s)))
    low = (f <= ipd_max_hz)[:, None]
    wl = w * low
    d_ipd = float(np.sum(wl * d_ipd_bin) / (wl.sum() + eps))
    return {"d_ild_tf_db": d_ild, "d_ipd_rad": d_ipd}


def cue_errors(est: np.ndarray, ref: np.ndarray, fs: int) -> dict:
    """All target-cue errors between estimate and clean target."""
    out = {
        "d_itd_us": abs(itd_gcc_phat(est, fs) - itd_gcc_phat(ref, fs)) * 1e6,
        "d_ild_db": abs(ild_broadband(est) - ild_broadband(ref)),
    }
    out.update(tf_cue_errors(est, ref, fs))
    return out


def residual_cue_shift(est: np.ndarray, mix: np.ndarray, ref: np.ndarray, fs: int) -> dict:
    """How much the *residual noise* moved in space: cues of (est - ref) vs (mix - ref).

    A common real gain leaves them unchanged; independent per-ear processing can
    collapse or move the noise image.
    """
    r_in = mix - ref
    r_out = est - ref
    return {
        "res_d_itd_us": abs(itd_gcc_phat(r_out, fs) - itd_gcc_phat(r_in, fs)) * 1e6,
        "res_d_ild_db": abs(ild_broadband(r_out) - ild_broadband(r_in)),
    }
