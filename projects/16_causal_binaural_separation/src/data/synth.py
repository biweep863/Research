"""Offline fallbacks: speech-like signals and a spherical-head HRTF.

``voiced_speech`` and ``pink_noise`` are copied from
``projects/15_spatial_audio_attention/src/data/simulate.py`` (projects do not
import each other). They are only used when the real corpora are missing
(unit tests, no network); a model trained on them does not transfer to real
speech, and the experiments say so in their output.
"""

from __future__ import annotations

import numpy as np

# Rough vowel formants (Hz) - enough spectral variety for MFCC/tilt features.
_VOWELS = [(730, 1090, 2440), (270, 2290, 3010), (300, 870, 2240), (530, 1840, 2480), (570, 840, 2410)]


def voiced_speech(
    duration_s: float,
    fs: int,
    f0: float = 140.0,
    seed: int = 0,
    syllable_hz: float = 4.0,
    pause_prob: float = 0.2,
) -> np.ndarray:
    """Speech-like signal: harmonic source with F0 contour, vowel formants,
    syllabic envelope, word pauses and a little aspiration noise. Unit RMS
    over the voiced part. Generalises home2 ``vad_benchmark.gen_voiced_speech``.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    drift = np.cumsum(rng.normal(0, 0.002, n))
    drift -= np.linspace(drift[0], drift[-1], n)
    f0_t = f0 * (1 + 0.06 * np.sin(2 * np.pi * 0.8 * t + rng.uniform(0, 6.28)) + drift)
    phase = 2 * np.pi * np.cumsum(f0_t) / fs

    syl_len = int(fs / syllable_hz)
    n_syl = int(np.ceil(n / syl_len))
    out = np.zeros(n)
    k_max = int((fs / 2 - 200) // (f0 * 1.1))
    for s in range(n_syl):
        a, b = s * syl_len, min(n, (s + 1) * syl_len)
        if rng.random() < pause_prob:
            continue
        formants = np.array(_VOWELS[rng.integers(len(_VOWELS))]) * rng.uniform(0.9, 1.1)
        seg = np.zeros(b - a)
        for k in range(1, k_max + 1):
            fk = k * f0
            amp = (1.0 / k) * sum(np.exp(-0.5 * ((fk - F) / (60 + 0.06 * F)) ** 2) for F in formants)
            seg += (amp + 0.02 / k) * np.sin(k * phase[a:b])
        env = np.sin(np.pi * np.linspace(0, 1, b - a)) ** 0.6
        out[a:b] = seg * env
    voiced = np.abs(out) > 0
    out += 0.02 * rng.standard_normal(n) * voiced
    rms = np.sqrt(np.mean(out[voiced] ** 2)) if voiced.any() else 1.0
    return out / rms


def pink_noise(n: int, rng) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec))
    spec /= np.sqrt(np.maximum(f, 1))
    x = np.fft.irfft(spec, n)
    return x / x.std()




def spherical_head_hrirs(
    fs: int,
    head_radius: float = 0.0875,
    c: float = 343.0,
    n_taps: int = 128,
    az_step_deg: float = 5.0,
    elevations_deg=(-20.0, 0.0, 20.0, 40.0),
):
    """Brown & Duda (1998) spherical-head model: Woodworth ITD + one-pole/one-zero
    head shadow per ear. Returns (az_deg, el_deg, hrir (D, 2, n_taps)).

    Head frame: x front, y left, z up; ears on the +/- y axis.
    """
    az = np.arange(-180.0, 180.0, az_step_deg)
    A, E = np.meshgrid(az, np.asarray(elevations_deg, float), indexing="ij")
    A, E = A.ravel(), E.ravel()
    u = np.stack([np.cos(np.deg2rad(E)) * np.cos(np.deg2rad(A)),
                  np.cos(np.deg2rad(E)) * np.sin(np.deg2rad(A)),
                  np.sin(np.deg2rad(E))], axis=1)
    nfft = 4 * n_taps
    w = 2 * np.pi * np.fft.rfftfreq(nfft, 1 / fs)
    w0 = c / head_radius
    out = np.zeros((len(A), 2, n_taps))
    for e, ear in enumerate((np.array([0, 1, 0]), np.array([0, -1, 0]))):
        theta = np.degrees(np.arccos(np.clip(u @ ear, -1, 1)))  # angle source <-> ear axis
        alpha = 1.05 + 0.95 * np.cos(np.deg2rad(theta / 150.0 * 180.0))
        # Woodworth path length to this ear, relative to the head centre
        ang = np.deg2rad(theta)
        tau = np.where(ang < np.pi / 2, -head_radius * np.cos(ang) / c,
                       head_radius * (ang - np.pi / 2) / c)
        tau = tau + head_radius / c + 2e-3  # keep every arrival causal inside the FIR
        H = (1 + 1j * alpha[:, None] * w[None] / (2 * w0)) / (1 + 1j * w[None] / (2 * w0))
        H = H * np.exp(-1j * w[None] * tau[:, None])
        h = np.fft.irfft(H, nfft)[:, :n_taps]
        out[:, e] = h * np.hanning(2 * n_taps)[n_taps:][None]
    return A, E, out
