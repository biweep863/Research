"""Synthetic bioacoustic species: parametric call families.

A *species* is a call family plus a species-specific parameter distribution
(narrow, so two species of the same family differ in a stable way, as real
species do).  Four families cover the deep-sea repertoire:

* tonal  -- frequency-modulated whistle/moan with harmonics (baleen moans,
            dolphin whistles),
* click  -- train of short broadband clicks with a characteristic
            inter-click interval and centre frequency (sperm/beaked whales),
* drum   -- pulsed low-frequency knocks from a swim bladder (fishes),
* burst  -- broadband noisy pulse (crustacean snaps, unknown mechanisms).

All calls are generated as pressure time series at the source and later
attenuated / mixed with ambient noise by ``propagate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAMILIES = ["tonal", "click", "drum", "burst"]


@dataclass
class Species:
    idx: int
    family: str
    params: dict = field(default_factory=dict)
    sl_db: float = 170.0


def make_species(idx: int, family: str, rng: np.random.Generator, sl_range=(150.0, 185.0)) -> Species:
    p: dict = {}
    if family == "tonal":
        p = {"f0": rng.uniform(150, 3000), "sweep": rng.uniform(-1.5, 1.5),      # octaves over the call
             "dur": rng.uniform(0.2, 1.2), "n_harm": int(rng.integers(1, 5)),
             "harm_decay": rng.uniform(0.3, 0.8), "vibrato_hz": rng.uniform(0, 12),
             "vibrato_depth": rng.uniform(0, 0.05)}
    elif family == "click":
        p = {"ici": rng.uniform(0.02, 0.4), "fc": rng.uniform(1500, 7000), "bw": rng.uniform(500, 3000),
             "click_dur": rng.uniform(0.0003, 0.003), "n_clicks": int(rng.integers(3, 30))}
    elif family == "drum":
        p = {"rate": rng.uniform(5, 60), "fc": rng.uniform(80, 600), "pulse_dur": rng.uniform(0.01, 0.05),
             "dur": rng.uniform(0.3, 1.5), "q": rng.uniform(2, 10)}
    elif family == "burst":
        p = {"dur": rng.uniform(0.005, 0.2), "fc": rng.uniform(500, 6000), "bw": rng.uniform(1000, 6000),
             "n": int(rng.integers(1, 4)), "spacing": rng.uniform(0.05, 0.5)}
    else:
        raise ValueError(family)
    return Species(idx, family, p, float(rng.uniform(*sl_range)))


def _jitter(v: float, rng: np.random.Generator, frac: float = 0.12) -> float:
    return v * float(np.exp(rng.normal(0, frac)))


def synth_call(sp: Species, fs: float, clip_s: float, rng: np.random.Generator) -> np.ndarray:
    """One call realisation (unit RMS over the active part), placed at a random
    offset inside a clip of ``clip_s`` seconds."""
    n = int(fs * clip_s)
    t = np.arange(n) / fs
    p = sp.params
    y = np.zeros(n)
    if sp.family == "tonal":
        dur = min(_jitter(p["dur"], rng), clip_s * 0.9)
        m = t < dur
        tt = t[m]
        f0 = _jitter(p["f0"], rng)
        sweep = p["sweep"] + rng.normal(0, 0.1)
        f_inst = f0 * 2 ** (sweep * tt / dur) * (1 + p["vibrato_depth"] * np.sin(2 * np.pi * p["vibrato_hz"] * tt))
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
        env = np.sin(np.pi * tt / dur) ** 0.5
        for h in range(1, p["n_harm"] + 1):
            y[m] += p["harm_decay"] ** (h - 1) * np.sin(h * phase) * env
    elif sp.family == "click":
        ici = _jitter(p["ici"], rng)
        n_clicks = max(1, int(p["n_clicks"] * np.exp(rng.normal(0, 0.2))))
        fc, bw = _jitter(p["fc"], rng), p["bw"]
        for k in range(n_clicks):
            t0 = 0.05 + k * ici * (1 + rng.normal(0, 0.05))
            if t0 > clip_s - 0.01:
                break
            tau = p["click_dur"]
            u = t - t0
            m = (u >= 0) & (u < 6 * tau)
            y[m] += np.exp(-u[m] / tau) * np.sin(2 * np.pi * fc * u[m]) * (1 + 0.3 * rng.normal())
        # broadband part
        y += 0.0
    elif sp.family == "drum":
        rate = _jitter(p["rate"], rng)
        dur = min(_jitter(p["dur"], rng), clip_s * 0.9)
        fc = _jitter(p["fc"], rng)
        k = 0
        while k / rate < dur:
            t0 = k / rate
            u = t - t0
            m = (u >= 0) & (u < p["pulse_dur"] * 4)
            y[m] += np.exp(-u[m] * fc / p["q"]) * np.sin(2 * np.pi * fc * u[m])
            k += 1
    elif sp.family == "burst":
        for k in range(p["n"]):
            t0 = 0.05 + k * p["spacing"]
            dur = _jitter(p["dur"], rng)
            m = (t >= t0) & (t < t0 + dur)
            noise = rng.normal(size=m.sum())
            # band-limit the noise around fc with a simple FFT mask
            N = len(noise)
            if N > 8:
                F = np.fft.rfft(noise)
                fr = np.fft.rfftfreq(N, 1 / fs)
                F *= np.exp(-0.5 * ((fr - p["fc"]) / (p["bw"] / 2)) ** 2)
                noise = np.fft.irfft(F, n=N)
            y[m] += noise * np.hanning(m.sum())
    rms = np.sqrt(np.mean(y[y != 0] ** 2)) if np.any(y != 0) else 1.0
    y = y / max(rms, 1e-9)
    # random placement inside the clip
    shift = int(rng.uniform(0, max(1, n - np.flatnonzero(y)[-1] - 1))) if np.any(y != 0) else 0
    return np.roll(y, shift)
