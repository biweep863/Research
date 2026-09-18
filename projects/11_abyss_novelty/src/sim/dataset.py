"""Build labelled log-mel clips for K known + U unknown synthetic species."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..novelty.features import log_mel
from .propagate import received
from .species import FAMILIES, Species, make_species, synth_call


@dataclass
class ClipSet:
    X: np.ndarray            # (N, n_mels, n_frames)
    y: np.ndarray            # species index
    known: np.ndarray        # bool
    snr: np.ndarray
    species: list[Species]


def _centre_freq(sp: Species) -> float:
    p = sp.params
    if sp.family == "tonal":
        return p["f0"]
    return p["fc"]


def make_species_set(n_known: int, n_unknown: int, rng: np.random.Generator, families=None,
                     sl_range=(150.0, 185.0)) -> list[Species]:
    fams = list(families or FAMILIES)
    out = []
    for i in range(n_known + n_unknown):
        out.append(make_species(i, fams[i % len(fams)], rng, sl_range))
    return out


def make_clips(species: list[Species], n_known: int, clips_per_species: int, audio: dict, sim: dict,
               rng: np.random.Generator, range_km=None) -> ClipSet:
    fs, clip_s = audio["fs_hz"], audio["clip_s"]
    r_lo, r_hi = range_km or sim["range_km"]
    X, y, known, snrs = [], [], [], []
    for sp in species:
        for _ in range(clips_per_species):
            call = synth_call(sp, fs, clip_s, rng)
            r = float(10 ** rng.uniform(np.log10(r_lo), np.log10(r_hi)) * 1000.0)
            x, snr = received(call, sp.sl_db, _centre_freq(sp), r, fs, sim["sea_state"], sim["shipping"], rng)
            X.append(log_mel(x, fs, audio["n_mels"], audio["n_frames"], audio["n_fft"], audio["fmin_hz"], audio["fmax_hz"]))
            y.append(sp.idx); known.append(sp.idx < n_known); snrs.append(snr)
    return ClipSet(np.stack(X), np.asarray(y), np.asarray(known), np.asarray(snrs), species)
