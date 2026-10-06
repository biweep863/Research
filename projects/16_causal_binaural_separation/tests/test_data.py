"""Data pipeline tests (offline: spherical-head HRTF + synthetic pools)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import torch

from src.analysis.binaural import ild_broadband, itd_gcc_phat
from src.data.corpora import Pool, _i16, _to_pool, speaker_split
from src.data.hrtf import load_hrtf, spherical_head
from src.data.mixtures import Mixer, fixed_set
from src.data.rooms import make_bank, make_brir
from src.data.synth import pink_noise, voiced_speech

FS = 16000
ROOMS = {
    "length_m": [4.0, 6.0], "width_m": [3.5, 5.0], "height_m": [2.5, 3.0], "rt60_s": [0.3, 0.5],
    "listener_height_m": [1.3, 1.6], "wall_margin_m": 1.0, "target_dist_m": [0.8, 1.5],
    "target_el_deg": [0.0, 20.0], "n_noise_dirs": 2, "ism_order": 4, "early_ms": 50.0, "brir_len_s": 0.4,
}


@pytest.mark.parametrize("name", ["spherical_head", "mit_kemar_normal_pinna"])
def test_hrtf_left_is_row0(name):
    h = load_hrtf(name, FS, "data/raw")
    for az, sign in ((90, 1), (-90, -1)):
        u = np.array([[np.cos(np.deg2rad(az)), np.sin(np.deg2rad(az)), 0.0]])
        x = np.zeros((2, 2048))
        x[:, : h.hrir.shape[-1]] = h.hrir[h.nearest(u)[0]]
        assert np.sign(ild_broadband(x)) == sign
        assert np.sign(itd_gcc_phat(x, FS)) == sign
        assert 400e-6 < abs(itd_gcc_phat(x, FS)) < 900e-6


def test_brir_early_is_prefix_and_rt60():
    h = spherical_head(FS)
    rng = np.random.default_rng(0)
    full, early = make_brir([6, 5, 3], 0.5, np.array([3, 2.5, 1.5]), np.array([4.2, 2.5, 1.5]), h, FS, rng,
                            ism_order=6, max_len_s=0.8)
    n_e = int(0.045 * FS)
    assert np.allclose(full[:, :n_e], early[:, :n_e], atol=1e-6)  # identical before the early cut
    assert np.sum(early[:, int(0.06 * FS):] ** 2) < 1e-8          # nothing in the target after it
    e = (full**2).sum(0)
    edc = 10 * np.log10(np.cumsum(e[::-1])[::-1] / e.sum())
    t60 = (np.argmax(edc < -25) - np.argmax(edc < -5)) / FS * 3
    assert 0.4 < t60 < 0.6


def test_speaker_split_disjoint_and_deterministic():
    spk = [str(i) for i in range(40)]
    a, b = speaker_split(spk, [0.75, 0.125, 0.125], 0), speaker_split(spk, [0.75, 0.125, 0.125], 0)
    assert a == b
    assert len(a["train"]) == 30 and len(a["val"]) == 5 and len(a["test"]) == 5
    assert not (set(a["train"]) & set(a["test"])) and not (set(a["val"]) & set(a["test"]))


def _mixer(snr):
    rng = np.random.default_rng(0)
    speech = _to_pool([_i16(0.1 * voiced_speech(3.0, FS, seed=i)) for i in range(3)], synthetic=True)
    noise = _to_pool([_i16(0.05 * pink_noise(4 * FS, rng))], synthetic=True)
    bank = make_bank(2, ROOMS, spherical_head(FS), FS, seed=0)
    cfg = {"fs": FS, "mix": {"snr_db": [snr, snr], "level_dbfs": [-25, -25], "warmup_s": 0.2}}
    return Mixer(speech, noise, bank, cfg, "cpu")


@pytest.mark.parametrize("snr", [-5.0, 10.0])
def test_mixer_snr_and_level(snr):
    m = _mixer(snr)
    out = m.sample(4, FS, np.random.default_rng(1), with_noise=True)
    assert out["mix"].shape == (4, 2, FS) and out["target"].shape == (4, 2, FS)
    snr_meas = 10 * torch.log10(out["target"].pow(2).mean((1, 2)) / out["noise"].pow(2).mean((1, 2)))
    assert torch.allclose(snr_meas, torch.full((4,), snr), atol=1e-3)
    rms = out["mix"].pow(2).mean((1, 2)).sqrt()
    assert torch.allclose(20 * torch.log10(rms), torch.full((4,), -25.0), atol=1e-3)
    assert m.synthetic


def test_fixed_set_is_deterministic():
    m = _mixer(5.0)
    a, b = fixed_set(m, 5, 4000, seed=3, batch=2), fixed_set(m, 5, 4000, seed=3, batch=2)
    assert torch.equal(a["mix"], b["mix"]) and a["mix"].shape == (5, 2, 4000)


def test_pool_crop_tiles_short_files():
    p = Pool(np.arange(100, dtype=np.int16), np.array([0]), np.array([100]))
    assert p.crop(250, np.random.default_rng(0)).shape == (250,)
