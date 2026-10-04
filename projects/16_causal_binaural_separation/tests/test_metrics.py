"""Metric unit tests with analytic answers (no data, no models)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.analysis.binaural import cue_errors, ild_broadband, itd_gcc_phat, residual_cue_shift, tf_cue_errors
from src.analysis.metrics import estoi, si_sdr, si_sdr_binaural, si_sdr_improvement

FS = 16000


def _noise(n=FS, seed=0):
    return np.random.default_rng(seed).standard_normal(n)


def _frac_delay(x, d_s):
    n = len(x)
    f = np.fft.rfftfreq(2 * n, 1 / FS)
    return np.fft.irfft(np.fft.rfft(x, 2 * n) * np.exp(-2j * np.pi * f * d_s), 2 * n)[:n]


def test_si_sdr_scaled_copy_is_huge_and_noise_is_low():
    s = _noise()
    assert si_sdr(3.0 * s, s) > 100
    assert si_sdr(_noise(seed=1), s) < -10


def test_si_sdr_known_value():
    s = _noise()
    e = _noise(seed=1)
    e -= e.mean()
    e = e - (e @ (s - s.mean())) / ((s - s.mean()) @ (s - s.mean())) * (s - s.mean())  # orthogonal
    e *= np.linalg.norm(s - s.mean()) / np.linalg.norm(e) * 10 ** (-10 / 20)
    assert abs(si_sdr(s + e, s) - 10.0) < 1e-6


def test_si_sdr_improvement_sign():
    s = np.stack([_noise(seed=2), _noise(seed=3)])
    n = np.stack([_noise(seed=4), _noise(seed=5)])
    assert si_sdr_improvement(s + 0.1 * n, s + n, s) > 15
    assert abs(si_sdr_binaural(s, s) - si_sdr(s[0], s[0])) < 1e-6 or si_sdr_binaural(s, s) > 100


@pytest.mark.parametrize("itd_us", [-500.0, -37.5, 0.0, 12.5, 250.0])
def test_itd_recovers_subsample_delay(itd_us):
    left = _noise()
    right = _frac_delay(left, itd_us * 1e-6)
    est = itd_gcc_phat(np.stack([left, right]), FS) * 1e6
    assert abs(est - itd_us) < 3.0  # 1/32 sample = 1.95 us grid


@pytest.mark.parametrize("ild_db", [-6.0, 0.0, 3.0])
def test_ild_of_gain(ild_db):
    s = _noise()
    x = np.stack([s * 10 ** (ild_db / 20), s])
    assert abs(ild_broadband(x) - ild_db) < 1e-9


def test_cue_errors_zero_for_identical_and_common_gain():
    s = np.stack([_noise(), _frac_delay(_noise(), 2e-4) * 0.5])
    e = cue_errors(0.3 * s, s, FS)  # common gain changes no cue
    assert e["d_itd_us"] < 3 and e["d_ild_db"] < 1e-9 and e["d_ild_tf_db"] < 1e-6 and e["d_ipd_rad"] < 1e-6


def test_cue_errors_detect_swapped_ears():
    s = np.stack([_noise(), _frac_delay(_noise(), 3e-4) * 0.5])
    e = cue_errors(s[::-1].copy(), s, FS)
    assert e["d_ild_db"] > 11 and e["d_itd_us"] > 500 and tf_cue_errors(s[::-1].copy(), s, FS)["d_ild_tf_db"] > 5


def test_residual_shift_zero_under_common_gain():
    s = np.stack([_noise(seed=7), _noise(seed=8)])
    n = np.stack([_noise(seed=9), 0.3 * _noise(seed=9)])
    mix = s + n
    est = s + 0.2 * n  # residual is a scaled copy of the input noise
    r = residual_cue_shift(est, mix, s, FS)
    assert r["res_d_ild_db"] < 1e-6 and r["res_d_itd_us"] < 3


def test_estoi_ordering():
    t = np.arange(2 * FS) / FS
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    s = np.stack([env * _noise(2 * FS, 1), env * _noise(2 * FS, 2)])
    n = np.stack([_noise(2 * FS, 3), _noise(2 * FS, 4)])
    assert estoi(s + 0.1 * n, s, FS) > estoi(s + 3 * n, s, FS)
