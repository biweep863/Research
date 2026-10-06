import numpy as np
import pytest

from src.analysis.doa import (
    cross_spectra, doa_from_tdoas, gcc_phat_doa, harmonic_weights, peak_azimuth, srp_phat,
)
from src.analysis.geometry import far_field_delays, mic_pairs, respeaker_v2_positions
from src.analysis.metrics import angular_error_deg
from src.analysis.stft import stft
from tests.conftest import plane_wave

MICS = respeaker_v2_positions()
GRID = np.deg2rad(np.arange(0, 360, 1.0))


def _R(az, seed=0):
    x = np.random.default_rng(seed).standard_normal(16000)
    freqs, X = stft(plane_wave(x, az), 16000, 512, 256)
    return freqs, cross_spectra(X)


@pytest.mark.parametrize("az", [0, 37, 90, 181.5, 270, 359])
def test_srp_phat_recovers_azimuth(az):
    freqs, R = _R(az)
    P = srp_phat(R, freqs, MICS, GRID)
    est, conf = peak_azimuth(P, GRID)
    assert angular_error_deg(np.rad2deg(est), az) < 1.5
    assert conf > 0.8


@pytest.mark.parametrize("az", [10, 135, 300])
def test_gcc_phat_baseline(az):
    freqs, R = _R(az, seed=1)
    est = gcc_phat_doa(R, freqs, MICS, 512, 16000)
    assert angular_error_deg(np.rad2deg(est), az) < 5


def test_doa_from_exact_tdoas():
    az = np.deg2rad(222.0)
    tau = far_field_delays(MICS, np.array([az]))[0]
    tdoas = {(i, j): tau[i] - tau[j] for i, j in mic_pairs(4)}
    assert angular_error_deg(np.rad2deg(doa_from_tdoas(tdoas, MICS)), 222.0) < 1e-6


def test_peak_azimuth_wraps_across_zero():
    P = np.cos(GRID - np.deg2rad(359.6))
    est, _ = peak_azimuth(P, GRID)
    assert angular_error_deg(np.rad2deg(est), 359.6) < 0.2


def test_harmonic_weights_peak_on_harmonics():
    f = np.linspace(0, 4000, 4001)
    w = harmonic_weights(f, 150.0)
    assert w[300] > 0.99 and w[375] < 0.2
    assert np.all(harmonic_weights(f, 0.0) == 1)
