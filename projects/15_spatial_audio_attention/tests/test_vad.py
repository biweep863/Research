import numpy as np

from src.analysis.vad import adaptive_threshold, autocorr_pitch, frame_vad, update_floor
from tests.conftest import harmonic

FS = 16000


def test_pitch_of_harmonic_complex():
    for f0 in (100.0, 150.0, 220.0):
        est, per = autocorr_pitch(harmonic(f0, 0.064), FS)
        assert abs(est - f0) < 3.0
        assert per > 0.8


def test_white_noise_is_not_voiced():
    x = 0.1 * np.random.default_rng(0).standard_normal(1024)
    r = frame_vad(x, FS, energy_threshold=0.01, corr_threshold=0.6)
    assert not r.is_speech and r.reason == "non-vocal"


def test_silence_rejected_by_energy_gate():
    r = frame_vad(1e-4 * harmonic(150, 0.064), FS, energy_threshold=0.01, corr_threshold=0.6)
    assert not r.is_speech and r.reason == "silence"


def test_hum_below_vocal_band_rejected():
    t = np.arange(1024) / FS
    hum = 0.1 * np.sin(2 * np.pi * 50 * t)  # 50 Hz mains, below the 85 Hz floor
    assert not frame_vad(hum, FS, 0.01, 0.6).is_speech


def test_floor_follows_down_fast_up_slow():
    f = update_floor(None, 1.0, up=0.01, down=0.5)
    assert f == 1.0
    assert update_floor(f, 0.0, 0.01, 0.5) == 0.5
    assert abs(update_floor(f, 2.0, 0.01, 0.5) - 1.01) < 1e-12
    assert adaptive_threshold(None, 0.02, 6) == 0.02
    assert abs(adaptive_threshold(0.1, 0.02, 6) - 0.1 * 10 ** 0.3) < 1e-9
