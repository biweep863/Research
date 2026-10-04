import numpy as np

from src.analysis.localizer import Localizer, analyze
from src.analysis.metrics import angular_error_deg
from tests.conftest import harmonic, plane_wave


def _clip(az, fs=16000):
    rng = np.random.default_rng(3)
    speech = harmonic(140, 1.2, amp=0.05)
    x = np.concatenate([np.zeros(int(0.5 * fs)), speech])
    mics = plane_wave(x, az)
    return mics + 1e-3 * rng.standard_normal(mics.shape)


def test_localizer_tracks_static_talker(cfg):
    loc, frames = analyze(_clip(123.0), cfg)
    assert any(f.is_speech for f in frames)
    assert not any(f.is_speech for f in frames[:20])  # lead silence
    last = frames[-1]
    assert last.track_active
    assert angular_error_deg(last.track_az_deg, 123.0) < 3
    assert loc.totals.n_n > 0  # silence fed the noise model
    s = loc.clip_summary()
    assert angular_error_deg(s["az_deg_pooled"], 123.0) < 2
    assert abs(s["f0_median_hz"] - 140) < 5


def test_streaming_is_block_size_invariant(cfg):
    x = _clip(300.0)
    _, a = analyze(x, cfg, block=97)
    _, b = analyze(x, cfg, block=4096)
    assert len(a) == len(b)
    np.testing.assert_allclose([f.rms for f in a], [f.rms for f in b])
    assert [f.is_speech for f in a] == [f.is_speech for f in b]


def test_frame_count(cfg):
    loc = Localizer(cfg)
    out = loc.process(np.zeros((4, 512 + 256 * 9)))
    assert len(out) == 10
