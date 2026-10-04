import numpy as np

from src.analysis.beamform import delay_and_sum_weights, mvdr_weights
from src.analysis.geometry import respeaker_v2_positions, steering_vectors
from src.analysis.metrics import si_sdr
from src.analysis.tracker import SpeakerTracker, circular_mean_deg, wrap_deg

MICS = respeaker_v2_positions()


def test_circular_mean_does_not_average_to_180():
    # the bug in home2 respeaker.py MovingAverage: mean(359, 1) = 180
    assert abs(wrap_deg(circular_mean_deg([359, 1]))) < 1e-9


def test_tracker_crosses_zero_without_jumping():
    tr = SpeakerTracker(alpha=0.6, beta=0.05)
    for k, az in enumerate([350, 354, 358, 2, 6, 10]):
        s = tr.update(0.1 * k, az, 1.5, 0.9)
        assert abs(wrap_deg(s.az_deg - az)) < 10
    assert abs(wrap_deg(s.az_deg - 10)) < 5


def test_tracker_ignores_low_confidence_and_times_out():
    tr = SpeakerTracker(hold_s=1.0, min_confidence=0.3)
    tr.update(0.0, 90, 2.0, 0.9)
    s = tr.update(0.1, 270, 2.0, 0.1)
    assert abs(wrap_deg(s.az_deg - 90)) < 1e-9
    assert not tr.predict(2.0).active


def test_tracker_range_is_log_smoothed():
    tr = SpeakerTracker(range_alpha=0.5)
    tr.update(0.0, 0, 1.0, 0.9)
    s = tr.update(0.1, 0, 4.0, 0.9)
    assert abs(s.range_m - 2.0) < 1e-9  # geometric mean


def test_beamformers_are_distortionless():
    f = np.fft.rfftfreq(512, 1 / 16000)
    az = np.deg2rad(40)
    a = steering_vectors(MICS, np.array([az]), f)[0]
    for W in (delay_and_sum_weights(MICS, az, f),
              mvdr_weights(np.tile(np.eye(4), (len(f), 1, 1)).astype(complex), MICS, az, f)):
        np.testing.assert_allclose(np.einsum("fm,fm->f", W.conj(), a), 1.0, atol=1e-9)


def test_mvdr_attenuates_point_interferer():
    f = np.fft.rfftfreq(512, 1 / 16000)
    sel = (f > 500) & (f < 3800)
    ai = steering_vectors(MICS, np.deg2rad([150]), f)[0]
    Rn = 100 * np.einsum("fi,fj->fij", ai, ai.conj()) + 0.01 * np.eye(4)[None]
    W = mvdr_weights(Rn, MICS, np.deg2rad(20), f, loading=1e-4)
    Wds = delay_and_sum_weights(MICS, np.deg2rad(20), f)
    g_mvdr = np.abs(np.einsum("fm,fm->f", W.conj(), ai))[sel]
    g_ds = np.abs(np.einsum("fm,fm->f", Wds.conj(), ai))[sel]
    assert np.median(g_mvdr) < 0.1 * np.median(g_ds)


def test_si_sdr_basic():
    x = np.random.default_rng(0).standard_normal(1000)
    assert si_sdr(3 * x, x) > 100
    assert abs(si_sdr(x + np.random.default_rng(1).standard_normal(1000), x)) < 1.0
