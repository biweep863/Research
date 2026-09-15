"""Unit tests: sensor chain, event physics, features, detectors, onset timing,
two-pulse separation, Bayesian fusion, and the INS.  Tiny synthetic inputs
with known answers; no downloads.  `make test` must pass before experiments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

from src.sim.imu import SensorSpec, lowpass, decimate, sense            # noqa: E402
from src.sim.gait import synth_gait, half_sine                          # noqa: E402
from src.sim.events import inject, ball_sensor, CLASSES                 # noqa: E402
from src.sim.dataset import make_windows, clean_templates               # noqa: E402
from src.detect.features import window_features, FEATURE_NAMES          # noqa: E402
from src.detect.threshold import ThresholdDetector, macro_f1, confusion # noqa: E402
from src.detect.ml import GBDetector                                    # noqa: E402
from src.detect.onset import onset_threshold, onset_matched, order_probability  # noqa: E402
from src.detect.separate import separate                                # noqa: E402
from src.decide.bayes import Fusion, decide, brier, ece, timing_score   # noqa: E402
from src.localize.ins import (true_trajectory, imu_measurements, run_ins,  # noqa: E402
                              wrong_side_probability)

CFG = yaml.safe_load(open(HERE / "configs" / "local_small.yaml"))
SIM = CFG["sim"]


# --------------------------------------------------------------------------- #
# Sensor chain
# --------------------------------------------------------------------------- #
def test_half_sine_and_lowpass():
    t = np.linspace(0, 1, 4001)
    p = half_sine(t, 0.5, 0.01)
    assert abs(p.max() - 1.0) < 1e-3 and p[t < 0.5].max() == 0.0
    x = np.sin(2 * np.pi * 5 * t)[:, None]
    y = lowpass(x, 500.0, 4000.0)
    assert np.abs(y[2000:] - x[2000:]).max() < 0.05     # 5 Hz passes a 500 Hz LPF
    assert decimate(x, 4000.0, 1000.0).shape[0] == 1001


def test_sensor_clips_and_adds_noise():
    rng = np.random.default_rng(0)
    T = 4000
    acc = np.zeros((T, 3)); acc[:, 2] = 1.0; acc[2000, 0] = 100.0
    gyr = np.zeros((T, 3))
    spec = SensorSpec(fs=1000, range_g=16.0, acc_noise_g=0.01, bias_g=0.0)
    a, w = sense(acc, gyr, spec, 4000.0, rng)
    assert a.shape == (1000, 3) and np.abs(a).max() <= 16.0 + 1e-9
    assert abs(a[:, 2].mean() - 1.0) < 0.02


# --------------------------------------------------------------------------- #
# Physics of the events
# --------------------------------------------------------------------------- #
def test_gait_has_one_g_at_rest_and_periodic_strikes():
    rng = np.random.default_rng(1)
    t, acc, gyr = synth_gait(2.0, 4000.0, rng)
    mag = np.linalg.norm(acc, axis=1)
    assert 0.5 < np.median(mag) < 3.0
    assert mag.max() > 5.0                                 # foot strikes present


def test_ball_is_shorter_and_harder_than_player():
    rng = np.random.default_rng(2)
    t = np.arange(int(0.6 * 4000)) / 4000
    peaks, widths = {}, {}
    for lab in ("ball", "player"):
        sig, truth = inject(t, lab, rng, SIM, t_center=0.3)
        mag = np.linalg.norm(sig[:, :3], axis=1)
        peaks[lab] = mag.max()
        widths[lab] = np.sum(mag > 0.5 * mag.max()) / 4000
        assert abs((truth.t_ball if lab == "ball" else truth.t_player) - 0.3) < 1e-9
    assert peaks["ball"] > peaks["player"] and widths["ball"] < widths["player"]


def test_both_gap_and_ball_sensor():
    rng = np.random.default_rng(3)
    t = np.arange(int(0.6 * 4000)) / 4000
    _, truth = inject(t, "both", rng, SIM, t_center=0.3, gap_ms=-12.0)
    assert abs(truth.gap_ms + 12.0) < 1e-9 and truth.ball_first is False
    rep = ball_sensor(truth, 0.6, CFG["sensors"]["ball"], SIM, rng)
    assert rep["detected"] and abs(rep["t_ball_hat"] - truth.t_ball) < 0.006
    _, none = inject(t, "player", rng, SIM)
    assert not ball_sensor(none, 0.6, CFG["sensors"]["ball"], SIM, rng)["detected"]


# --------------------------------------------------------------------------- #
# Features and detectors
# --------------------------------------------------------------------------- #
def _small_set(rng, fs=500, n=25, **kw):
    return make_windows(n, fs, CFG["sensors"]["sport"], SIM, rng,
                        ball_cfg=CFG["sensors"]["ball"], **kw)


def test_features_shape_and_finite():
    rng = np.random.default_rng(4)
    ws = _small_set(rng, n=3)
    F = window_features(ws.X[0], ws.fs)
    assert F.shape == (len(FEATURE_NAMES),) and np.all(np.isfinite(F))


def test_detectors_beat_chance():
    rng = np.random.default_rng(5)
    tr, te = _small_set(rng, n=40), _small_set(rng, n=20)
    thr = ThresholdDetector().fit(tr.X, tr.y, tr.fs)
    gb = GBDetector(max_iter=60).fit(tr.X, tr.y, tr.fs)
    f_thr = macro_f1(te.y, thr.predict(te.X, te.fs))
    f_gb = macro_f1(te.y, gb.predict(te.X))
    assert f_thr > 0.4 and f_gb > 0.7
    M = confusion(te.y, gb.predict(te.X))
    assert M.sum() == len(te.y) and M.shape == (4, 4)


# --------------------------------------------------------------------------- #
# Onset timing and ordering theory
# --------------------------------------------------------------------------- #
def test_onset_within_a_few_ms_at_1khz():
    rng = np.random.default_rng(6)
    ws = make_windows(30, 1000, CFG["sensors"]["sport"], SIM, rng, classes=["ball"])
    errs = [1000 * (onset_threshold(w, 1000) - tr.t_ball) for w, tr in zip(ws.X, ws.truths)]
    tpl = clean_templates(1000, SIM, rng, n_avg=10)["ball"]
    errs_c = [1000 * (onset_matched(w, 1000, tpl) - tr.t_ball) for w, tr in zip(ws.X, ws.truths)]
    assert np.sqrt(np.mean(np.square(errs))) < 4.0
    assert np.sqrt(np.mean(np.square(errs_c))) < 6.0


def test_order_probability_limits():
    assert abs(order_probability(0.0, 2.0) - 0.5) < 1e-9
    assert order_probability(50.0, 2.0) > 0.999


# --------------------------------------------------------------------------- #
# Two-pulse separation
# --------------------------------------------------------------------------- #
def test_separator_recovers_order_at_wide_gap():
    rng = np.random.default_rng(7)
    fs = 1000
    tpl = clean_templates(fs, SIM, rng, n_avg=10)
    ok = 0
    for gap in (25.0, -25.0):
        ws = make_windows(6, fs, CFG["sensors"]["sport"], SIM, rng, classes=["both"], gap_ms=gap)
        for w, tr in zip(ws.X, ws.truths):
            r = separate(w, fs, tpl, step_ms=2.0)
            ok += (r["order"] == "ball_first") == tr.ball_first
    assert ok >= 9   # of 12


# --------------------------------------------------------------------------- #
# Bayesian fusion and decision rule
# --------------------------------------------------------------------------- #
def test_fusion_improves_brier_and_decision_thresholds():
    rng = np.random.default_rng(8)
    n = 600
    y = rng.random(n) < 0.6
    s1 = np.clip(0.5 + 0.35 * (2 * y - 1) + rng.normal(0, 0.25, n), 0, 1)
    s2 = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(0, 0.25, n), 0, 1)
    f = Fusion(["a", "b"]).fit({"a": s1[:300], "b": s2[:300]}, y[:300])
    prior = np.full(300, 0.6)
    p_a = f.posterior({"a": s1[300:], "b": s2[300:]}, prior, use=["a"])
    p_ab = f.posterior({"a": s1[300:], "b": s2[300:]}, prior)
    assert brier(p_ab, y[300:]) < brier(p_a, y[300:])
    e, rows = ece(p_ab, y[300:].astype(float), 5)
    assert 0 <= e < 0.5 and len(rows) > 0
    d = decide(np.array([0.2, 0.55, 0.9]), 1.0, 1.0, review_band=0.1)
    assert list(d) == [0, 2, 1]
    d2 = decide(np.array([0.55]), 3.0, 1.0)        # false fouls costly -> threshold 0.25
    assert d2[0] == 1
    assert timing_score(None, 0.3, 3.0) == 0.0
    assert timing_score(0.3, 0.3, 3.0) == 0.5 and timing_score(0.29, 0.3, 3.0) > 0.9


# --------------------------------------------------------------------------- #
# INS localisation
# --------------------------------------------------------------------------- #
def test_ins_zupt_and_uwb_reduce_error():
    rng = np.random.default_rng(9)
    fs = 200
    t, p, v, a, pitch, yaw, stance = true_trajectory(3.0, fs, 5.0, rng)
    assert stance.any() and np.linalg.norm(v[stance], axis=1).max() == 0.0
    grade = {"acc_noise_g": 0.05, "gyr_noise_dps": 0.5, "gyr_bias_dps": 1.0}
    f_b, w_b = imu_measurements(t, a, pitch, yaw, fs, grade, rng)
    e_free = run_ins(t, f_b, w_b, fs, 0.0, 0.1, p, rng, zupt=False)
    e_zupt = run_ins(t, f_b, w_b, fs, 0.0, 0.1, p, rng, zupt=True)
    e_uwb = run_ins(t, f_b, w_b, fs, 10.0, 0.1, p, rng, zupt=True)
    err = lambda e: np.linalg.norm(e[-1] - p[-1])
    assert err(e_zupt) < err(e_free) and err(e_uwb) < err(e_free)
    assert err(e_uwb) < 0.5
    assert abs(wrong_side_probability(0.0, 0.2) - 0.5) < 1e-9
    assert wrong_side_probability(1.0, 0.2) < 1e-4
