import numpy as np
import pytest

from src.analysis.distance import (
    DistanceModel, cdr_db, cdr_doa_dependent, cdr_doa_independent, diffuse_coherence,
)
from src.analysis.geometry import far_field_delays, respeaker_v2_positions

MICS = respeaker_v2_positions()


@pytest.mark.parametrize("true_cdr", [0.1, 1.0, 10.0])
def test_cdr_estimators_invert_the_mixture_model(true_cdr):
    f = np.linspace(1500, 6000, 50)
    gn = diffuse_coherence(f, 0.064)
    gs = np.exp(-2j * np.pi * f * 1e-4)
    gx = (true_cdr * gs + gn) / (true_cdr + 1)
    np.testing.assert_allclose(cdr_doa_dependent(gx, gn, gs), true_cdr, rtol=1e-6)
    np.testing.assert_allclose(cdr_doa_independent(gx, gn), true_cdr, rtol=1e-6)


def _phi(direct_power, diffuse_power, az=0.7):
    freqs = np.fft.rfftfreq(512, 1 / 16000)
    tau = far_field_delays(MICS, np.array([az]))[0]
    a = np.exp(-2j * np.pi * freqs[:, None] * tau[None, :])  # (F, M)
    D = np.linalg.norm(MICS[:, None] - MICS[None], axis=-1)
    G = np.sinc(2 * freqs[:, None, None] * D[None] / 343.0)
    return freqs, direct_power * np.einsum("fi,fj->fij", a, a.conj()) + diffuse_power * G


def test_cdr_db_monotone_in_direct_to_diffuse_ratio():
    vals = []
    for ratio in (10.0, 1.0, 0.1):
        freqs, Phi = _phi(ratio, 1.0)
        vals.append(cdr_db(Phi, freqs, MICS))
    assert vals[0] > vals[1] > vals[2]
    assert abs(vals[1]) < 0.5  # equal powers -> ~0 dB


def test_model_fit_recovers_log_linear_law_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    d = rng.uniform(0.5, 4, 60)
    rows = [{"cdr_db": -20 * np.log10(x) + 5, "level_db": -30 - 20 * np.log10(x) + rng.normal(0, 0.1),
             "snr_db": 0.0, "tilt_db": 0.0} for x in d]
    m = DistanceModel.fit(rows, d, ridge=1e-6)
    pred = np.array([m.predict(r) for r in rows])
    assert np.max(np.abs(pred - d)) < 0.05
    m.to_json(tmp_path / "m.json")
    m2 = DistanceModel.from_json(tmp_path / "m.json")
    assert m2.predict(rows[0]) == pytest.approx(m.predict(rows[0]))


def test_untrained_prior_is_inverse_square():
    m = DistanceModel(ref_level_db=-30.0)
    assert m.predict({"level_db": -30.0}) == pytest.approx(1.0)
    assert m.predict({"level_db": -36.0206}) == pytest.approx(2.0, rel=1e-3)
