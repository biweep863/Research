import numpy as np
import pytest

from src.analysis import spatial
from src.analysis.geometry import SPEED_OF_SOUND, far_field_delays, respeaker_v2_positions
from src.analysis.metrics import si_sdr, snr_db, wer
from src.analysis.stft import istft, stft
from src.data import noises
from src.data.align import align

FS = 16000
MICS = respeaker_v2_positions(0.032)


def plane_wave(signal, az_deg):
    tau = far_field_delays(MICS, np.deg2rad([az_deg]), SPEED_OF_SOUND)[0]
    n = len(signal)
    S = np.fft.rfft(signal)
    f = np.fft.rfftfreq(n, 1 / FS)
    return np.stack([np.fft.irfft(S * np.exp(-2j * np.pi * f * t), n) for t in tau])


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    T = 4 * FS
    s = plane_wave(noises.babble_synth(T, FS, rng, n_talkers=1), 90.0)
    s[:, : FS] = 0  # talker silent for the first second (the lead)
    n = plane_wave(noises.pink(T, FS, rng), 200.0) + 0.01 * rng.standard_normal((4, T))
    return s, n


def test_mvdr_distortionless_and_nulls(scene):
    s, n = scene
    f, S = stft(s, FS, 512, 128)
    _, N = stft(n, FS, 512, 128)
    X = S + N
    Rn = spatial.covariance(X, slice(0, FS // 128))
    h = spatial.gevd_rtf(spatial.covariance(X, slice(FS // 128, None)), Rn)
    W = spatial.mvdr(Rn, h)
    band = (f > 300) & (f < 6000)
    resp_t = np.abs(np.einsum("fm,fm->f", W.conj(), spatial.farfield_h(MICS, np.deg2rad(90), f)))[band]
    resp_n = np.abs(np.einsum("fm,fm->f", W.conj(), spatial.farfield_h(MICS, np.deg2rad(200), f)))[band]
    assert np.median(np.abs(resp_t - 1)) < 0.05  # distortionless toward the talker
    assert np.median(20 * np.log10(resp_n + 1e-12)) < -20  # deep null toward the interferer
    T = s.shape[1]
    so = istft(spatial.apply(W, S), FS, 512, 128, T)
    no = istft(spatial.apply(W, N), FS, 512, 128, T)
    # broadband gain is capped by the bass (pink noise power < 300 Hz, where a 6.4 cm
    # aperture cannot null without amplifying the mics' self-noise)
    assert snr_db(so, no) - snr_db(s[0], n[0]) > 6


def test_gsc_beats_reference(scene):
    s, n = scene
    f, S = stft(s, FS, 512, 128)
    _, N = stft(n, FS, 512, 128)
    X = S + N
    lead = FS // 128
    h = spatial.farfield_h(MICS, np.deg2rad(90), f)
    W = spatial.gsc(X, h, lead)
    T = s.shape[1]
    so = istft(spatial.apply(W, S), FS, 512, 128, T)
    no = istft(spatial.apply(W, N), FS, 512, 128, T)
    assert snr_db(so, no) - snr_db(s[0], n[0]) > 6


def test_muting_one_mic_barely_helps(scene):
    """The user's idea, analytically: on a 6.4 cm array every omni mic hears a
    far-field interferer at the same level, so dropping one changes little."""
    s, n = scene
    f, S = stft(s, FS, 512, 128)
    _, N = stft(n, FS, 512, 128)
    Rn = spatial.covariance(S + N, slice(0, FS // 128))
    T = s.shape[1]
    gains = []
    for W in (spatial.mute_loudest(Rn), spatial.mute_soft(Rn)):
        so = istft(spatial.apply(W, S), FS, 512, 128, T)
        no = istft(spatial.apply(W, N), FS, 512, 128, T)
        gains.append(snr_db(so, no) - snr_db(s[0], n[0]))
    assert max(gains) < 1.5


def test_reference_weights_are_identity(scene):
    s, _ = scene
    f, S = stft(s, FS, 512, 128)
    y = istft(spatial.apply(spatial.ref_mic(4, len(f)), S), FS, 512, 128, s.shape[1])
    assert si_sdr(y[FS:], s[0, FS:]) > 60


def test_chirp_alignment_recovers_delay_and_drift():
    fs_play, fs_rec = 48000, 16000
    tl = {"fs": fs_play, "chirp1": 28800, "chirp2": 28800 + 10 * fs_play}
    ch = noises.chirp(fs_rec)
    true_off, ratio = 14571.3, 1.00004
    T = int(14 * fs_rec)
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal(T)
    for k in (0, 1):
        start = true_off + k * 10 * fs_rec * ratio
        t = np.arange(len(ch)) + start
        x += np.interp(np.arange(T), t, ch, left=0, right=0)
    al = align(x, fs_rec, tl)
    assert abs(al.offset - true_off) < 0.5
    assert abs(al.ratio - ratio) < 2e-5


def test_wer_normalisation():
    assert wer("Mi número es el diecisiete.", "mi numero es el 17") == 0.0
    assert wer("hola mundo", "") == 1.0
