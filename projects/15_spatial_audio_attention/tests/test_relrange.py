from src.analysis.relrange import RelativeRange


def _utterance(rr, level, snr=15.0, n=10):
    for _ in range(n):
        band, _ = rr.update(level, snr)
    rr.end_utterance()
    return band


def test_static_talker_stays_mid():
    rr = RelativeRange(prior_speech_db_1m=None)
    bands = [_utterance(rr, -40.0 + 0.5 * (k % 3)) for k in range(10)]
    assert set(bands) == {"mid"}


def test_learns_near_and_far_relative_to_history():
    rr = RelativeRange(prior_speech_db_1m=None)
    for k in range(6):
        _utterance(rr, -40.0 + (k % 2))
    assert _utterance(rr, -32.0) == "near"   # +8 dB ~ 2.5x closer
    assert _utterance(rr, -48.0) == "far"


def test_prior_used_before_history_and_washed_out():
    rr = RelativeRange(prior_speech_db_1m=-44.0)
    assert _utterance(rr, -36.0) == "near"   # louder than normal speech at 1 m
    for _ in range(8):
        _utterance(rr, -36.0)                # the talker just stays there
    assert _utterance(rr, -36.0) == "mid"


def test_low_snr_frames_ignored():
    rr = RelativeRange(prior_speech_db_1m=-44.0)
    band, _ = rr.update(-20.0, snr_db=1.0)
    assert band is None and rr.cur == []
