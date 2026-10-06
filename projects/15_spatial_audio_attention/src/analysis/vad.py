"""Pitch-gated VAD, ported from home2 ``hri/packages/speech/scripts/voice_detection.py``.

home2 cascades (1) an RMS energy gate, (2) an autocorrelation periodicity test
inside the vocal F0 band 85-255 Hz, (3) an MFCC speaker lock. Here we keep
(1) and (2), additionally *return* the F0 (the "Hz of the voice"), and make the
energy gate adaptive to a noise floor learned from silence frames, because the
raw ReSpeaker channels are not AGC'd and their level differs from ch0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INT16_SCALE = 32768.0
VOWEL_FREQ_LOW = 85.0
VOWEL_FREQ_HIGH = 255.0


@dataclass
class VadResult:
    is_speech: bool
    f0_hz: float  # 0.0 when no pitch was found
    rms: float  # same scale as the input
    periodicity: float  # normalised autocorrelation peak in the F0 lag band
    reason: str


def autocorr_pitch(
    x: np.ndarray, fs: int, f0_range=(VOWEL_FREQ_LOW, VOWEL_FREQ_HIGH)
) -> tuple[float, float]:
    """Return (f0_hz, periodicity) from the autocorrelation peak in the F0 band.

    Same statistic as home2 ``_is_periodic`` (peak / r[0]), computed via FFT and
    with parabolic interpolation of the lag for a sub-sample F0.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(np.abs(spec) ** 2)[:n]
    if r[0] <= 0:
        return 0.0, 0.0
    lo = max(1, int(fs / f0_range[1]))
    hi = min(n - 2, int(fs / f0_range[0]))
    if hi <= lo:
        return 0.0, 0.0
    k = lo + int(np.argmax(r[lo:hi]))
    periodicity = float(r[k] / r[0])
    a, b, c = r[k - 1], r[k], r[k + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom < 0 else 0.0
    return float(fs / (k + shift)), periodicity


def frame_vad(
    x: np.ndarray,
    fs: int,
    energy_threshold: float,
    corr_threshold: float,
    f0_range=(VOWEL_FREQ_LOW, VOWEL_FREQ_HIGH),
) -> VadResult:
    """Energy gate + periodicity test on one mono window (float, any scale)."""
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms < energy_threshold:
        return VadResult(False, 0.0, rms, 0.0, "silence")
    f0, per = autocorr_pitch(x, fs, f0_range)
    if per <= corr_threshold:
        return VadResult(False, f0, rms, per, "non-vocal")
    return VadResult(True, f0, rms, per, "voiced")


def update_floor(floor: float | None, rms: float, up: float, down: float) -> float:
    """Asymmetric noise-floor follower: drops fast, rises slowly.

    Only fed with non-speech frames, so it tracks the "sound of silence" of
    the room (fans, motors) that the energy gate should sit above.
    """
    if floor is None or not np.isfinite(floor):
        return rms
    k = down if rms < floor else up
    return (1 - k) * floor + k * rms


def adaptive_threshold(floor: float | None, abs_threshold: float, snr_gate_db: float):
    if floor is None:
        return abs_threshold
    return max(abs_threshold, floor * 10 ** (snr_gate_db / 20.0))
