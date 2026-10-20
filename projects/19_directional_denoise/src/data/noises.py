"""Interferers played through the PC speaker, and the alignment markers.

All generators return float64 at ``fs`` with unit RMS (the session scales them
to the calibrated playback level). Real recordings come from the corpora
project 16 already downloaded (DEMAND cafeteria, LibriSpeech dev-clean); the
rest are synthetic so the session also runs without them.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import scipy.signal

P16_RAW = Path(__file__).resolve().parents[3] / "16_causal_binaural_separation" / "data" / "raw"


def _unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return x / (np.sqrt(np.mean(x**2)) + 1e-12)


def _fade(x: np.ndarray, fs: int, s: float = 0.05) -> np.ndarray:
    n = int(s * fs)
    w = np.ones(len(x))
    w[:n] = np.linspace(0, 1, n)
    w[-n:] = np.linspace(1, 0, n)
    return x * w


def pink(n: int, fs: int, rng) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0
    X[f < 60] = 0  # small desktop speakers cannot play it anyway
    return _unit(np.fft.irfft(X, n))


def fan(n: int, fs: int, rng) -> np.ndarray:
    """Broadband airflow (brown-ish, low-passed) + motor hum harmonics + blade-rate AM."""
    t = np.arange(n) / fs
    b, a = scipy.signal.butter(2, [80, 1500], btype="band", fs=fs)
    air = scipy.signal.lfilter(b, a, rng.standard_normal(n))
    hum = sum(np.sin(2 * np.pi * k * 120 * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 6))
    am = 1 + 0.3 * np.sin(2 * np.pi * 18 * t)
    return _unit(_unit(air) * am + 0.5 * _unit(hum))


def music(n: int, fs: int, rng) -> np.ndarray:
    """Chord progression (additive synth, decaying notes) over a kick/hat beat."""
    t = np.arange(n) / fs
    out = np.zeros(n)
    beat = 0.5  # 120 bpm
    roots = [57, 53, 60, 55]  # A F C G (MIDI)
    for k, start in enumerate(np.arange(0, n / fs, 4 * beat)):
        root = roots[k % 4]
        for semis in (0, 4 if k % 4 in (2, 3) else 3, 7, 12):
            f0 = 440 * 2 ** ((root + semis - 69) / 12)
            i0 = int(start * fs)
            tt = t[: min(n - i0, int(4 * beat * fs))]
            note = sum(np.sin(2 * np.pi * h * f0 * tt) / h**1.5 for h in range(1, 6)) * np.exp(-tt / 1.2)
            out[i0 : i0 + len(tt)] += note
    for start in np.arange(0, n / fs, beat):
        i0 = int(start * fs)
        tt = t[: min(n - i0, int(0.15 * fs))]
        out[i0 : i0 + len(tt)] += 3 * np.sin(2 * np.pi * (60 + 80 * np.exp(-tt / 0.02)) * tt) * np.exp(-tt / 0.05)
        h0 = i0 + int(beat / 2 * fs)
        m = min(n - h0, int(0.04 * fs))
        if m > 0:
            out[h0 : h0 + m] += 0.6 * np.diff(rng.standard_normal(m + 1)) * np.exp(-np.arange(m) / (0.01 * fs))
    return _unit(out)


def _resample(x: np.ndarray, fs_in: int, fs: int) -> np.ndarray:
    g = np.gcd(fs_in, fs)
    return scipy.signal.resample_poly(x, fs // g, fs_in // g)


def cafeteria(n: int, fs: int, rng) -> np.ndarray:
    """DEMAND PCAFETER ch01 (real babble + dishes); synthetic babble fallback."""
    z = P16_RAW / "PCAFETER_16k.zip"
    if not z.exists():
        return babble_synth(n, fs, rng)
    import soundfile as sf

    with zipfile.ZipFile(z) as zf:
        name = next(m for m in zf.namelist() if m.endswith("ch01.wav"))
        x, fs_in = sf.read(io.BytesIO(zf.read(name)))
    x = _resample(x, fs_in, fs)
    # offset independent of n, so takes of different length replay the same
    # excerpt (the 2026-10-18 live take did not: doc/01)
    i0 = int(rng.integers(0, max(1, len(x) - 40 * fs)))
    return _unit(np.resize(x[i0:], n))


def talker(n: int, fs: int, rng, speaker: str | None = None) -> np.ndarray:
    """One LibriSpeech dev-clean speaker (a 'TV' / second person), pauses trimmed to 0.3 s."""
    tgz = P16_RAW / "dev-clean.tar.gz"
    if not tgz.exists():
        return babble_synth(n, fs, rng, n_talkers=1)
    import soundfile as sf

    chunks, total = [], 0
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if not m.name.endswith(".flac"):
                continue
            spk = m.name.split("/")[-3]
            if speaker is None:
                speaker = spk
            if spk != speaker:
                if chunks:
                    break
                continue
            x, fs_in = sf.read(io.BytesIO(tf.extractfile(m).read()))
            chunks.append(_resample(x, fs_in, fs))
            chunks.append(np.zeros(int(0.3 * fs)))
            total += len(chunks[-2]) + len(chunks[-1])
            if total >= n:
                break
    return _unit(np.resize(np.concatenate(chunks), n))


def babble_synth(n: int, fs: int, rng, n_talkers: int = 6) -> np.ndarray:
    """Fallback: speech-shaped noise with 4 Hz syllabic envelopes."""
    out = np.zeros(n)
    t = np.arange(n) / fs
    b, a = scipy.signal.butter(2, [150, 4000], btype="band", fs=fs)
    for _ in range(n_talkers):
        env = np.clip(np.sin(2 * np.pi * rng.uniform(3, 5) * t + rng.uniform(0, 6)), 0, None) ** 2
        out += scipy.signal.lfilter(b, a, rng.standard_normal(n)) * env
    return _unit(out)


NOISES = {"pink": pink, "fan": fan, "cafeteria": cafeteria, "talker": talker, "music": music}


def make(kind: str, seconds: float, fs: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _fade(NOISES[kind](int(seconds * fs), fs, rng), fs)


# ------------------------------------------------------------------ markers
def chirp(fs: int, dur: float = 0.25, f0: float = 500.0, f1: float = 4000.0) -> np.ndarray:
    """Linear up-chirp with 5 ms ramps: start/end marker for sample alignment."""
    t = np.arange(int(dur * fs)) / fs
    x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t**2))
    return _fade(x, fs, 0.005)


def beep(fs: int, dur: float = 0.12, f: float = 1000.0) -> np.ndarray:
    """'Talk now' cue (a pure tone, so it is never confused with the chirp)."""
    t = np.arange(int(dur * fs)) / fs
    return _fade(np.sin(2 * np.pi * f * t), fs, 0.01)
