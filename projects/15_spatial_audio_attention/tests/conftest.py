import numpy as np
import pytest
import yaml
from pathlib import Path

from src.analysis.geometry import SPEED_OF_SOUND, far_field_delays, respeaker_v2_positions

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return yaml.safe_load((ROOT / "configs" / "local_small.yaml").read_text())


def plane_wave(signal: np.ndarray, az_deg: float, fs: int = 16000, radius: float = 0.032):
    """Exact fractional delays of a far-field source on the ReSpeaker geometry."""
    mics = respeaker_v2_positions(radius)
    tau = far_field_delays(mics, np.deg2rad([az_deg]), SPEED_OF_SOUND)[0]
    n = len(signal)
    S = np.fft.rfft(signal)
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.stack([np.fft.irfft(S * np.exp(-2j * np.pi * f * t), n) for t in tau])


def harmonic(f0: float, dur: float = 1.0, fs: int = 16000, n_harm: int = 20, amp: float = 0.1):
    t = np.arange(int(dur * fs)) / fs
    x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_harm + 1) if k * f0 < fs / 2)
    return amp * x / np.std(x)
