# Vendored from projects/15_spatial_audio_attention/src/analysis/geometry.py (kept identical).
"""Array geometry for the ReSpeaker USB Mic Array v2.0 and far-field steering.

Conventions used everywhere in this project:
- Azimuth is measured in the array plane, 0 deg = +x axis, counter-clockwise.
- A plane wave from azimuth ``az`` reaches mic ``m`` at delay
  ``tau_m = -(p_m . u(az)) / c`` relative to the array centre, so mics that
  face the source hear it first (negative delay).
- STFT model: ``X_m(f) = S(f) * exp(-j 2 pi f tau_m)``.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

SPEED_OF_SOUND = 343.0


def respeaker_v2_positions(radius: float = 0.032) -> np.ndarray:
    """XY positions (m) of the 4 raw mics (firmware channels 1..4).

    Order follows the ODAS ``respeaker_usb_4_mic_array`` config. The absolute
    orientation relative to the robot (and to firmware ``DOAANGLE``) is NOT
    assumed: it is absorbed by ``azimuth_offset_deg`` / ``mirror`` in the
    config and calibrated on hardware (see doc/05_decisions.md).
    """
    r = float(radius)
    return np.array([[-r, 0.0], [0.0, -r], [r, 0.0], [0.0, r]])


def mic_pairs(n_mics: int) -> list[tuple[int, int]]:
    return list(combinations(range(n_mics), 2))


def unit_vectors(az_rad: np.ndarray) -> np.ndarray:
    az = np.atleast_1d(az_rad)
    return np.stack([np.cos(az), np.sin(az)], axis=-1)


def far_field_delays(
    mics: np.ndarray, az_rad: np.ndarray, c: float = SPEED_OF_SOUND
) -> np.ndarray:
    """(n_az, M) arrival delays in seconds relative to the array centre."""
    return -(unit_vectors(az_rad) @ mics.T) / c


def steering_vectors(
    mics: np.ndarray, az_rad: np.ndarray, freqs: np.ndarray, c: float = SPEED_OF_SOUND
) -> np.ndarray:
    """(n_az, F, M) far-field steering vectors exp(-j 2 pi f tau)."""
    tau = far_field_delays(mics, az_rad, c)
    return np.exp(-2j * np.pi * freqs[None, :, None] * tau[:, None, :])


def max_pair_delay(mics: np.ndarray, c: float = SPEED_OF_SOUND) -> float:
    """Largest possible |TDOA| over all pairs (aperture / c)."""
    d = max(np.linalg.norm(mics[i] - mics[j]) for i, j in mic_pairs(len(mics)))
    return d / c


def array_to_robot_deg(az_deg, offset_deg: float = 0.0, mirror: bool = False):
    """Map array-frame azimuth to the robot/display frame (calibrated on HW)."""
    a = -np.asarray(az_deg) if mirror else np.asarray(az_deg)
    return np.mod(a + offset_deg, 360.0)
