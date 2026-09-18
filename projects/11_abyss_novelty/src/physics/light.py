"""Ocean optics: how far light gets, one way and two ways.

Beer-Lambert: E(z) = E0 exp(-Kd z), with Kd the diffuse attenuation
coefficient of Jerlov water types (Jerlov 1976; Solonenko & Mobley 2015).
A satellite or lidar sees light that went down *and back up*, so the depth it
can probe is where the round-trip attenuation hits the detector floor.
"""
from __future__ import annotations

import numpy as np


def irradiance(z_m, e0: float, kd: float) -> np.ndarray:
    return e0 * np.exp(-kd * np.asarray(z_m, dtype=float))


def reach_depth(e0: float, floor: float, kd: float, two_way: bool = False) -> float:
    """Depth at which E drops to ``floor`` (halved for a round trip)."""
    z = np.log(e0 / floor) / kd
    return float(z / 2.0 if two_way else z)


def euphotic_depth(kd: float) -> float:
    """1 % light level, the classic photosynthesis floor."""
    return float(np.log(100.0) / kd)


def bioluminescence_range(photons: float, threshold: float, aperture_m2: float, kd: float) -> float:
    """Distance at which a flash of ``photons`` still delivers ``threshold``
    photons into a camera of area ``aperture_m2``: inverse-square spreading
    times exponential attenuation, solved numerically."""
    r = np.logspace(-2, 4, 4000)
    received = photons * aperture_m2 / (4 * np.pi * r ** 2) * np.exp(-kd * r)
    ok = received >= threshold
    return float(r[ok].max()) if ok.any() else 0.0
