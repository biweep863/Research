"""Propulsion -> per-hop cruise speed -> per-hop range.

The bridge between engineering and the star graph: a technology fixes the
cruise speed a ship can reach *and stop from* on one tank (mass ratio R),
and a hardware/crew lifetime tau_max per leg fixes how far one hop can go:

    r_hop = v_cruise * tau_max.

Rockets: relativistic rocket equation, half the rapidity budget to
accelerate and half to brake (rendezvous, so the colony can refuel and
relaunch).  Laser sails: launched by a beam from the departure star, but
braking needs the *target* star's light (photogravitational braking,
Heller & Hippke 2017), so the usable speed depends on the target luminosity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.model.kinematics import C_KMS, KMS_TO_PCMYR

YR_PER_MYR = 1.0e6


@dataclass(frozen=True)
class Rocket:
    name: str
    v_exhaust_kms: float


# Exhaust velocities: chemical LOX/LH2 ~4.4 km/s; nuclear-thermal ~9 km/s;
# electric/ion (high-Isp, e.g. VASIMR upper range) ~100 km/s; Daedalus-class
# inertial fusion ~10,000 km/s; idealised beamed-core antimatter ~0.33 c.
ROCKETS = [
    Rocket("chemical", 4.4),
    Rocket("nuclear thermal", 9.0),
    Rocket("electric", 100.0),
    Rocket("fusion", 10_000.0),
    Rocket("antimatter", 0.33 * C_KMS),
]


def rocket_cruise_speed_kms(v_exhaust_kms, mass_ratio):
    """Cruise speed (km/s) when the mass ratio pays for accelerate + brake.

    Relativistic rocket: rapidity phi = (v_e/c) ln R; half of it is spent
    on each burn, so v = c tanh(v_e ln R / 2c)."""
    phi_half = 0.5 * np.asarray(v_exhaust_kms) / C_KMS * np.log(mass_ratio)
    return C_KMS * np.tanh(phi_half)


def sail_brake_speed_kms(lum_lsun, v_ref_kms=13_800.0, l_ref=1.52, exponent=0.4):
    """Highest arrival speed a photon sail can shed at a star of luminosity L.

    Energy balance at closest approach r_min: v^2 ~ L / (sigma c r_min), with
    r_min proportional to the stellar radius R ~ L^0.2 on the main sequence,
    gives v ~ L^0.4. Calibrated to Heller, Hippke & Kervella (2017):
    13,800 km/s at alpha Cen A (1.52 L_sun) for a graphene-class sail."""
    return v_ref_kms * (np.asarray(lum_lsun) / l_ref) ** exponent


def hop_range_pc(v_kms, tau_max_yr):
    """Distance covered in tau_max years at v (pc)."""
    return np.asarray(v_kms) * KMS_TO_PCMYR * (np.asarray(tau_max_yr) / YR_PER_MYR)


def brachistochrone_times_yr(dist_pc, accel_g=1.0):
    """Constant-proper-acceleration flip-and-burn trip (relativistic).

    Returns (coordinate time, proper/crew time) in years."""
    g = 9.80665
    c = C_KMS * 1e3
    pc_m = 3.0856775814913673e16
    yr_s = 365.25 * 86400
    a = accel_g * g
    half = 0.5 * np.asarray(dist_pc) * pc_m
    tau_half = (c / a) * np.arccosh(1.0 + a * half / c ** 2)
    t_half = (c / a) * np.sinh(a * tau_half / c)
    return 2 * t_half / yr_s, 2 * tau_half / yr_s
