"""Solar-resource model: from the solar constant to surface irradiance.

Everything here is first-principles physics with published constants, so the
numbers are computable ground truth (no external datasets). Two layers:

1. **Geometry** — the standard astronomical formulas for the daily
   extraterrestrial (top-of-atmosphere, TOA) insolation on a horizontal
   surface as a function of latitude and day of year. Closed form, checked in
   the tests against direct numerical integration of the instantaneous
   sun-angle.

2. **Atmospheric cascade** — the ordered loss terms that take the global-mean
   TOA irradiance (S0/4) down to the global-mean surface irradiance, with the
   ozone absorption broken out as its own term (the user asked specifically
   about the ozone layer).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- Physical constants (published) --------------------------------------- #
S0 = 1361.0            # solar constant, W/m^2 (TSI, Kopp & Lean 2011)
R_EARTH = 6.371e6      # mean Earth radius, m
SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.0
TILT_DEG = 23.45       # obliquity of the ecliptic (deg)


def toa_intercepted_power() -> float:
    """Total solar power intercepted by Earth's disk (W). ~1.74e17 W."""
    return S0 * math.pi * R_EARTH ** 2


def earth_surface_area() -> float:
    """Total Earth surface area (m^2). ~5.10e14."""
    return 4.0 * math.pi * R_EARTH ** 2


def solar_declination(doy: float) -> float:
    """Solar declination angle (radians) for day-of-year ``doy`` in [1,365]."""
    return math.radians(TILT_DEG) * math.sin(2.0 * math.pi * (284.0 + doy) / DAYS_PER_YEAR)


def eccentricity_factor(doy: float) -> float:
    """Earth-Sun distance correction (dimensionless), ~1 +/- 0.033."""
    return 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / DAYS_PER_YEAR)


def sunset_hour_angle(lat_deg: float, dec: float) -> float:
    """Sunset hour angle (radians). 0 at polar night, pi at midnight sun."""
    lat = math.radians(lat_deg)
    x = -math.tan(lat) * math.tan(dec)
    if x >= 1.0:
        return 0.0        # sun never rises (polar night)
    if x <= -1.0:
        return math.pi    # sun never sets (midnight sun)
    return math.acos(x)


def daily_toa_insolation(lat_deg: float, doy: float) -> float:
    """Daily extraterrestrial insolation on a horizontal surface (J/m^2/day).

    Closed form (Duffie & Beckman, *Solar Engineering of Thermal Processes*).
    """
    lat = math.radians(lat_deg)
    dec = solar_declination(doy)
    e0 = eccentricity_factor(doy)
    ws = sunset_hour_angle(lat_deg, dec)
    bracket = (math.cos(lat) * math.cos(dec) * math.sin(ws)
               + ws * math.sin(lat) * math.sin(dec))
    return (SECONDS_PER_DAY / math.pi) * S0 * e0 * bracket


def daily_toa_mean_irradiance(lat_deg: float, doy: float) -> float:
    """Day-averaged TOA irradiance on a horizontal surface (W/m^2)."""
    return daily_toa_insolation(lat_deg, doy) / SECONDS_PER_DAY


def instantaneous_toa_irradiance(lat_deg: float, doy: float, hour_angle: float) -> float:
    """Instantaneous TOA irradiance on a horizontal surface (W/m^2).

    ``hour_angle`` in radians (0 = solar noon). Returns 0 at night.
    """
    lat = math.radians(lat_deg)
    dec = solar_declination(doy)
    e0 = eccentricity_factor(doy)
    cos_zenith = (math.sin(lat) * math.sin(dec)
                  + math.cos(lat) * math.cos(dec) * math.cos(hour_angle))
    return S0 * e0 * max(cos_zenith, 0.0)


def daily_toa_insolation_numeric(lat_deg: float, doy: float, n_steps: int = 4000) -> float:
    """Numerically integrate the instantaneous irradiance over a full day.

    Used only to validate :func:`daily_toa_insolation` in the tests.
    """
    ws = np.linspace(-math.pi, math.pi, n_steps)
    vals = np.array([instantaneous_toa_irradiance(lat_deg, doy, w) for w in ws])
    dt = SECONDS_PER_DAY * (ws[1] - ws[0]) / (2.0 * math.pi)
    return float(np.sum(vals) * dt)


def global_annual_mean_toa(n_lat: int = 180, n_doy: int = 73) -> float:
    """Area- and time-weighted global-mean TOA horizontal irradiance (W/m^2).

    Should equal S0/4 ~ 340.25 W/m^2 (checked in tests).
    """
    lats = np.linspace(-89.5, 89.5, n_lat)
    doys = np.linspace(1, 365, n_doy)
    weights = np.cos(np.radians(lats))
    band_means = np.array([
        np.mean([daily_toa_mean_irradiance(float(lat), float(d)) for d in doys])
        for lat in lats
    ])
    return float(np.sum(band_means * weights) / np.sum(weights))


# --- Atmospheric loss cascade --------------------------------------------- #
# Fractions of the global-mean TOA horizontal irradiance (S0/4 ~ 340 W/m^2).
# Values from standard Earth energy-budget accounting (Trenberth et al. 2009);
# ozone is separated from the rest of atmospheric absorption because the user
# asked about the ozone layer specifically. Panels receive the surface global
# horizontal irradiance (GHI) = TOA minus what the atmosphere reflects and
# absorbs (ground albedo does not reduce what reaches the panel plane).
CLOUD_ATM_REFLECTION = 0.23   # reflected back to space by clouds + aerosols
OZONE_ABSORPTION = 0.02       # absorbed by stratospheric ozone (mostly UV)
WATER_AEROSOL_ABSORPTION = 0.21  # water vapour, aerosols, cloud absorption
SURFACE_TRANSMITTANCE = 1.0 - CLOUD_ATM_REFLECTION - OZONE_ABSORPTION - WATER_AEROSOL_ABSORPTION


@dataclass(frozen=True)
class CascadeLevel:
    label: str
    irradiance: float   # W/m^2 remaining at this stage


def loss_cascade() -> list[CascadeLevel]:
    """Ordered waterfall from the solar constant to surface GHI (W/m^2)."""
    toa = S0 / 4.0
    surface = toa * SURFACE_TRANSMITTANCE
    return [
        CascadeLevel("Solar constant (noon, equator)", S0),
        CascadeLevel("TOA horizontal avg (geometry + day/night)", toa),
        CascadeLevel("after cloud/aerosol reflection", toa * (1 - CLOUD_ATM_REFLECTION)),
        CascadeLevel("after ozone absorption", toa * (1 - CLOUD_ATM_REFLECTION - OZONE_ABSORPTION)),
        CascadeLevel("surface GHI (after water/aerosol absorption)", surface),
    ]


def global_mean_surface_ghi() -> float:
    """Global-mean surface global horizontal irradiance (W/m^2). ~184."""
    return (S0 / 4.0) * SURFACE_TRANSMITTANCE


# Band transmittance profile: heavy clouds at the equator, clear subtropical
# desert belt, thinner but low-angle light toward the poles. Averages to
# roughly SURFACE_TRANSMITTANCE. Keyed by absolute latitude band.
def band_transmittance(abs_lat_deg: float) -> float:
    a = abs(abs_lat_deg)
    if a < 15:
        return 0.45   # equatorial cloud belt
    if a < 35:
        return 0.62   # subtropical desert belt (clearest skies)
    if a < 55:
        return 0.50   # temperate
    return 0.42       # high latitude


def band_surface_ghi(abs_lat_deg: float, n_doy: int = 24) -> float:
    """Annual-mean surface GHI for a latitude band (W/m^2)."""
    doys = np.linspace(1, 365, n_doy)
    toa = float(np.mean([daily_toa_mean_irradiance(abs_lat_deg, float(d)) for d in doys]))
    return toa * band_transmittance(abs_lat_deg)


def diurnal_light_profile(lat_deg: float, doy: float, n_hours: int = 24) -> np.ndarray:
    """Hourly surface irradiance over one representative day (W/m^2).

    Clear-sky TOA shape scaled by the band transmittance; used as the light
    supply ``L[h]`` in the ecosystem light-budget optimisation.
    """
    hours = np.linspace(0, 24, n_hours, endpoint=False)
    hour_angles = np.radians((hours - 12.0) * 15.0)  # 15 deg per hour
    toa = np.array([instantaneous_toa_irradiance(lat_deg, doy, float(w)) for w in hour_angles])
    return toa * band_transmittance(lat_deg)
