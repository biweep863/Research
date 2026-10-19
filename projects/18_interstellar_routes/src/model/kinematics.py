"""Astrometry -> heliocentric Cartesian phase space, and linear propagation.

Units throughout the project: positions in parsec, times in Myr, velocities
in pc/Myr (1 km/s = 1.0227 pc/Myr). The frame is heliocentric ICRS axes;
only *relative* star-star geometry enters the routing, so the choice of
inertial frame does not matter and the Sun is the node at the origin with
zero velocity.

Motion is linear (straight lines at constant velocity). Over <~2 Myr and
inside 100 pc the Galactic potential bends paths by far less than the
inter-star spacing (epicyclic period ~170 Myr), which is the regime used by
Gaia close-encounter studies for their first pass (Bailer-Jones 2018).
"""
from __future__ import annotations

import numpy as np

K_AUYR = 4.740470446          # km/s per (arcsec/yr * pc)
KMS_TO_PCMYR = 1.0227121650537077   # 1 km/s in pc/Myr
C_KMS = 299_792.458
C_PCMYR = C_KMS * KMS_TO_PCMYR
M_G_SUN = 4.67                 # absolute Gaia G magnitude of the Sun


def unit_vectors(ra_deg, dec_deg):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    ca, sa, cd, sd = np.cos(a), np.sin(a), np.cos(d), np.sin(d)
    r_hat = np.stack([cd * ca, cd * sa, sd], axis=-1)
    p_hat = np.stack([-sa, ca, np.zeros_like(a)], axis=-1)       # east (+RA)
    q_hat = np.stack([-sd * ca, -sd * sa, cd], axis=-1)          # north (+Dec)
    return r_hat, p_hat, q_hat


def astrometry_to_cartesian(ra, dec, parallax_mas, pmra_masyr, pmdec_masyr, rv_kms):
    """Positions (pc) and velocities (pc/Myr), shape (N, 3) each.

    `pmra` is Gaia's mu_alpha* (already multiplied by cos dec)."""
    r_hat, p_hat, q_hat = unit_vectors(np.asarray(ra), np.asarray(dec))
    plx = np.asarray(parallax_mas, dtype=float)
    dist = 1000.0 / plx
    vt_ra = K_AUYR * np.asarray(pmra_masyr) / plx           # km/s
    vt_dec = K_AUYR * np.asarray(pmdec_masyr) / plx
    rv = np.asarray(rv_kms, dtype=float)
    pos = dist[:, None] * r_hat
    vel = (rv[:, None] * r_hat + vt_ra[:, None] * p_hat + vt_dec[:, None] * q_hat)
    return pos, vel * KMS_TO_PCMYR


def fill_missing_rv(ra, dec, parallax_mas, pmra, pmdec, rv, has_rv, rng, v_fit_max_kms=200.0):
    """Draw each missing RV from the local velocity ellipsoid *conditioned on
    the star's measured tangential velocity*.

    A Gaussian N(mu, Sigma) is fitted to the 3D heliocentric velocities of the
    stars with RVs (speed < v_fit_max_kms, to keep halo stars from inflating
    Sigma). For a star with radial unit vector a and tangential basis M = [p; q]
    the radial component is then Gaussian with
        mean = a.mu + a S M^T (M S M^T)^-1 (t - M mu),
        var  = a S a - a S M^T (M S M^T)^-1 M S a.
    Drawing RVs unconditionally instead breaks the coherence of the velocity
    field and makes faint stars fall toward the Sun (the 15-pc star count
    tripled within 1 Myr in testing)."""
    rv = np.array(rv, dtype=float)
    miss = ~np.asarray(has_rv)
    if not miss.any():
        return rv
    plx = np.asarray(parallax_mas, dtype=float)
    r_hat, p_hat, q_hat = unit_vectors(np.asarray(ra), np.asarray(dec))
    vt_ra = K_AUYR * np.asarray(pmra) / plx
    vt_dec = K_AUYR * np.asarray(pmdec) / plx
    v3 = rv[:, None] * r_hat + vt_ra[:, None] * p_hat + vt_dec[:, None] * q_hat
    fit = ~miss & (np.linalg.norm(np.nan_to_num(v3, nan=1e9), axis=1) < v_fit_max_kms)
    mu = v3[fit].mean(0)
    S = np.cov(v3[fit].T)
    a = r_hat[miss]
    M = np.stack([p_hat[miss], q_hat[miss]], axis=1)                 # (m, 2, 3)
    t = np.stack([vt_ra[miss], vt_dec[miss]], axis=1)                # (m, 2)
    aS = a @ S                                                        # (m, 3)
    MS = np.einsum("mij,jk->mik", M, S)                               # (m, 2, 3)
    MSM = np.einsum("mik,mjk->mij", MS, M)                            # (m, 2, 2)
    aSM = np.einsum("mk,mjk->mj", aS, M)                              # (m, 2)
    K = np.linalg.solve(MSM, aSM[..., None])[..., 0]                  # (m, 2)
    mean = a @ mu + np.einsum("mj,mj->m", K, t - np.einsum("mij,j->mi", M, mu))
    var = np.einsum("mk,mk->m", aS, a) - np.einsum("mj,mj->m", K, aSM)
    rv[miss] = mean + np.sqrt(np.maximum(var, 0.0)) * rng.normal(size=int(miss.sum()))
    return rv


def phase_space(cat: dict, rng, perturb: bool = False):
    """Catalogue -> (pos, vel) with the Sun prepended as node 0.

    perturb=True resamples parallax / proper motion / RV from their Gaia
    uncertainties (Gaussian) -- one astrometric bootstrap draw."""
    plx = cat["parallax"].copy()
    pmra, pmdec = cat["pmra"].copy(), cat["pmdec"].copy()
    if perturb:
        plx = plx + rng.normal(size=plx.size) * cat["parallax_error"]
        plx = np.maximum(plx, 0.5 * cat["parallax"])        # poe>=10 -> never binds in practice
        pmra = pmra + rng.normal(size=plx.size) * np.nan_to_num(cat["pmra_error"])
        pmdec = pmdec + rng.normal(size=plx.size) * np.nan_to_num(cat["pmdec_error"])
        rv_err = np.where(cat["has_rv"], np.nan_to_num(cat["radial_velocity_error"]), 0.0)
        rv_obs = cat["radial_velocity"] + rng.normal(size=plx.size) * rv_err
    else:
        rv_obs = cat["radial_velocity"]
    rv = fill_missing_rv(cat["ra"], cat["dec"], plx, pmra, pmdec, rv_obs, cat["has_rv"], rng)
    pos, vel = astrometry_to_cartesian(cat["ra"], cat["dec"], plx, pmra, pmdec, rv)
    pos = np.vstack([np.zeros((1, 3)), pos])
    vel = np.vstack([np.zeros((1, 3)), vel])
    return pos, vel


def propagate(pos, vel, t_myr):
    return pos + vel * t_myr


def closest_approach(rel_pos, rel_vel):
    """Time (Myr) and distance (pc) of closest approach for linear relative motion."""
    rel_pos, rel_vel = np.atleast_2d(rel_pos), np.atleast_2d(rel_vel)
    w2 = np.einsum("ij,ij->i", rel_vel, rel_vel)
    t = -np.einsum("ij,ij->i", rel_pos, rel_vel) / np.where(w2 > 0, w2, np.inf)
    d = np.linalg.norm(rel_pos + rel_vel * t[:, None], axis=1)
    return t, d


def luminosity_lsun(g_mag, parallax_mas):
    """Rough luminosity from Gaia G with zero bolometric correction."""
    m_g = np.asarray(g_mag) + 5.0 * np.log10(np.asarray(parallax_mas) / 100.0)
    return 10.0 ** (-0.4 * (m_g - M_G_SUN))
