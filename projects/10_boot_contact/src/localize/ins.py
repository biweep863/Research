"""Where was the contact?  Foot-mounted inertial navigation with zero-velocity
updates (ZUPT, Foxlin 2005) and sparse UWB position fixes.

Simulation: a running foot in the world frame (x forward, z up).  Each stride
the foot is planted (stance: zero velocity for ~0.12 s) and then swung forward
along a smooth arc.  The IMU measures specific force and angular rate in the
foot frame; the foot pitches during swing so orientation must be integrated
from the gyro, and gyro bias leaks gravity into the horizontal acceleration --
the classic INS drift.

Estimator: a 9-state error-state Kalman filter is overkill for a study; we use
a 6-state linear KF on [p, v] driven by the world-frame acceleration obtained
from the integrated orientation, with two measurement types:
    ZUPT  : v = 0 during a detected stance (|gyro| small and |a| ~ 1 g),
    UWB   : p = anchor fix + N(0, sigma_uwb), at `uwb_rate` Hz.
The metric is the position error at the (known) contact instant, over many
runs.  A rule-on-a-line decision (was the contact inside the penalty area?) is
then a Gaussian tail: P(wrong side) = Phi(-d / sigma_pos).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

G = 9.80665


def true_trajectory(duration_s: float, fs: float, speed: float, rng: np.random.Generator):
    """Return t, p (T,3), v, a (world), pitch (rad), pitch_rate, stance mask."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    stride_len = speed / 2.8                     # ~2.8 strides/s at 6 m/s
    period = stride_len / speed
    stance_frac = 0.35
    p = np.zeros((n, 3)); v = np.zeros((n, 3)); pitch = np.zeros(n); stance = np.zeros(n, bool)
    x0 = 0.0
    k = 0
    for i, ti in enumerate(t):
        ph = (ti % period) / period
        kk = int(ti // period)
        if kk != k:
            x0 += stride_len; k = kk
        if ph < stance_frac:
            p[i] = [x0, 0.0, 0.0]; stance[i] = True
        else:
            u = (ph - stance_frac) / (1 - stance_frac)      # 0..1 in swing
            s = u - np.sin(2 * np.pi * u) / (2 * np.pi)     # smooth min-jerk-ish
            p[i] = [x0 + stride_len * s, 0.02 * np.sin(np.pi * u), 0.15 * np.sin(np.pi * u)]
            pitch[i] = np.deg2rad(35.0) * np.sin(np.pi * u) ** 2
    # A lateral cut mid-run so the problem is genuinely 3-D.
    yaw_rate = np.where((t > 0.45 * duration_s) & (t < 0.55 * duration_s), np.deg2rad(60) / (0.1 * duration_s), 0.0)
    yaw = np.cumsum(yaw_rate) / fs
    # Rotate the planar path by yaw (incrementally), keep z.
    dp = np.diff(p, axis=0, prepend=p[:1])
    c, s_ = np.cos(yaw), np.sin(yaw)
    dpx = c * dp[:, 0] - s_ * dp[:, 1]; dpy = s_ * dp[:, 0] + c * dp[:, 1]
    p = np.stack([np.cumsum(dpx), np.cumsum(dpy), p[:, 2]], axis=1)
    v = np.gradient(p, 1.0 / fs, axis=0)
    a = np.gradient(v, 1.0 / fs, axis=0)
    v[stance] = 0.0
    return t, p, v, a, pitch, yaw, stance


def rot_y(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def imu_measurements(t, a_world, pitch, yaw, fs, grade: dict, rng: np.random.Generator):
    """Body-frame specific force (m/s^2) and angular rate (rad/s) with noise
    and a constant gyro bias."""
    n = len(t)
    f_b = np.zeros((n, 3)); w_b = np.zeros((n, 3))
    pr = np.gradient(pitch, 1.0 / fs); yr = np.gradient(yaw, 1.0 / fs)
    for i in range(n):
        R = rot_z(yaw[i]) @ rot_y(pitch[i])             # body -> world
        f_b[i] = R.T @ (a_world[i] + np.array([0, 0, G]))
        w_b[i] = [0.0, pr[i], yr[i]]
    k = np.sqrt(fs / 1000.0)
    f_b += rng.normal(0, grade["acc_noise_g"] * G * k, f_b.shape)
    bias = np.deg2rad(rng.normal(0, grade["gyr_bias_dps"], 3))
    w_b += bias + rng.normal(0, np.deg2rad(grade["gyr_noise_dps"]) * k, w_b.shape)
    return f_b, w_b


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def run_ins(t, f_b, w_b, fs, uwb_rate: float, uwb_sigma: float, p_true, rng,
            zupt: bool = True, R0=None):
    """Kalman INS with ZUPT and UWB.  Returns estimated positions (T,3)."""
    n = len(t); dt = 1.0 / fs
    R = np.eye(3) if R0 is None else R0.copy()
    x = np.zeros(6)                                  # [p, v]
    P = np.diag([0.01] * 3 + [0.01] * 3)
    F = np.eye(6); F[:3, 3:] = dt * np.eye(3)
    Q = np.diag([1e-4] * 3 + [0.5 * dt] * 3)
    Hz = np.zeros((3, 6)); Hz[:, 3:] = np.eye(3)
    Hp = np.zeros((3, 6)); Hp[:, :3] = np.eye(3)
    Rz = np.eye(3) * 0.01 ** 2
    Ru = np.eye(3) * uwb_sigma ** 2
    est = np.zeros((n, 3))
    uwb_every = int(round(fs / uwb_rate)) if uwb_rate > 0 else None
    gyr_mag = np.linalg.norm(w_b, axis=1)
    acc_mag = np.linalg.norm(f_b, axis=1)
    for i in range(n):
        # Orientation from the gyro (first-order).
        R = R @ (np.eye(3) + _skew(w_b[i] * dt))
        u, _, vt = np.linalg.svd(R); R = u @ vt        # re-orthonormalise
        a_w = R @ f_b[i] - np.array([0, 0, G])
        # Predict.
        x = F @ x; x[3:] += a_w * dt; x[:3] += 0.5 * a_w * dt * dt
        P = F @ P @ F.T + Q
        # ZUPT when the foot looks planted.
        if zupt and gyr_mag[i] < np.deg2rad(40) and abs(acc_mag[i] - G) < 3.0:
            K = P @ Hz.T @ np.linalg.inv(Hz @ P @ Hz.T + Rz)
            x = x + K @ (np.zeros(3) - Hz @ x); P = (np.eye(6) - K @ Hz) @ P
        if uwb_every and i % uwb_every == 0:
            z = p_true[i] + rng.normal(0, uwb_sigma, 3)
            K = P @ Hp.T @ np.linalg.inv(Hp @ P @ Hp.T + Ru)
            x = x + K @ (z - Hp @ x); P = (np.eye(6) - K @ Hp) @ P
        est[i] = x[:3]
    return est


def wrong_side_probability(d_m: float | np.ndarray, sigma_m: float) -> np.ndarray:
    """P(a contact at distance d from a line is placed on the wrong side)."""
    return norm.cdf(-np.asarray(d_m) / max(sigma_m, 1e-9))
