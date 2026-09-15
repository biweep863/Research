"""Two-template separation for the anomalous case: ball and player pulses that
overlap inside one window.

We model the baseline-removed |a| as

    y(t) = A * T_ball(t - t1) + B * T_player(t - t2) + noise,   A, B >= 0

with clean templates ``T_ball``, ``T_player`` (from ``clean_templates``), and
search (t1, t2) on a grid, solving for the amplitudes by non-negative least
squares at each grid point (Lawson & Hanson 1974).  For two columns NNLS has
a closed form -- the unconstrained 2x2 solution if it is non-negative, else
the better of the two clamped single-column fits -- so the whole (t1, t2)
grid is evaluated with vectorised Gram matrices rather than a Python loop.
The best residual gives the two onsets, hence the signed gap t2 - t1 and the
order.  A model-selection step compares the two-pulse fit with the best
single-pulse fits (BIC-like penalty) so that a window holding only one
contact is not forced into two.
"""
from __future__ import annotations

import numpy as np


def _shift_matrix(tpl: np.ndarray, ks: np.ndarray, n: int) -> np.ndarray:
    """Columns: template shifted to start at each lag in ``ks`` (n x len(ks))."""
    M = np.zeros((n, len(ks)))
    L = len(tpl)
    for j, k in enumerate(ks):
        m = min(n - k, L)
        if m > 0:
            M[k:k + m, j] = tpl[:m]
    return M


def separate(w: np.ndarray, fs: float, templates: dict[str, np.ndarray],
             step_ms: float = 1.0, search_ms: float = 60.0, penalty: float = 2.0) -> dict:
    """Return dict(t_ball, t_player, gap_ms, order, model) in window seconds.
    ``model`` is one of 'ball', 'player', 'both'."""
    mag = np.linalg.norm(w[:, :3].astype(np.float64), axis=1)
    n = len(mag)
    n_bg = max(8, n // 4)
    base = np.median(mag[:n_bg])
    y = mag - base
    peak = int(np.argmax(y))
    sig2 = (1.4826 * np.median(np.abs(y[:n_bg])) + 1e-3) ** 2

    step = max(1, int(round(step_ms * fs / 1000.0)))
    span = int(round(search_ms * fs / 1000.0))
    ks = np.arange(peak - span, peak + span // 2, step)
    ks = ks[(ks >= 0) & (ks < n)]
    if len(ks) == 0:
        ks = np.array([peak])
    Mb = _shift_matrix(templates["ball"], ks, n)
    Mp = _shift_matrix(templates["player"], ks, n)

    yy = float(y @ y)
    bb = np.sum(Mb * Mb, axis=0)            # (K,)
    pp = np.sum(Mp * Mp, axis=0)
    by = Mb.T @ y                           # (K,)
    py = Mp.T @ y
    bp = Mb.T @ Mp                          # (K, K): cross-Gram for (k1, k2)

    # Single-pulse fits (amplitude clamped at 0).
    amp_b = np.maximum(by / (bb + 1e-12), 0.0)
    amp_p = np.maximum(py / (pp + 1e-12), 0.0)
    res_b = yy - 2 * amp_b * by + amp_b ** 2 * bb
    res_p = yy - 2 * amp_p * py + amp_p ** 2 * pp
    ib, ip = int(np.argmin(res_b)), int(np.argmin(res_p))
    r_b, r_p = float(res_b[ib]), float(res_p[ip])

    # Two-pulse fit on the (k1, k2) grid: closed-form 2x2 NNLS.
    B = bb[:, None]; P = pp[None, :]; C = bp
    BY = by[:, None]; PY = py[None, :]
    det = B * P - C * C
    A_un = (P * BY - C * PY) / (det + 1e-12)
    B_un = (B * PY - C * BY) / (det + 1e-12)
    both_pos = (A_un > 0) & (B_un > 0) & (det > 1e-12)
    res_two = yy - 2 * A_un * BY - 2 * B_un * PY + A_un ** 2 * B + B_un ** 2 * P + 2 * A_un * B_un * C
    res_two = np.where(both_pos, res_two, np.inf)
    if np.isfinite(res_two).any():
        k1i, k2i = np.unravel_index(int(np.argmin(res_two)), res_two.shape)
        r2 = float(res_two[k1i, k2i])
        k1, k2 = int(ks[k1i]), int(ks[k2i])
    else:
        r2, k1, k2 = np.inf, None, None

    # Model selection: 2 extra parameters (amp + onset) must pay for themselves.
    cost1 = min(r_b, r_p) / sig2
    cost2 = (r2 / sig2 + penalty * 2 * np.log(n)) if k1 is not None else np.inf
    if cost2 < cost1:
        return {"model": "both", "t_ball": k1 / fs, "t_player": k2 / fs,
                "gap_ms": 1000.0 * (k2 - k1) / fs,
                "order": "ball_first" if k1 <= k2 else "player_first",
                "t_first": min(k1, k2) / fs}
    if r_b <= r_p:
        return {"model": "ball", "t_ball": ks[ib] / fs, "t_player": None, "gap_ms": None,
                "order": "ball_first", "t_first": ks[ib] / fs}
    return {"model": "player", "t_ball": None, "t_player": ks[ip] / fs, "gap_ms": None,
            "order": "player_first", "t_first": ks[ip] / fs}
