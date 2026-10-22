"""Time-dependent star graph: stars move, ships may wait for a close pass.

Every star moves on a straight line, so the distance between stars i and j is

    d_ij(t) = |p_ij + w_ij t|,   p_ij = x_j(0) - x_i(0),  w_ij = v_j - v_i,

and d_ij(t)^2 is a convex quadratic in t. A hop of range r is therefore open
on exactly one time interval [a_ij, b_ij] (possibly empty), computable in
closed form. Earliest-arrival routing on such single-window edges satisfies
FIFO when waiting is allowed (leaving later never arrives earlier), so a
Dijkstra over arrival times is exact.

Hop feasibility. A ship flying at speed v (in the departure star's frame)
toward a target with relative velocity w flies the intercept distance
L = v * tau, with tau solving |p + w tau| = v tau. We open the edge when
d(t) <= r (1 - |w|/v): this guarantees L <= r (since L <= d + |w| tau and
tau <= r / v) and is within |w|/v ~ 1e-3 of exact for v = 0.1 c.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree


def candidate_pairs(pos, vel, r_max, t_max, v_cap=None, n_slices=None):
    """All unordered pairs whose separation drops to <= r_max at some t in
    [0, t_max]. Time-sliced KD-tree search: in a slice of length dt the
    separation changes by at most V dt, so querying radius r_max + V dt / 2
    at the slice midpoint finds every pair with relative speed <= V
    (V = v_cap; the default covers ~all of the catalogue, see tests)."""
    speed = np.linalg.norm(vel, axis=1)
    if v_cap is None:
        v_cap = 2.0 * np.quantile(speed, 0.999)
    if t_max <= 0:
        n_slices = 1
    elif n_slices is None:
        n_slices = max(1, int(np.ceil(v_cap * t_max / (2.0 * r_max))))
    dt = t_max / n_slices if t_max > 0 else 0.0
    keys = []
    n = len(pos)
    for k in range(n_slices):
        tm = (k + 0.5) * dt
        x = pos + vel * tm
        rq = r_max + 0.5 * v_cap * dt
        pr = cKDTree(x).query_pairs(rq, output_type="ndarray")
        if pr.size == 0:
            continue
        i, j = pr[:, 0], pr[:, 1]
        dmin = min_separation(pos[j] - pos[i], vel[j] - vel[i], 0.0, t_max)
        ok = dmin <= r_max
        keys.append(i[ok].astype(np.int64) * n + j[ok])
    if not keys:
        return np.empty((0, 2), dtype=np.int64)
    keys = np.unique(np.concatenate(keys))
    return np.stack([keys // n, keys % n], axis=1)


def min_separation(p, w, t0, t1):
    """Minimum of |p + w t| over t in [t0, t1] (vectorised)."""
    w2 = np.einsum("ij,ij->i", w, w)
    ts = -np.einsum("ij,ij->i", p, w) / np.where(w2 > 0, w2, np.inf)
    ts = np.clip(ts, t0, t1)
    return np.linalg.norm(p + w * ts[:, None], axis=1)


def open_windows(p, w, r_eff, t_max):
    """Interval [a, b] within [0, t_max] where |p + w t| <= r_eff.
    Empty windows are returned with a > b."""
    A = np.einsum("ij,ij->i", w, w)
    B = 2.0 * np.einsum("ij,ij->i", p, w)
    C = np.einsum("ij,ij->i", p, p) - r_eff ** 2
    a = np.full(len(p), np.inf)
    b = np.full(len(p), -np.inf)
    moving = A > 0
    disc = B ** 2 - 4 * A * C
    ok = moving & (disc >= 0) & (r_eff > 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    a[ok] = (-B[ok] - sq[ok]) / (2 * A[ok])
    b[ok] = (-B[ok] + sq[ok]) / (2 * A[ok])
    still = ~moving & (C <= 0) & (r_eff > 0)
    a[still], b[still] = -np.inf, np.inf
    a = np.maximum(a, 0.0)
    b = np.minimum(b, t_max)
    return a, b


def intercept_time(p, w, v):
    """tau solving |p + w tau| = v tau (v > |w|), vectorised over rows."""
    pw = np.einsum("ij,ij->i", p, w)
    p2 = np.einsum("ij,ij->i", p, p)
    den = v ** 2 - np.einsum("ij,ij->i", w, w)
    return (pw + np.sqrt(pw ** 2 + den * p2)) / den


class TemporalGraph:
    """Directed single-window edges for one hop range r."""

    def __init__(self, pos, vel, pairs, r, v_ship, t_max):
        i, j = pairs[:, 0], pairs[:, 1]
        p = pos[j] - pos[i]
        w = vel[j] - vel[i]
        r_eff = r * (1.0 - np.linalg.norm(w, axis=1) / v_ship)
        a, b = open_windows(p, w, r_eff, t_max)
        keep = a <= b
        i, j, p, w, a, b = i[keep], j[keep], p[keep], w[keep], a[keep], b[keep]
        # both directions: (i->j) uses (p, w); (j->i) uses (-p, -w)
        src = np.concatenate([i, j])
        dst = np.concatenate([j, i])
        P = np.concatenate([p, -p])
        W = np.concatenate([w, -w])
        A = np.concatenate([a, a])
        Bw = np.concatenate([b, b])
        order = np.argsort(src, kind="stable")
        self.n = len(pos)
        self.dst, self.P, self.W = dst[order], P[order], W[order]
        self.a, self.b = A[order], Bw[order]
        self.indptr = np.searchsorted(src[order], np.arange(self.n + 1))
        self.v = v_ship
        self.t_max = t_max
        self.n_edges = len(self.dst)

    def earliest_arrival(self, source=0, t0=0.0, settle=0.0):
        """Earliest arrival time (Myr) at every star; inf if not by t_max.
        `settle` = time a new colony needs before it can launch again."""
        best = np.full(self.n, np.inf)
        best[source] = t0
        done = np.zeros(self.n, dtype=bool)
        heap = [(t0, source)]
        while heap:
            t_u, u = heapq.heappop(heap)
            if done[u]:
                continue
            done[u] = True
            lo, hi = self.indptr[u], self.indptr[u + 1]
            if lo == hi:
                continue
            ready = t_u + (settle if u != source else 0.0)
            dep = np.maximum(ready, self.a[lo:hi])
            ok = dep <= self.b[lo:hi]
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0] + lo
            dep = dep[ok]
            p = self.P[idx] + self.W[idx] * dep[:, None]
            arr = dep + intercept_time(p, self.W[idx], self.v)
            v = self.dst[idx]
            better = (arr < best[v]) & (arr <= self.t_max)
            for vv, tt in zip(v[better].tolist(), arr[better].tolist()):
                if tt < best[vv]:
                    best[vv] = tt
                    heapq.heappush(heap, (tt, vv))
        return best


def crossing_r(r_grid, frac, target):
    """Smallest r (linear interpolation on the grid) with frac(r) >= target."""
    frac = np.asarray(frac)
    idx = np.nonzero(frac >= target)[0]
    if idx.size == 0:
        return float("nan")
    k = idx[0]
    if k == 0:
        return float(r_grid[0])
    r0, r1, f0, f1 = r_grid[k - 1], r_grid[k], frac[k - 1], frac[k]
    return float(r0 + (target - f0) * (r1 - r0) / max(f1 - f0, 1e-12))
