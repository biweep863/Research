"""Multi-probe exploration tours (open paths from the Sun).

Baseline = nearest-neighbour rule of Bjork (2007); improvement = 2-opt local
search (the move family behind Lin-Kernighan); bound = the MST length of the
probe's star set plus the Sun, which lower-bounds any open path through it.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist


def partition_directions(pos: np.ndarray, k: int, rng, iters: int = 50) -> np.ndarray:
    """k-means on unit direction vectors: one sky sector per probe."""
    u = pos / np.linalg.norm(pos, axis=1, keepdims=True)
    c = u[rng.choice(len(u), size=k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(u @ c.T, axis=1)
        for m in range(k):
            sel = u[lab == m]
            if len(sel):
                v = sel.sum(0)
                c[m] = v / np.linalg.norm(v)
    return np.argmax(u @ c.T, axis=1)


def path_length(D: np.ndarray, order: np.ndarray) -> float:
    return float(D[order[:-1], order[1:]].sum())


def nn_tour(D: np.ndarray, start: int = 0) -> np.ndarray:
    n = len(D)
    seen = np.zeros(n, dtype=bool)
    order = [start]
    seen[start] = True
    for _ in range(n - 1):
        d = np.where(seen, np.inf, D[order[-1]])
        nxt = int(np.argmin(d))
        order.append(nxt)
        seen[nxt] = True
    return np.array(order)


def two_opt(D: np.ndarray, order: np.ndarray, max_passes: int = 50) -> np.ndarray:
    """Open-path 2-opt, first node fixed. Reversing order[i..j] replaces edges
    (i-1,i),(j,j+1) by (i-1,j),(i,j+1); the last segment may be reversed
    freely (no closing edge)."""
    order = order.copy()
    n = len(order)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            a = order[i - 1]
            b = order[i]
            js = np.arange(i + 1, n)
            c = order[js]
            nxt = np.empty_like(js)
            nxt[:-1] = order[js[:-1] + 1]
            has_next = js < n - 1
            old = D[a, b] + np.where(has_next, D[c, np.where(has_next, nxt, 0)], 0.0)
            new = D[a, c] + np.where(has_next, D[b, np.where(has_next, nxt, 0)], 0.0)
            gain = old - new
            k = int(np.argmax(gain))
            if gain[k] > 1e-12:
                j = js[k]
                order[i:j + 1] = order[i:j + 1][::-1]
                improved = True
        if not improved:
            break
    return order


def mst_length(D: np.ndarray) -> float:
    return float(minimum_spanning_tree(D).sum())


def probe_tours(pos: np.ndarray, k: int, rng):
    """For stars in `pos` (Sun excluded), split among k probes and return
    per-probe (nn_order, opt_order, D, mst) with the Sun as local node 0."""
    lab = partition_directions(pos, k, rng)
    out = []
    for m in range(k):
        pts = np.vstack([np.zeros((1, 3)), pos[lab == m]])
        D = cdist(pts, pts)
        nn = nn_tour(D, 0)
        opt = two_opt(D, nn)
        out.append(dict(nn=nn, opt=opt, D=D, mst=mst_length(D)))
    return out
