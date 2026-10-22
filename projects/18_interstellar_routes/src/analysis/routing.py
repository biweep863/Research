"""Point-to-point routing on a static range graph, and two constrained variants.

* Dijkstra (scipy) and A* (heuristic = straight-line distance, admissible and
  consistent on Euclidean graphs) for shortest flown distance.
* Greedy geographic routing -- hop to the in-range neighbour closest to the
  target -- the natural "heuristic pilot", and the point-to-point analogue of
  the nearest-neighbour rule used in earlier galactic-exploration models.
* Fuel bottleneck with refuelling only at selected stars (e.g. known
  exoplanet hosts): the smallest tank range R that reaches each star.
* Photon-sail network: directed edges whose speed is capped by the *target*
  star's braking luminosity.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

from src.analysis.graph import minimax_from_source


def astar(g: csr_matrix, pos: np.ndarray, s: int, t: int):
    """A* on a CSR graph with Euclidean weights. Returns (dist, n_expanded)."""
    h = lambda u: float(np.linalg.norm(pos[u] - pos[t]))  # noqa: E731
    gsc = {s: 0.0}
    heap = [(h(s), s)]
    closed = set()
    while heap:
        f, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == t:
            return gsc[u], len(closed)
        lo, hi = g.indptr[u], g.indptr[u + 1]
        for v, w in zip(g.indices[lo:hi].tolist(), g.data[lo:hi].tolist()):
            nd = gsc[u] + w
            if v not in closed and nd < gsc.get(v, np.inf):
                gsc[v] = nd
                heapq.heappush(heap, (nd + h(v), v))
    return np.inf, len(closed)


def dijkstra_expansions(g: csr_matrix, s: int, t: int):
    """Plain Dijkstra stopped at t (same code path as A* with h=0)."""
    dist = {s: 0.0}
    heap = [(0.0, s)]
    closed = set()
    while heap:
        d, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == t:
            return d, len(closed)
        lo, hi = g.indptr[u], g.indptr[u + 1]
        for v, w in zip(g.indices[lo:hi].tolist(), g.data[lo:hi].tolist()):
            nd = d + w
            if v not in closed and nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return np.inf, len(closed)


def greedy_route(g: csr_matrix, pos: np.ndarray, s: int, t: int, max_hops: int = 10_000):
    """Greedy geographic routing. Returns (flown distance, success flag).
    Fails when no in-range neighbour is closer to the target (a 'void')."""
    u, flown = s, 0.0
    for _ in range(max_hops):
        if u == t:
            return flown, True
        lo, hi = g.indptr[u], g.indptr[u + 1]
        nb = g.indices[lo:hi]
        if nb.size == 0:
            return flown, False
        dn = np.linalg.norm(pos[nb] - pos[t], axis=1)
        k = int(np.argmin(dn))
        if dn[k] >= np.linalg.norm(pos[u] - pos[t]):
            return flown, False
        flown += float(g.data[lo + k])
        u = int(nb[k])
    return flown, False


def fuel_bottleneck(g: csr_matrix, refuel: np.ndarray, source: int = 0, limit=np.inf):
    """Smallest tank range R (distance flown between refuels, hops limited by
    the range graph g) that reaches every star from `source`.

    Exact: with refuelling allowed only at nodes in `refuel` (source included),
    the station-to-station requirement is the minimax over the station graph
    whose edge weight is the shortest in-graph distance between stations; the
    last leg to a non-station star adds max(., dist(station, star))."""
    stations = np.unique(np.concatenate([[source], np.nonzero(refuel)[0]]))
    D = dijkstra(g, directed=False, indices=stations, limit=limit)   # (S, N)
    S = D[:, stations]
    S = np.where(np.isfinite(S), S, 0.0)
    np.fill_diagonal(S, 0.0)
    t = minimum_spanning_tree(csr_matrix(S))
    t = (t + t.T).tocsr()
    src_local = int(np.nonzero(stations == source)[0][0])
    b_st, _, _ = minimax_from_source(t, src_local)               # per station
    need = np.maximum(b_st[:, None], D).min(axis=0)               # per star
    return need, stations


def sail_graph(pos, lum, v_launch_kms, tau_max_yr, brake_fn, base_pairs, base_w):
    """Directed photon-sail graph from an undirected candidate pair list.

    Hop i->j flies at v = min(v_launch, v_brake(L_j)); it is allowed when the
    flight time d/v <= tau_max. Edge weight = flight time (yr)."""
    from src.model.kinematics import KMS_TO_PCMYR
    i, j = base_pairs[:, 0], base_pairs[:, 1]
    src = np.concatenate([i, j])
    dst = np.concatenate([j, i])
    d = np.concatenate([base_w, base_w])
    v = np.minimum(v_launch_kms, brake_fn(lum[dst]))                 # km/s
    t_yr = d / (v * KMS_TO_PCMYR) * 1e6
    ok = t_yr <= tau_max_yr
    n = len(pos)
    return coo_matrix((t_yr[ok], (src[ok], dst[ok])), shape=(n, n)).tocsr()
