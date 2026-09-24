"""Comparison graphs and the operators of combinatorial Hodge theory.

Items are vertices 0..n-1.  An *edge flow* y assigns to every oriented edge
(i, j), i < j, a number: how strongly j is preferred to i (a log-odds, or a
mean outcome in [-1, 1]).  Three operators (Jiang, Lim, Yao & Ye 2011):

  grad  B : R^V -> R^E,  (B s)_ij = s_j - s_i        (a global ranking)
  curl  C : R^E -> R^T,  (C y)_ijk = y_ij + y_jk - y_ik   over triangles i<j<k
  Hodge Laplacian  L1 = B B^T + C^T C

and the orthogonal decomposition  R^E = im(B) (+) im(C^T) (+) ker(L1):
gradient (transitive), curl (locally cyclic, rock-paper-scissors on a
triangle), harmonic (globally cyclic around holes of the graph).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class ComparisonGraph:
    n: int
    edges: np.ndarray        # (E, 2) with i < j
    triangles: np.ndarray    # (T, 3) with i < j < k, all three edges present

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def B(self) -> np.ndarray:
        E = self.n_edges
        B = np.zeros((E, self.n))
        B[np.arange(E), self.edges[:, 0]] = -1.0
        B[np.arange(E), self.edges[:, 1]] = 1.0
        return B

    def C(self) -> np.ndarray:
        idx = {(int(i), int(j)): e for e, (i, j) in enumerate(self.edges)}
        C = np.zeros((len(self.triangles), self.n_edges))
        for t, (i, j, k) in enumerate(self.triangles):
            C[t, idx[(i, j)]] = 1.0
            C[t, idx[(j, k)]] = 1.0
            C[t, idx[(i, k)]] = -1.0
        return C

    def L1(self) -> np.ndarray:
        B, C = self.B(), self.C()
        return B @ B.T + C.T @ C


def from_edges(n: int, edges) -> ComparisonGraph:
    e = np.array(sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges if a != b}), dtype=int).reshape(-1, 2)
    es = {tuple(x) for x in e.tolist()}
    adj = [set() for _ in range(n)]
    for i, j in es:
        adj[i].add(j); adj[j].add(i)
    tri = []
    for i, j in es:
        for k in adj[i] & adj[j]:
            if k > j:
                tri.append((i, j, k))
    t = np.array(sorted(tri), dtype=int).reshape(-1, 3)
    return ComparisonGraph(n, e, t)


def random_graph(n: int, density: float, rng: np.random.Generator, connected: bool = True) -> ComparisonGraph:
    """Erdos-Renyi comparison graph; a random spanning path is added so the
    graph is connected (a ranking is then identifiable up to a constant)."""
    all_pairs = list(combinations(range(n), 2))
    keep = [p for p in all_pairs if rng.random() < density]
    if connected:
        perm = rng.permutation(n)
        keep += [(int(perm[i]), int(perm[i + 1])) for i in range(n - 1)]
    return from_edges(n, keep)


def complete_graph(n: int) -> ComparisonGraph:
    return from_edges(n, list(combinations(range(n), 2)))
