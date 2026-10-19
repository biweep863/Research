"""Static star graph: Euclidean MST, minimax ("bottleneck") reach, range graphs.

Key fact used throughout: on a complete Euclidean graph, the path between two
nodes that minimises the *largest single hop* runs along the minimum spanning
tree. So the smallest per-hop range r that lets a ship from the Sun reach star
j is the largest MST edge on the tree path Sun -> j (its "bottleneck"), and
the reachable fraction at range r is the fraction of stars with bottleneck <= r.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree


def knn_graph(pos: np.ndarray, k: int) -> csr_matrix:
    """Symmetric k-nearest-neighbour graph with Euclidean edge weights."""
    n = len(pos)
    k = min(k, n - 1)
    d, idx = cKDTree(pos).query(pos, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    w = d[:, 1:].ravel()
    g = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    return g.maximum(g.T)


def euclidean_mst(pos: np.ndarray, k: int = 16) -> csr_matrix:
    """EMST via the MST of a kNN graph; k is doubled until the graph is
    connected (the EMST is a subgraph of the kNN graph for modest k in 3D;
    tests check equality against the dense-graph MST)."""
    while True:
        g = knn_graph(pos, k)
        ncomp, _ = connected_components(g, directed=False)
        if ncomp == 1 or k >= len(pos) - 1:
            break
        k *= 2
    t = minimum_spanning_tree(g)
    return (t + t.T).tocsr()


def minimax_from_source(tree: csr_matrix, source: int = 0):
    """Bottleneck (largest hop), hop count and path length along the tree
    from `source` to every node."""
    order, pred = breadth_first_order(tree, source, directed=False, return_predecessors=True)
    n = tree.shape[0]
    bott = np.full(n, np.inf)
    hops = np.full(n, -1, dtype=np.int64)
    plen = np.full(n, np.inf)
    bott[source], hops[source], plen[source] = 0.0, 0, 0.0
    child = order[1:]
    parent = pred[child]
    w_edge = np.asarray(tree[parent, child]).ravel()
    for v, u, w in zip(child.tolist(), parent.tolist(), w_edge.tolist()):
        bott[v] = max(bott[u], w)
        hops[v] = hops[u] + 1
        plen[v] = plen[u] + w
    return bott, hops, plen


def minimax_bruteforce(pos: np.ndarray, source: int = 0) -> np.ndarray:
    """O(n^3) widest-path reference on the complete graph (tests only)."""
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    b = d.copy()
    n = len(pos)
    for m in range(n):
        b = np.minimum(b, np.maximum(b[:, m:m + 1], b[m:m + 1, :]))
    out = b[source].copy()
    out[source] = 0.0
    return out


def reach_curve(bott: np.ndarray, mask: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    """Fraction of masked stars whose bottleneck <= r, for each r."""
    b = np.sort(bott[mask])
    return np.searchsorted(b, r_grid, side="right") / b.size


def range_graph(pos: np.ndarray, r: float) -> csr_matrix:
    """All pairs closer than r (exact), Euclidean weights."""
    pairs = cKDTree(pos).query_pairs(r, output_type="ndarray")
    w = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    n = len(pos)
    g = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    return (g + g.T).tocsr()


def nearest_neighbour_distances(pos: np.ndarray) -> np.ndarray:
    d, _ = cKDTree(pos).query(pos, k=2)
    return d[:, 1]
