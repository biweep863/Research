"""Orthogonal Hodge projectors and HodgeRank."""
from __future__ import annotations

import numpy as np

from .graph import ComparisonGraph

COMPONENTS = ("grad", "curl", "harm")


def _range_projector(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Orthogonal projector onto the column space of A."""
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros((A.shape[0], A.shape[0]))
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    r = int(np.sum(s > tol * max(s.max(), 1.0)))
    Ur = U[:, :r]
    return Ur @ Ur.T


def projectors(g: ComparisonGraph) -> dict[str, np.ndarray]:
    """P_grad, P_curl, P_harm: orthogonal, idempotent, sum to identity."""
    B, C = g.B(), g.C()
    Pg = _range_projector(B)
    Pc = _range_projector(C.T) if len(g.triangles) else np.zeros((g.n_edges, g.n_edges))
    Ph = np.eye(g.n_edges) - Pg - Pc
    Ph = 0.5 * (Ph + Ph.T)
    return {"grad": Pg, "curl": Pc, "harm": Ph}


def dims(g: ComparisonGraph, P: dict | None = None) -> dict[str, int]:
    P = P or projectors(g)
    return {c: int(round(np.trace(P[c]))) for c in COMPONENTS}


def split(y: np.ndarray, P: dict) -> dict[str, np.ndarray]:
    return {c: P[c] @ y for c in COMPONENTS}


def energies(y: np.ndarray, P: dict) -> dict[str, float]:
    return {c: float(np.sum((P[c] @ y) ** 2)) for c in COMPONENTS}


def energy_fractions(y: np.ndarray, P: dict) -> dict[str, float]:
    e = energies(y, P)
    tot = sum(e.values()) + 1e-300
    return {c: e[c] / tot for c in COMPONENTS}


def hodgerank(g: ComparisonGraph, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Least-squares global ranking s (mean zero) with  B s ~ y  (weighted)."""
    B = g.B()
    if w is not None:
        sw = np.sqrt(w)[:, None]
        s, *_ = np.linalg.lstsq(sw * B, sw[:, 0] * y, rcond=None)
    else:
        s, *_ = np.linalg.lstsq(B, y, rcond=None)
    return s - s.mean()
