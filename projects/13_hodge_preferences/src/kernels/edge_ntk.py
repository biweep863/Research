"""Edge-level kernels of preference models and their Hodge structure.

A preference model predicts an edge flow g_ij (preference of j over i).
Linearised gradient descent on the squared loss moves the residual on the
training edges as  r_{t+1} = (I - eta/E * Theta_E) r_t,  with Theta_E the NTK
between edges.  Three models:

* scalar     g_ij = f(x_j) - f(x_i)              Theta_E = B K B^T
* antisym    g_ij = h(x_i,x_j) - h(x_j,x_i)      Theta_E = K(z,z') - K(z,~z') - K(~z,z') + K(~z,~z')
             with z = [x_i, x_j] and ~z = [x_j, x_i]
* GPM        g_ij = e(x_i)^T R e(x_j), R skew    (empirical NTK only; see empirical.py)

The Hodge *rate* of a component c is the mean eigenvalue of Theta_E restricted
to that subspace, kappa_c = tr(P_c Theta_E P_c) / dim_c (the expected Rayleigh
quotient of a random flow in the subspace); for a specific target flow y_c it
is the Rayleigh quotient y_c^T Theta_E y_c / |y_c|^2.
"""
from __future__ import annotations

import numpy as np

from ..hodge.decompose import COMPONENTS
from ..hodge.graph import ComparisonGraph
from .ntk_relu import relu_ntk


def scalar_edge_ntk(g: ComparisonGraph, X: np.ndarray, depth: int = 1, **kw) -> np.ndarray:
    K = relu_ntk(X, depth=depth, **kw)
    B = g.B()
    return B @ K @ B.T


def pair_inputs(g: ComparisonGraph, X: np.ndarray):
    i, j = g.edges[:, 0], g.edges[:, 1]
    return np.concatenate([X[i], X[j]], 1), np.concatenate([X[j], X[i]], 1)


def antisym_edge_ntk(g: ComparisonGraph, X: np.ndarray, depth: int = 1, **kw) -> np.ndarray:
    Z, Zs = pair_inputs(g, X)
    A = np.concatenate([Z, Zs], 0)
    K = relu_ntk(A, depth=depth, **kw)
    E = len(Z)
    Kzz, Kzs, Ksz, Kss = K[:E, :E], K[:E, E:], K[E:, :E], K[E:, E:]
    return Kzz - Kzs - Ksz + Kss


def hodge_rates(Theta: np.ndarray, P: dict) -> dict[str, float]:
    out = {}
    for c in COMPONENTS:
        d = np.trace(P[c])
        out[c] = float(np.trace(P[c] @ Theta @ P[c]) / d) if d > 0.5 else float("nan")
    return out


def rayleigh(Theta: np.ndarray, y: np.ndarray) -> float:
    n2 = float(y @ y)
    return float(y @ Theta @ y / n2) if n2 > 0 else float("nan")


def spectral_mass(Theta: np.ndarray, P: dict) -> dict:
    """Eigen-decomposition of Theta and, for each eigenvector, the fraction of
    its mass in each Hodge component."""
    w, V = np.linalg.eigh(0.5 * (Theta + Theta.T))
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    mass = {c: np.sum((P[c] @ V) ** 2, 0) for c in COMPONENTS}
    return {"eig": w, **mass}


def linearised_fit(Theta: np.ndarray, y: np.ndarray, P: dict, eta: float, steps: np.ndarray) -> dict:
    """Fraction of each component fitted after t steps of linearised GD on
    L = 1/(2E) sum (g - y)^2 from g_0 = 0:  r_t = (I - eta/E Theta)^t y."""
    E = len(y)
    w, V = np.linalg.eigh(0.5 * (Theta + Theta.T))
    a = V.T @ y
    out = {c: [] for c in COMPONENTS}
    for t in steps:
        r = V @ (a * (1 - eta * w / E) ** t)
        for c in COMPONENTS:
            yc = P[c] @ y
            n = float(yc @ yc)
            out[c].append(1 - float(np.sum((P[c] @ r) ** 2)) / n if n > 0 else float("nan"))
    return out
