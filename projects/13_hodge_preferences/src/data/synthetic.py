"""Synthetic preferences with controlled Hodge content.

Items have centred features x_i ~ N(0, I_d).  The true preference of j over i
is a function of the two feature vectors,

    g*(x_i, x_j) = beta_g [s(x_j) - s(x_i)]  +  beta_c * c(x_i, x_j)

with s a smooth utility (a random 1-hidden-layer tanh network) -- a pure
gradient flow on any graph -- and c a skew nonlinear interaction
c(x, y) = tanh(x^T R y) with R skew: on a complete graph with centred
features its divergence vanishes to first order, so it lives almost entirely
in cycle space (rock-paper-scissors in feature space).  Because the truth is
a function of features, held-out pairs have well-defined targets.  The
exact component energies on the training graph are measured by projection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hodge.decompose import energy_fractions, projectors
from ..hodge.graph import ComparisonGraph, random_graph


@dataclass
class PrefTask:
    X: np.ndarray                 # (n, d)
    g_train: ComparisonGraph
    y_train: np.ndarray           # true flow on training edges
    heldout: np.ndarray           # (H, 2) held-out pairs i<j
    y_heldout: np.ndarray
    P: dict
    fractions: dict
    params: dict


def _utility(X, rng, hidden=16):
    d = X.shape[1]
    W = rng.normal(0, 1 / np.sqrt(d), (d, hidden)); v = rng.normal(0, 1, hidden)
    u = rng.normal(0, 1 / np.sqrt(d), d)
    return np.tanh(X @ W) @ v / np.sqrt(hidden) + X @ u


def _skew(X, rng, scale=1.0):
    d = X.shape[1]
    A = rng.normal(0, 1, (d, d)); R = (A - A.T) / np.sqrt(2 * d)
    return lambda i, j: np.tanh(scale * np.einsum("ed,dk,ek->e", X[i], R, X[j]))


def make_task(n: int, d: int, density: float, beta_g: float, beta_c: float, rng: np.random.Generator,
              heldout_frac: float = 0.3) -> PrefTask:
    X = rng.normal(0, 1, (n, d)); X -= X.mean(0)
    s = _utility(X, rng)
    c = _skew(X, rng)
    g = random_graph(n, density, rng)
    all_pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    have = {tuple(e) for e in g.edges.tolist()}
    rest = np.array([p for p in all_pairs.tolist() if tuple(p) not in have])
    if len(rest):
        rest = rest[rng.permutation(len(rest))[: max(1, int(heldout_frac * len(all_pairs)))]]

    def truth(pairs):
        pairs = np.asarray(pairs)
        i, j = pairs[:, 0], pairs[:, 1]
        return beta_g * (s[j] - s[i]) + beta_c * c(i, j)

    y = truth(g.edges)
    P = projectors(g)
    return PrefTask(X, g, y, rest, truth(rest) if len(rest) else np.zeros(0), P, energy_fractions(y, P),
                    {"n": n, "d": d, "density": density, "beta_g": beta_g, "beta_c": beta_c})


def gaussian_noise(y: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return y + rng.normal(0, sigma, y.shape)


def votes(y: np.ndarray, n_votes: int, rng: np.random.Generator, flip: float = 0.0):
    """Binary outcomes per edge from P(j beats i) = sigmoid(y); a fraction
    `flip` of votes is flipped at random (annotator noise)."""
    p = 1 / (1 + np.exp(-y))
    wins = rng.binomial(n_votes, p)
    if flip > 0:
        flips_w = rng.binomial(wins, flip)
        flips_l = rng.binomial(n_votes - wins, flip)
        wins = wins - flips_w + flips_l
    return wins, np.full_like(wins, n_votes)
