"""Unit tests: Hodge decomposition, edge kernels, networks, dynamics formula,
noise fraction, HodgeRank.  Seconds, no data downloads."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

from src.hodge.graph import complete_graph, from_edges, random_graph          # noqa: E402
from src.hodge.decompose import (dims, energy_fractions, hodgerank,           # noqa: E402
                                 projectors)
from src.kernels.edge_ntk import (antisym_edge_ntk, hodge_rates,              # noqa: E402
                                  linearised_fit, scalar_edge_ntk)
from src.kernels.empirical import empirical_edge_ntk                          # noqa: E402
from src.models.nets import build                                             # noqa: E402
from src.data.synthetic import make_task                                      # noqa: E402


def test_projectors_orthogonal_idempotent_complete():
    g = random_graph(15, 0.3, np.random.default_rng(0)); P = projectors(g)
    I = np.eye(g.n_edges)
    assert np.allclose(P["grad"] + P["curl"] + P["harm"], I, atol=1e-9)
    for a in P:
        assert np.allclose(P[a] @ P[a], P[a], atol=1e-8)
        for b in P:
            if a != b:
                assert np.abs(P[a] @ P[b]).max() < 1e-8
    d = dims(g, P)
    assert d["grad"] == 14                                    # connected: n - 1


def test_rock_paper_scissors_is_curl_and_square_is_harmonic():
    g = complete_graph(3)                                     # edges (0,1),(0,2),(1,2)
    y = np.array([1.0, -1.0, 1.0])                            # 0<1, 2<0 (0 beats 2), 1<2: a cycle
    f = energy_fractions(y, projectors(g))
    assert f["curl"] > 1 - 1e-9
    sq = from_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)])       # 4-cycle, no triangles
    flow = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0, (0, 3): -1.0}   # 0<1<2<3<0
    y = np.array([flow[tuple(e)] for e in sq.edges.tolist()])
    f = energy_fractions(y, projectors(sq))
    assert f["harm"] > 1 - 1e-9


def test_scalar_edge_ntk_is_blind_to_cycles():
    rng = np.random.default_rng(1)
    g = random_graph(20, 0.3, rng); X = rng.normal(size=(20, 4)); P = projectors(g)
    r = hodge_rates(scalar_edge_ntk(g, X), P)
    assert abs(r["curl"]) < 1e-10 and (np.isnan(r["harm"]) or abs(r["harm"]) < 1e-10) and r["grad"] > 0


def test_antisym_and_gpm_are_antisymmetric():
    x = torch.randn(7, 3); y = torch.randn(7, 3)
    for kind in ("antisym", "gpm"):
        m = build(kind, 3, 64, 1, seed=0)
        # perturb weights so the centred output is non-zero
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(0.1 * torch.randn_like(p))
        assert torch.allclose(m(x, y), -m(y, x), atol=1e-5)
        assert m(x, x).abs().max() < 1e-5


def test_analytic_ntk_matches_empirical_at_width():
    rng = np.random.default_rng(2)
    g = random_graph(12, 0.5, rng); X = rng.normal(size=(12, 3)); X -= X.mean(0)
    Xi = torch.tensor(X[g.edges[:, 0]], dtype=torch.float32); Xj = torch.tensor(X[g.edges[:, 1]], dtype=torch.float32)
    for kind, ref in (("scalar", scalar_edge_ntk(g, X)), ("antisym", antisym_edge_ntk(g, X))):
        Te = empirical_edge_ntk(build(kind, 3, 4096, 1, seed=3), Xi, Xj)
        assert np.linalg.norm(Te - ref) / np.linalg.norm(ref) < 0.08


def test_linearised_flow_formula():
    rng = np.random.default_rng(3)
    g = random_graph(10, 0.5, rng); P = projectors(g)
    A = rng.normal(size=(g.n_edges, g.n_edges)); Th = A @ A.T / g.n_edges
    y = rng.normal(size=g.n_edges); E = g.n_edges
    eta = 0.5 * E / np.linalg.eigvalsh(Th).max()
    r = y.copy()
    for _ in range(25):
        r = r - eta / E * Th @ r
    lin = linearised_fit(Th, y, P, eta, np.array([25]))
    fit_g = 1 - np.sum((P["grad"] @ r) ** 2) / np.sum((P["grad"] @ y) ** 2)
    assert abs(lin["grad"][0] - fit_g) < 1e-9


def test_noise_fraction_in_gradient_space():
    rng = np.random.default_rng(4)
    g = random_graph(30, 0.4, rng); P = projectors(g)
    z = rng.normal(size=(400, g.n_edges))
    fr = np.sum((z @ P["grad"]) * z, 1) / np.sum(z * z, 1)
    theory = 29 / g.n_edges
    assert abs(fr.mean() - theory) < 3 * fr.std() / np.sqrt(len(fr)) + 1e-3


def test_hodgerank_recovers_planted_ranking():
    rng = np.random.default_rng(5)
    g = random_graph(25, 0.3, rng); s = rng.normal(size=25)
    y = g.B() @ s + 0.01 * rng.normal(size=g.n_edges)
    sh = hodgerank(g, y)
    assert np.corrcoef(sh, s - s.mean())[0, 1] > 0.999


def test_synthetic_task_has_all_components():
    T = make_task(30, 6, 0.2, 1.0, 2.0, np.random.default_rng(6))
    assert all(v > 0.02 for v in T.fractions.values())
    assert abs(sum(T.fractions.values()) - 1) < 1e-9 and len(T.heldout) > 0
