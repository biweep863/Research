"""Experiment 1 -- static percolation of the real star graph.

Freeze the stars at their Gaia DR3 (J2016) positions and ask: with a ship
whose single hop can be at most r parsecs (refuelling at every star), what
fraction of the stars within `eval_radius_pc` can it reach from the Sun?
Exact answer: the bottleneck of the Euclidean MST path Sun -> star.

Outputs results/static.json: reach curve, quantile ranges r_q (smallest
range reaching a fraction q), bootstrap CIs over Gaia astrometric errors,
eval-radius sweep (boundary check), route stretch/hops, the Sun's local
"pocket" and the hop that bridges it, a Gaia-only variant (no supplement),
an injected unseen-brown-dwarf sweep, and map data.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from src.analysis.graph import (
    euclidean_mst, minimax_from_source, nearest_neighbour_distances, range_graph, reach_curve,
)
from src.experiments.common import setup, write
from src.model.kinematics import phase_space

QUANTILES = [0.1, 0.5, 0.9, 0.99]


R_FINE = np.linspace(1.5, 3.0, 151)


def quantiles(pos, R, k, curve_out=None):
    d = np.linalg.norm(pos, axis=1)
    ev = (d <= R) & (d > 0)
    _, b, _, _ = bottleneck(pos, k)
    if curve_out is not None:
        curve_out.append(reach_curve(b, ev, R_FINE).tolist())
    return {str(q): float(np.quantile(b[ev], q)) for q in QUANTILES}


def sun_pocket(tree, bott, r_jump, dist, sid):
    pocket = np.nonzero(bott < r_jump - 1e-9)[0]
    tc = tree.tocoo()
    br = np.isin(tc.row, pocket) & ~np.isin(tc.col, pocket) & (np.abs(tc.data - r_jump) < 1e-9)
    bridge = [{"from_source_id": int(sid[a]), "to_source_id": int(sid[b]),
               "from_dist_pc": float(dist[a]), "to_dist_pc": float(dist[b]),
               "length_pc": float(w)} for a, b, w in zip(tc.row[br], tc.col[br], tc.data[br])]
    return pocket, bridge


def inject_brown_dwarfs(pos, vel, frac, radius, rng):
    """Unseen population: frac x N extra nodes, uniform in the catalogue ball,
    velocities borrowed from random catalogue stars."""
    n = int(round(frac * (len(pos) - 1)))
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x = u * radius * rng.uniform(size=(n, 1)) ** (1 / 3)
    v = vel[1 + rng.integers(0, len(pos) - 1, size=n)]
    return np.vstack([pos, x]), np.vstack([vel, v])


def bottleneck(pos, k):
    tree = euclidean_mst(pos, k)
    return tree, *minimax_from_source(tree, 0)


def main():
    cfg, cat, rng = setup("static")
    sc = cfg["static"]
    pos, _ = phase_space(cat, rng)
    dist = np.linalg.norm(pos, axis=1)
    R = sc["eval_radius_pc"]
    ev = (dist <= R) & (dist > 0)
    r_grid = np.array(sc["r_grid"], dtype=float)

    tree, bott, hops, plen = bottleneck(pos, sc["knn_k"])
    curve = reach_curve(bott, ev, r_grid)
    rq = {str(q): float(np.quantile(bott[ev], q)) for q in QUANTILES}
    nn = nearest_neighbour_distances(pos[1:])

    sun_first_hop = float(np.min(dist[1:]))

    # The Sun's pocket: everything reachable with a range just below the
    # first big jump of the reach curve (r_10), and the MST hop bridging it.
    sid = np.concatenate([[0], cat["source_id"]])
    pocket, bridge = sun_pocket(tree, bott, float(np.quantile(bott[ev], 0.1)), dist, sid)

    # Gaia-only variant: drop the supplement (bright stars + 2 brown dwarfs).
    keep = np.concatenate([[True], cat["source_id"] >= 0])
    curves = {"gaia_plus": reach_curve(bott, ev, R_FINE).tolist()}
    c = []
    q_gaia_only = quantiles(pos[keep], R, sc["knn_k"], c)
    curves["gaia_only"] = c[0]
    pg = pos[keep]
    tg, bg, _, _ = bottleneck(pg, sc["knn_k"])
    dg = np.linalg.norm(pg, axis=1)
    pocket_g, bridge_g = sun_pocket(tg, bg, q_gaia_only["0.1"], dg, sid[keep])

    # Null model: same number of stars, uniform (Poisson) in the same ball.
    # Continuum percolation: n (4/3) pi r_c^3 = 2.7348 (Lorenz & Ziff 2001).
    n_ball = len(pos) - 1
    dens = n_ball / (4 / 3 * np.pi * cfg["data"]["radius_pc"] ** 3)
    r_c_poisson = float((2.7348 / (4 / 3 * np.pi * dens)) ** (1 / 3))
    poisson = []
    for d in range(3):
        rp = np.random.default_rng(700 + d)
        u = rp.normal(size=(n_ball, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        x = u * cfg["data"]["radius_pc"] * rp.uniform(size=(n_ball, 1)) ** (1 / 3)
        c = []
        poisson.append(quantiles(np.vstack([np.zeros((1, 3)), x]), R, sc["knn_k"], c))
        if d == 0:
            curves["poisson"] = c[0]
    q_poisson = {q: float(np.mean([x[q] for x in poisson])) for q in poisson[0]}

    # Unseen brown dwarfs: inject a random population and recompute.
    pos_full, vel_full = phase_space(cat, rng)
    bd = []
    for f in sc["bd_fracs"]:
        c = []
        draws = [quantiles(inject_brown_dwarfs(pos_full, vel_full, f, cfg["data"]["radius_pc"],
                                               np.random.default_rng(500 + d))[0], R, sc["knn_k"], c)
                 for d in range(sc["bd_draws"] if f > 0 else 1)]
        curves[f"bd_{f:g}"] = c[0]
        bd.append({"frac": f, **{q: float(np.mean([x[q] for x in draws])) for q in draws[0]},
                   "r50_min": float(min(x["0.5"] for x in draws)),
                   "r50_max": float(max(x["0.5"] for x in draws))})

    # Shortest flown routes at the range that reaches 90% of stars.
    r90 = rq["0.9"] * 1.0001
    g = range_graph(pos, r90)
    d_fly = dijkstra(g, indices=0, directed=False)
    n_hops = dijkstra(g, indices=0, directed=False, unweighted=True)
    ok = ev & np.isfinite(d_fly)
    stretch = d_fly[ok] / dist[ok]
    # stretch / hops by straight-line distance bin
    bins = np.linspace(0, R, 6)
    by_dist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (dist > lo) & (dist <= hi)
        if m.any():
            by_dist.append({"lo": float(lo), "hi": float(hi),
                            "stretch_median": float(np.median(d_fly[m] / dist[m])),
                            "hops_median": float(np.median(n_hops[m]))})

    # Boundary check: r_50 for several evaluation radii.
    sweep = []
    for Rs in sc["radius_sweep_pc"]:
        m = (dist <= Rs) & (dist > 0)
        sweep.append({"eval_radius_pc": Rs, "n": int(m.sum()),
                      "r50": float(np.quantile(bott[m], 0.5)),
                      "r90": float(np.quantile(bott[m], 0.9))})

    # Astrometric bootstrap (parallax / PM / RV errors; RV-fill resampled).
    boots = []
    for b in range(sc["n_boot"]):
        rb = np.random.default_rng(1000 + b)
        pb, _ = phase_space(cat, rb, perturb=True)
        db = np.linalg.norm(pb, axis=1)
        eb = (db <= R) & (db > 0)
        _, bb, _, _ = bottleneck(pb, sc["knn_k"])
        boots.append({str(q): float(np.quantile(bb[eb], q)) for q in QUANTILES})
    ci = {q: [float(np.quantile([x[q] for x in boots], 0.025)),
              float(np.quantile([x[q] for x in boots], 0.975))] for q in boots[0]}

    # Map data: stars within 8 pc, their MST edges and bottlenecks.
    near = np.nonzero(dist <= 8.0)[0]
    sub = tree[near][:, near].tocoo()
    edges = [[int(a), int(b)] for a, b in zip(sub.row, sub.col) if a < b]

    write(cfg, "static", {
        "n_stars": int(len(pos) - 1),
        "eval_radius_pc": R,
        "n_eval": int(ev.sum()),
        "r_grid": r_grid.tolist(),
        "r_fine": R_FINE.tolist(),
        "curves": curves,
        "reach_frac": curve.tolist(),
        "r_quantiles": rq,
        "r_quantiles_ci95": ci,
        "nn_median_pc": float(np.median(nn)),
        "nn_p90_pc": float(np.quantile(nn, 0.9)),
        "sun_first_hop_pc": sun_first_hop,
        "pocket_size": int(len(pocket)),
        "pocket_source_ids": [int(sid[k]) for k in pocket][:200],
        "pocket_dist_pc": [float(dist[k]) for k in pocket][:200],
        "bridge": bridge,
        "pocket_gaia_only_size": int(len(pocket_g)),
        "pocket_gaia_only_dist_pc": [float(dg[k]) for k in pocket_g][:200],
        "bridge_gaia_only": bridge_g,
        "r_quantiles_gaia_only": q_gaia_only,
        "density_pc3": float(dens),
        "r_c_poisson_analytic": r_c_poisson,
        "r_quantiles_poisson": q_poisson,
        "brown_dwarf_sweep": bd,
        "max_bottleneck_eval_pc": float(bott[ev].max()),
        "route_r90_pc": float(r90),
        "route_stretch_median": float(np.median(stretch)),
        "route_hops_median": float(np.median(n_hops[ok])),
        "route_by_distance": by_dist,
        "radius_sweep": sweep,
        "bottleneck_hist": np.histogram(bott[ev], bins=60, range=(0, 6))[0].tolist(),
        "map": {"xyz": pos[near].tolist(), "bott": bott[near].tolist(),
                "edges": edges},
    })


if __name__ == "__main__":
    main()
