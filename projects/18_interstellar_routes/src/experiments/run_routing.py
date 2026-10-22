"""Experiment 3 -- propulsion, cost-aware routing, refuelling and sails.

(a) Technology -> range. Cruise speed for each rocket at mass ratios R (accel
    + brake on one tank), and the per-hop lifetime tau* a ship must survive to
    percolate: tau* = r_50 / v_cruise, using r_50 for frozen stars (Exp. 1) and
    for moving stars with waiting (Exp. 2).
(b) Pilots. On the static range graph: A* vs Dijkstra (node expansions) and
    the greedy geographic pilot (success rate and stretch vs optimal).
(c) Refuelling only at known exoplanet hosts: the tank range needed to reach
    each star (exact fuel bottleneck), vs refuelling anywhere.
(d) Photon sails braking on the target star's light: reach and travel time vs
    a rocket at the same launch speed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from src.analysis.graph import euclidean_mst, minimax_from_source, range_graph
from src.analysis.routing import astar, dijkstra_expansions, fuel_bottleneck, greedy_route, sail_graph
from src.experiments.common import setup, write
from src.model.kinematics import C_KMS, luminosity_lsun, phase_space
from src.model.propulsion import ROCKETS, hop_range_pc, rocket_cruise_speed_kms, sail_brake_speed_kms


def _load(cfg, name):
    p = Path(cfg.get("output_dir", "results")) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    cfg, cat, rng = setup("routing")
    rc = cfg["routing"]
    pos, _ = phase_space(cat, rng)
    dist = np.linalg.norm(pos, axis=1)
    R = rc["eval_radius_pc"]
    ev = (dist <= R) & (dist > 0)
    bott, _, _ = minimax_from_source(euclidean_mst(pos), 0)

    # ---------------- (a) technology -> range ---------------- #
    st, td = _load(cfg, "static"), _load(cfg, "timedep")
    r50_static = st["r_quantiles"]["0.5"] if st else float(np.quantile(bott[ev], 0.5))
    r50_moving = None
    if td:
        H = td["horizons_myr"]
        k = H.index(0.5) if 0.5 in H else len(H) - 1
        r50_moving = {"T_myr": H[k], "r50": td["r_q_of_T"]["0.5"][k]}
    techs = []
    for rk in ROCKETS:
        for mr in rc["mass_ratios"]:
            v = float(rocket_cruise_speed_kms(rk.v_exhaust_kms, mr))
            row = {"tech": rk.name, "v_exhaust_kms": rk.v_exhaust_kms, "mass_ratio": mr,
                   "v_cruise_kms": v, "v_cruise_c": v / C_KMS,
                   "tau_star_static_yr": r50_static / float(hop_range_pc(v, 1.0)),
                   "reach_by_lifetime": []}
            if r50_moving:
                row["tau_star_moving_yr"] = r50_moving["r50"] / float(hop_range_pc(v, 1.0))
            for tau in rc["lifetimes_yr"]:
                r = float(hop_range_pc(v, tau))
                row["reach_by_lifetime"].append(
                    {"tau_yr": tau, "r_hop_pc": r, "reach_static": float(np.mean(bott[ev] <= r))})
            techs.append(row)

    # ---------------- (b) pilots on the static range graph ---------------- #
    pilots = []
    qrng = np.random.default_rng(cfg.get("seed", 0) + 7)
    for r in rc["route_ranges_pc"]:
        g = range_graph(pos, r)
        d0 = dijkstra(g, indices=0, directed=False)
        reach = np.nonzero(ev & np.isfinite(d0))[0]
        tg = qrng.choice(reach, size=min(rc["n_route_queries"], len(reach)), replace=False)
        ok, stretch, exp_a, exp_d = [], [], [], []
        for t in tg.tolist():
            flown, success = greedy_route(g, pos, 0, t)
            ok.append(success)
            if success:
                stretch.append(flown / d0[t])
        for t in tg[: rc["n_astar_queries"]].tolist():
            da, na = astar(g, pos, 0, t)
            dd, nd = dijkstra_expansions(g, 0, t)
            assert abs(da - dd) < 1e-9
            exp_a.append(na)
            exp_d.append(nd)
        pilots.append({
            "r_pc": r, "n_queries": int(len(tg)),
            "greedy_success": float(np.mean(ok)),
            "greedy_stretch_median": float(np.median(stretch)) if stretch else None,
            "optimal_stretch_median": float(np.median(d0[tg] / dist[tg])),
            "astar_expansion_ratio_median": float(np.median(np.array(exp_a) / np.array(exp_d))),
        })

    # ---------------- (c) refuel only at exoplanet hosts ---------------- #
    fr = rc["fuel_radius_pc"]
    sub = np.nonzero(dist <= fr)[0]                          # Sun is index 0 of sub
    host = np.concatenate([[False], cat["is_host"]])[sub]
    gsub = range_graph(pos[sub], rc["fuel_hop_pc"])
    need, stations = fuel_bottleneck(gsub, host, 0)
    evs = (dist[sub] <= min(R, fr)) & (dist[sub] > 0)
    b_any, _, _ = minimax_from_source(euclidean_mst(pos[sub]), 0)
    fuel = {
        "radius_pc": fr, "hop_pc": rc["fuel_hop_pc"], "n_stars": int(len(sub) - 1),
        "n_hosts": int(host.sum()),
        "need_quantiles": {str(q): float(np.quantile(need[evs], q)) for q in (0.1, 0.5, 0.9)},
        "reach_frac_by_tank": [{"tank_pc": t, "frac": float(np.mean(need[evs] <= t))}
                               for t in rc["tank_grid_pc"]],
        "refuel_anywhere_r50": float(np.quantile(b_any[evs], 0.5)),
        "nearest_host_pc": float(np.min(dist[sub][host])),
    }

    # ---------------- (d) photon sails ---------------- #
    # Missing G -> treated as dark (no braking); the Sun only launches.
    lum = np.concatenate([[1.0], luminosity_lsun(np.nan_to_num(cat["phot_g_mean_mag"], nan=30.0),
                                                 cat["parallax"])])
    v_launch = rc["sail_v_launch_c"] * C_KMS
    pairs = cKDTree(pos).query_pairs(rc["sail_range_cap_pc"], output_type="ndarray")
    w = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    sails = []
    for tau in rc["sail_lifetimes_yr"]:
        res = {"tau_yr": tau}
        for label, brake in (("sail", sail_brake_speed_kms),
                             ("rocket_same_speed", lambda L: np.full_like(L, np.inf))):
            g = sail_graph(pos, lum, v_launch, tau, brake, pairs, w)
            t_arr = dijkstra(g, indices=0, directed=True)
            reach = np.isfinite(t_arr)
            res[label] = {"reach": float(np.mean(reach[ev])),
                          "t50_yr": float(np.quantile(t_arr[ev], 0.5)) if np.mean(reach[ev]) >= 0.5 else None}
        sails.append(res)
    lum_ev = lum[ev]
    brake_v = sail_brake_speed_kms(lum_ev)

    write(cfg, "routing", {
        "eval_radius_pc": R,
        "r50_static": r50_static, "r50_moving": r50_moving,
        "techs": techs,
        "pilots": pilots,
        "fuel": fuel,
        "sail": {"v_launch_c": rc["sail_v_launch_c"], "by_lifetime": sails,
                 "brake_speed_kms_quantiles": {str(q): float(np.quantile(brake_v, q))
                                               for q in (0.1, 0.5, 0.9)},
                 "frac_targets_brake_above_0p01c": float(np.mean(brake_v > 0.01 * C_KMS))},
    })


if __name__ == "__main__":
    main()
