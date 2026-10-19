"""Experiment 2 -- moving stars: how much does waiting for a close pass help?

Same question as Experiment 1, but the stars move on their Gaia DR3 velocities
and a colony may wait at its star until a neighbour drifts within range. For
each hop range r we run one exact earliest-arrival Dijkstra over the
time-dependent graph up to t_max; the arrival times give the reachable
fraction at every horizon T <= t_max at once. T = 0 is the frozen-star
(static) answer from the Euclidean MST.

Outputs results/timedep.json: reach[r][T], r_q(T) and the relative range
reduction vs frozen, arrival-time quantiles (front speed), settle-time and
ship-speed sweeps, catalogue-boundary diagnostic, and bootstrap CIs.
"""
from __future__ import annotations

import time

import numpy as np

from src.analysis.graph import euclidean_mst, minimax_from_source
from src.analysis.temporal import TemporalGraph, candidate_pairs, crossing_r
from src.experiments.common import setup, write
from src.model.kinematics import C_PCMYR, phase_space

QS = [0.1, 0.5, 0.9]


def reach_table(pos, vel, ev, r_grid, horizons, v_ship, t_max, settle, pairs=None):
    """frac[i_r, i_T] reachable among eval stars; T=0 means frozen stars."""
    if pairs is None:
        pairs = candidate_pairs(pos, vel, float(max(r_grid)), t_max)
    bott, _, _ = minimax_from_source(euclidean_mst(pos), 0)
    frac = np.zeros((len(r_grid), len(horizons)))
    arrivals = {}
    for a, r in enumerate(r_grid):
        tg = TemporalGraph(pos, vel, pairs, r, v_ship, t_max)
        arr = tg.earliest_arrival(0, settle=settle)
        arrivals[r] = arr
        for b, T in enumerate(horizons):
            if T == 0:
                frac[a, b] = np.mean(bott[ev] <= r)               # frozen stars
            else:
                frac[a, b] = np.mean(arr[ev] <= T)
    return frac, arrivals, len(pairs)


def r_of_T(r_grid, frac, q):
    return [crossing_r(r_grid, frac[:, b], q) for b in range(frac.shape[1])]


def main():
    cfg, cat, rng = setup("timedep")
    tc = cfg["timedep"]
    pos, vel = phase_space(cat, rng)
    dist = np.linalg.norm(pos, axis=1)
    R = tc["eval_radius_pc"]
    ev = (dist <= R) & (dist > 0)
    r_grid = np.array(tc["r_grid"], dtype=float)
    H = [float(h) for h in tc["horizons_myr"]]
    T = float(tc["t_max_myr"])
    v_ship = tc["v_ship_c"] * C_PCMYR

    t0 = time.time()
    pairs = candidate_pairs(pos, vel, float(r_grid.max()), T)
    t_pairs = time.time() - t0
    frac, arrivals, n_pairs = reach_table(pos, vel, ev, r_grid, H, v_ship, T,
                                          tc["settle_myr"], pairs)
    print(f"[timedep] {n_pairs} candidate pairs ({t_pairs:.1f}s), "
          f"total {time.time() - t0:.1f}s")

    rq = {str(q): r_of_T(r_grid, frac, q) for q in QS}
    r50_0 = rq["0.5"][0]
    reduction = [1 - x / r50_0 if np.isfinite(x) else float("nan") for x in rq["0.5"]]

    # Arrival-time quantiles and effective expansion speed for sub-threshold r.
    front = []
    for r in r_grid:
        arr = arrivals[r][ev]
        tq = {str(q): float(np.quantile(arr, q)) if np.mean(np.isfinite(arr)) >= q else None
              for q in (0.1, 0.25, 0.5)}
        t50 = tq["0.5"]
        v_eff = R * 0.5 ** (1 / 3) / t50 if t50 else None            # pc/Myr
        front.append({"r": float(r), "t_q": tq, "v_eff_pc_myr": v_eff,
                      "frac_by_tmax": float(np.mean(np.isfinite(arr)))})

    # Settle-time sweep (colony rebuild delay) and ship-speed sweep at fixed r.
    r_fix = float(tc.get("sweep_r_pc", 1.5))
    settle = []
    for s in tc["settle_sweep_myr"]:
        tg = TemporalGraph(pos, vel, pairs, r_fix, v_ship, T)
        arr = tg.earliest_arrival(0, settle=s)[ev]
        settle.append({"settle_myr": s, "frac": [float(np.mean(arr <= h)) for h in H]})
    speeds = []
    for vc in tc.get("v_sweep_c", [0.001, 0.01, 0.1]):
        tg = TemporalGraph(pos, vel, pairs, r_fix, vc * C_PCMYR, T)
        arr = tg.earliest_arrival(0, settle=tc["settle_myr"])[ev]
        speeds.append({"v_c": vc, "frac": [float(np.mean(arr <= h)) for h in H]})

    # Boundary diagnostic: catalogue stars inside rho at T vs at 0 (stars that
    # would enter from beyond the catalogue radius are missing).
    retention = []
    for rho in (R, 2 * R):
        n0 = np.sum(np.linalg.norm(pos, axis=1) <= rho)
        retention.append({"rho_pc": rho, "ratio": [
            float(np.sum(np.linalg.norm(pos + vel * h, axis=1) <= rho) / n0) for h in H]})

    # Astrometric bootstrap: full recomputation per draw.
    boots = []
    for b in range(tc["n_boot"]):
        rb = np.random.default_rng(2000 + b)
        pb, vb = phase_space(cat, rb, perturb=True)
        db = np.linalg.norm(pb, axis=1)
        eb = (db <= R) & (db > 0)
        fb, _, _ = reach_table(pb, vb, eb, r_grid, H, v_ship, T, tc["settle_myr"])
        boots.append(r_of_T(r_grid, fb, 0.5))
    boots = np.array(boots)
    ci = [[float(np.nanquantile(boots[:, k], 0.025)), float(np.nanquantile(boots[:, k], 0.975))]
          for k in range(len(H))] if len(boots) else []
    red_boot = (1 - boots / boots[:, :1]) if len(boots) else np.zeros((0, len(H)))
    red_ci = [[float(np.nanquantile(red_boot[:, k], 0.025)), float(np.nanquantile(red_boot[:, k], 0.975))]
              for k in range(len(H))] if len(boots) else []

    write(cfg, "timedep", {
        "eval_radius_pc": R, "n_eval": int(ev.sum()), "t_max_myr": T,
        "v_ship_c": tc["v_ship_c"], "settle_myr": tc["settle_myr"],
        "n_candidate_pairs": int(n_pairs),
        "r_grid": r_grid.tolist(), "horizons_myr": H,
        "reach_frac": frac.tolist(),
        "r_q_of_T": rq,
        "r50_reduction": reduction,
        "r50_ci95": ci, "r50_reduction_ci95": red_ci, "n_boot": int(tc["n_boot"]),
        "front": front,
        "sweep_r_pc": r_fix,
        "settle_sweep": settle,
        "speed_sweep": speeds,
        "retention": retention,
    })


if __name__ == "__main__":
    main()
