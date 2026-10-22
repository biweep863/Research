"""Experiment 3: the discrete-event simulator against the closed forms.

M/M/c (Erlang C) and M/D/1 / Kingman over a utilisation grid, and a
fixed-time signal against Webster and the HCM delay.  This is what licenses
using the formulas (and the CTM, tested separately against the fluid queue)
for the corridor.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from src.analysis.capacity import hcm_delay, webster_delay
from src.analysis.fluid import fluid_queue
from src.analysis.queues import kingman_wq, md1, mmc
from src.experiments.common import ctm_params, dump, load_cfg
from src.sim.ctm import Carriageway, active_bottlenecks, simulate
from src.sim.des import multi_server_fifo, signal_queue
from utils.stats import bootstrap_ci


def main():
    cfg, out = load_cfg()
    v = cfg["validation"]
    rng = np.random.default_rng(cfg["seed"])
    mu, n = v["mu"], v["n_vehicles"]
    rows = []
    for c in v["servers_grid"]:
        for rho in v["rho_grid"]:
            lam = rho * c * mu
            for svc in ("exp", "det"):
                sims = [multi_server_fifo(lam, mu, c, n, rng, svc)["Wq"] * 3600 for _ in range(v["reps"])]
                m, lo, hi = bootstrap_ci(sims, seed=cfg["seed"])
                if svc == "exp":
                    theory = mmc(lam, mu, c)["Wq"] * 3600
                    name = "Erlang C"
                elif c == 1:
                    theory = md1(lam, mu)["Wq"] * 3600
                    name = "M/D/1"
                else:
                    # Kingman-style scaling of M/M/c by (ca2 + cs2) / 2 (Allen-Cunneen).
                    theory = mmc(lam, mu, c)["Wq"] * 3600 * 0.5
                    name = "Allen-Cunneen"
                rows.append({"c": c, "rho": rho, "service": svc, "sim_s": m, "lo": lo, "hi": hi,
                             "theory_s": theory, "formula": name,
                             "kingman_s": kingman_wq(lam / c, mu, 1.0, 1.0 if svc == "exp" else 0.0) * 3600
                             if c == 1 else None,
                             "rel_err": (m - theory) / theory})
    sig = []
    for x in (0.3, 0.5, 0.7, 0.85, 0.95):
        g, C, s = 0.5, 90.0, 1900.0
        cap = s * g
        q = x * cap
        sims = [signal_queue(q, s, C, g, n, rng) for _ in range(v["reps"])]
        m, lo, hi = bootstrap_ci(sims, seed=cfg["seed"])
        sig.append({"x": x, "sim_s": m, "lo": lo, "hi": hi,
                    "webster_s": webster_delay(C, g, q, cap),          # with Webster's 0.9 correction
                    "webster_raw_s": webster_delay(C, g, q, cap) / 0.9,
                    "hcm_s": hcm_delay(C, g, q, cap, period_h=n / q)})

    # CTM vs fluid point queue at a single bottleneck (spatial queue vs vertical queue).
    # Newell's equivalence holds for the plain triangular CTM: no capacity drop.
    p = dataclasses.replace(ctm_params(cfg), cap_drop=0.0)
    ctm_fluid = []
    for excess in (1.05, 1.15, 1.3):
        cap = np.full(60, 1900.0 * 3)
        cap[45] = 1900.0 * 2
        cw = Carriageway(lanes=np.full(60, 3.0), cap_vph=cap, beta=np.zeros(60), ramp_cell=np.array([0]),
                         ramp_cap_vph=np.array([5700.0]), ramp_lanes=np.array([3.0]))
        T = int(4 * 3600 / p.dt_s)
        t = (np.arange(T) + 0.5) * p.dt_h
        lam = np.where(t < 1.0, excess * 3800.0, 2400.0)
        r = simulate(cw, lam[:, None], p)
        f = fluid_queue(lam, 3800.0, p.dt_h)
        ab = active_bottlenecks(r["density"], p.kc_vpkm)
        ctm_fluid.append({"excess": excess, "ctm_vh": r["delay_total_vh"], "fluid_vh": f["total_delay_vh"],
                          "rel_err": (r["delay_total_vh"] - f["total_delay_vh"]) / f["total_delay_vh"],
                          "bottleneck_cell": int(np.argmax(ab))})

    errs = [abs(r["rel_err"]) for r in rows]
    res = {"queues": rows, "signal": sig, "ctm_fluid": ctm_fluid,
           "mean_abs_rel_err": float(np.mean(errs)), "max_abs_rel_err": float(np.max(errs)),
           "signal_mean_abs_rel_err_webster": float(np.mean([abs(s["sim_s"] - s["webster_s"]) / s["webster_s"]
                                                             for s in sig])),
           "ctm_fluid_mean_abs_rel_err": float(np.mean([abs(c["rel_err"]) for c in ctm_fluid]))}
    dump(out, "validation.json", res)
    print(f"[validation] queues mean |err| {res['mean_abs_rel_err']:.1%} (max {res['max_abs_rel_err']:.1%}); "
          f"signal vs Webster {res['signal_mean_abs_rel_err_webster']:.1%}; CTM vs fluid {res['ctm_fluid_mean_abs_rel_err']:.1%}")


if __name__ == "__main__":
    main()
