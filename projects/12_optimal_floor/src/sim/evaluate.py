"""Place a resident on floor h, run the building, and measure their trips."""
from __future__ import annotations

import numpy as np

from ..analytic.closed_form import expected_elevator_time
from ..model.kinematics import Kinematics
from ..model.stairs import Stairs
from ..model.trips import TripGenerator
from .elevator import Building, Passenger, poisson_background


def simulate_resident(home: int, n_floors: int, alpha: float, kin: Kinematics, parking: str,
                      rate_per_min: float, n_cars: int, horizon_s: float, trip_interval_s: float,
                      rng: np.random.Generator, stairs: Stairs | None = None, alpha_bg: float | None = None,
                      capacity: int = 10, warmup_s: float = 600.0) -> dict:
    alpha_bg = alpha if alpha_bg is None else alpha_bg
    b = Building(n_floors, n_cars, kin, parking=parking, capacity=capacity)
    for p in poisson_background(rate_per_min, horizon_s, n_floors, alpha_bg, rng):
        b.add_passenger(p)
    gen = TripGenerator(n_floors, home, alpha, rng)
    records = []
    car_pos_at_call = []
    t = warmup_s + rng.uniform(0, trip_interval_s)
    pid = 10_000_000
    while t < horizon_s:
        b.run(t)
        o, d, kind = gen.next()
        te = expected_elevator_time(o, d, kin, parking, n_floors)
        if stairs is not None and stairs.take_stairs(o, d, te):
            dt = stairs.time(o, d)
            records.append({"t": t, "o": o, "d": d, "kind": kind, "mode": "stairs",
                            "wait": 0.0, "ride": dt, "total": dt, "stops": 0})
            t_done = t + dt
        else:
            car_pos_at_call.append(list(b.car_positions()))
            p = Passenger(pid, t, o, d, resident=True, kind=kind); pid += 1
            b.add_passenger(p)
            # advance until the resident alights (bounded)
            t_lim = t + 1800.0
            while np.isnan(p.t_alight) and b.events and b.events[0][0] <= t_lim:
                b.run(b.events[0][0])
            if np.isnan(p.t_alight):
                raise RuntimeError("resident stranded: check dispatch logic")
            records.append({"t": t, "o": o, "d": d, "kind": kind, "mode": "elevator",
                            "wait": p.t_board - t, "ride": p.t_alight - p.t_board,
                            "total": p.t_alight - t, "stops": p.stops_on_board})
            t_done = p.t_alight
        t = max(t + trip_interval_s + rng.uniform(-0.2, 0.2) * trip_interval_s, t_done + 5.0)
    b.run(horizon_s)
    tot = np.array([r["total"] for r in records])
    elev = [r for r in records if r["mode"] == "elevator"]
    return {
        "home": home, "n_trips": len(records), "mean": float(tot.mean()), "p90": float(np.quantile(tot, 0.9)),
        "std": float(tot.std()), "mean_wait": float(np.mean([r["wait"] for r in elev])) if elev else 0.0,
        "mean_ride": float(np.mean([r["ride"] for r in elev])) if elev else 0.0,
        "mean_stops": float(np.mean([r["stops"] for r in elev])) if elev else 0.0,
        "frac_walked": float(np.mean([r["mode"] == "stairs" for r in records])),
        "car_pos_at_call": car_pos_at_call, "totals": tot.tolist(),
        "bg_served": len([p for p in b.done if not p.resident]),
    }
