"""Closed-form expected trip time for one car and no other traffic.

For a trip (o, d) the car must first come from where it is parked, then
carry the resident:

    T(o, d | c) = run(|c - o|) + stop + run(|o - d|)

where c is the car's position when called and the trip is counted from the
button press to the car's arrival at the destination (one dwell -- boarding
at the origin -- lies inside it; the door opening at the destination is not
counted, on either side of the comparison with the simulator).  Parking policies give c:
  lobby  -> c = 1                 (car returns to the ground floor when idle)
  middle -> c = round((N+1)/2)
  last   -> c = o                 (the car stayed where it last stopped, and
                                   with no other traffic that is where the
                                   resident got off -- the wait is one door cycle)
  random -> c ~ Uniform{1..N}     (the heavy-traffic limit: the car could be anywhere)

E[T(h)] sums over the exact (o, d) distribution of chained trips.  With
stairs, each trip costs min(elevator, stairs) when the climb is allowed.
"""
from __future__ import annotations

import numpy as np

from ..model.kinematics import Kinematics
from ..model.stairs import Stairs
from ..model.trips import od_distribution

PARKING = ("lobby", "middle", "last", "random")


def car_position_pmf(policy: str, n_floors: int, origin: int) -> np.ndarray:
    """pmf over floors 1..N of the car's position when the resident calls."""
    p = np.zeros(n_floors)
    if policy == "lobby":
        p[0] = 1.0
    elif policy == "middle":
        p[int(round((n_floors + 1) / 2)) - 1] = 1.0
    elif policy == "last":
        p[origin - 1] = 1.0
    elif policy == "random":
        p[:] = 1.0 / n_floors
    else:
        raise ValueError(policy)
    return p


def expected_elevator_time(origin: int, dest: int, kin: Kinematics, policy: str, n_floors: int) -> float:
    pmf = car_position_pmf(policy, n_floors, origin)
    floors = np.arange(1, n_floors + 1)
    wait = float(np.sum(pmf * kin.run(floors - origin)))
    ride = kin.stop() + float(kin.run(abs(dest - origin)))
    return wait + ride


def expected_trip_time(home: int, n_floors: int, alpha: float, kin: Kinematics, policy: str,
                       stairs: Stairs | None = None) -> dict:
    """E[T(h)] and its decomposition; also the fraction of trips walked."""
    total, walked, elev_only = 0.0, 0.0, 0.0
    for o, d, p in od_distribution(n_floors, home, alpha):
        te = expected_elevator_time(o, d, kin, policy, n_floors)
        elev_only += p * te
        if stairs is not None and stairs.take_stairs(o, d, te):
            total += p * stairs.time(o, d); walked += p
        else:
            total += p * te
    return {"mean": total, "mean_elevator_only": elev_only, "frac_walked": walked}


def curve(n_floors: int, alpha: float, kin: Kinematics, policy: str, stairs: Stairs | None = None,
          homes: list[int] | None = None) -> dict:
    homes = homes or list(range(2, n_floors + 1))
    means = np.array([expected_trip_time(h, n_floors, alpha, kin, policy, stairs)["mean"] for h in homes])
    return {"homes": homes, "mean": means, **regret_stats(homes, means, n_floors)}


def regret_stats(homes, means, n_floors: int) -> dict:
    means = np.asarray(means, dtype=float)
    i = int(np.argmin(means))
    best = float(means[i])
    mid = int(round((n_floors + 1) / 2))
    j_mid = homes.index(mid) if mid in homes else int(np.argmin(np.abs(np.array(homes) - mid)))
    band = [h for h, m in zip(homes, means) if m <= 1.05 * best]
    return {
        "h_star": int(homes[i]), "best": best,
        "regret_middle_s": float(means[j_mid] - best), "regret_middle_pct": float(100 * (means[j_mid] / best - 1)),
        "regret_worst_s": float(means.max() - best), "regret_worst_pct": float(100 * (means.max() / best - 1)),
        "regret_top_pct": float(100 * (means[-1] / best - 1)),
        "band_5pct": [int(band[0]), int(band[-1])], "band_5pct_width": len(band),
    }
