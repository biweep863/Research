"""Synthetic stand-in for the OSM inventory (no network / tests).

A 10 km corridor with 3 express + 2-3 lateral lanes per direction, junctions
every ~1.2 km and minor/service accesses as a Poisson process along each side.
Same dictionary layout as corridor.build_corridor, flagged synthetic.
"""
from __future__ import annotations

import numpy as np

from src.data.corridor import BIN_M


def synthetic_corridor(seed: int = 0, length_m: float = 10000.0) -> dict:
    rng = np.random.default_rng(seed)
    bins = np.arange(0.0, length_m, BIN_M)
    lanes = {}
    for d in ("WB", "EB"):
        lat = np.where(rng.random(len(bins)) < 0.3, 3.0, 2.0)
        lanes[d] = {"express": np.full(len(bins), 3.0), "lateral": lat}

    junctions, approaches, departures, signals = [], [], [], []
    jc = np.arange(400.0, length_m, 1200.0)
    for k, c in enumerate(jc):
        junctions.append({"chainage": float(c), "c0": float(c), "c1": float(c),
                          "name": f"Junction {k + 1}", "n_major": 2})
        for d in ("WB", "EB"):
            signals.append({"node": f"s{k}{d}", "dir": d, "chainage": float(c), "on": ["lateral"]})
            for kind, hw, ln in (("crossing", "secondary", 2), ("terminal", "trunk_link", 1)):
                a = _rec(f"j{k}{d}{kind}", f"Junction {k + 1}", hw, "major", ln, c, d, kind, "lateral", True)
                approaches.append(a)
                departures.append(dict(a))
    for d in ("WB", "EB"):
        for dclass, hw, rate in (("minor", "residential", 3.0), ("service", "service", 5.0), ("tertiary", "tertiary", 0.5)):
            n = rng.poisson(rate * length_m / 1000)
            for i, c in enumerate(np.sort(rng.uniform(0, length_m, n))):
                a = _rec(f"{hw}{d}{i}", f"{hw} {i}" if hw != "service" else "", hw, dclass,
                         2 if hw == "tertiary" else 1, c, d, "terminal", "lateral", False)
                approaches.append(a)
                departures.append(dict(a))
    from src.data.corridor import _cluster_accesses
    return {"chainage_start_m": 0.0, "chainage_end_m": length_m, "length_km": length_m / 1000,
            "bins_m": bins, "lanes": lanes, "signals": signals, "approaches": approaches,
            "departures": departures, "accesses": _cluster_accesses(approaches, 60),
            "junctions": junctions, "internal_links": 0, "corridor_ways": 0,
            "corridor_lanes_tag_coverage": 1.0, "synthetic_data": True}


def _rec(way, name, hw, dclass, lanes, c, d, kind, side, signal):
    return {"way": way, "node": way, "name": name, "highway": hw, "dclass": dclass, "lanes": lanes,
            "lanes_tagged": True, "chainage": float(c), "dirs": [d], "dir": d, "side": side,
            "kind": kind, "service": "", "signal": signal}
