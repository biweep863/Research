"""Field counts that replace the assumed demand of an intersection.

field_counts.csv (project root, versioned), one row per counted movement onto
the corridor:

    name,dir,veh_min_green,veh_min_red,green_frac,free_lanes,note
    Avenida Alfonso Reyes,WB,14-18,7-10,,1,two lanes; one continuous right turn

* name       street name as in OSM (see results/inventory.json)
* dir        corridor direction it feeds: WB = towards San Pedro, EB = towards Garza Sada
* veh_min_green / veh_min_red
             vehicles per minute entering the corridor while the cross street
             has green / red; a range "lo-hi" or a single number
* green_frac share of the cycle the cross street has green; empty = the model's
             side green (1 - g_main - lost)
* free_lanes lanes not stopped by the signal (continuous right turn, slip lane)
* veh_per_h  alternatively, one hourly figure (range or number) instead of the two rates

A count covers the whole intersection: the named approaches plus unnamed ramps
of the same direction within `radius_m`, so nothing is counted twice.  The
observed range becomes [lo, hi] veh/h; the base run uses the middle and the
Monte Carlo draws inside it.  Counts measure what entered, which is the demand
only if no queue stays on the cross street after green.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _rng(v: str) -> tuple[float, float] | None:
    v = (v or "").strip()
    if not v:
        return None
    if "-" in v[1:]:
        lo, hi = v.split("-", 1) if v[0] != "-" else (v, v)
        return float(lo), float(hi)
    return float(v), float(v)


def hourly_range(row: dict, side_green: float) -> tuple[float, float]:
    """(lo, hi) veh/h from a CSV row."""
    if _rng(row.get("veh_per_h")):
        return _rng(row["veh_per_h"])
    g = float(row["green_frac"]) if (row.get("green_frac") or "").strip() else side_green
    gr, rd = _rng(row["veh_min_green"]), _rng(row["veh_min_red"]) or (0.0, 0.0)
    return (60 * (gr[0] * g + rd[0] * (1 - g)), 60 * (gr[1] * g + rd[1] * (1 - g)))


def apply_field_counts(inv: dict, path: Path, side_green: float = 0.35, radius_m: float = 80.0) -> int:
    """Attach observed ranges to the approaches of each counted intersection."""
    inv["field_counts"] = []
    if not path.exists():
        return 0
    applied = 0
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("name") or "").strip() and not r["name"].startswith("#")]
    for row in rows:
        d = row["dir"].strip().upper()
        nm = row["name"].strip().lower()
        named = [a for a in inv["approaches"] if a["name"].strip().lower() == nm and a["dir"] in (d, "both")]
        if not named:
            continue
        ch = [a["chainage"] for a in named]
        match = named + [a for a in inv["approaches"]
                         if not a["name"] and not a.get("uturn") and a["dir"] in (d, "both") and a not in named
                         and min(abs(a["chainage"] - c) for c in ch) <= radius_m]
        lo, hi = hourly_range(row, side_green)
        free = float(row.get("free_lanes") or 0)
        lanes = sum(a["lanes"] for a in match)
        for a in match:
            w = a["lanes"] / lanes
            a["observed_vph"] = (lo * w, hi * w)
            a["free_lanes"] = free * w
        inv["field_counts"].append({"name": row["name"], "dir": d, "vph_lo": lo, "vph_hi": hi,
                                    "free_lanes": free, "approaches": len(match),
                                    "members": [[a["way"], a["node"]] for a in match],
                                    "chainage_m": sum(ch) / len(ch), "note": row.get("note", "")})
        applied += 1
    inv["field_counts_applied"] = applied
    return applied
