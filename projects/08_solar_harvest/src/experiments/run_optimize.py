"""Ecosystem-aware light-budget optimisation experiment (headline).

Builds the light supply L[cell, hour] from the insolation model, weights
harvest value by a diurnal electricity-demand curve, and compares three
policies across a sweep of ecosystem reserve fractions f_min:

  * no-cap  -- harvest everything (ecosystem-blind upper bound),
  * flat    -- reserve the same fraction every hour (naive but feasible),
  * optimal -- water-filling: reserve the ecosystem's quota from the
               lowest-demand hours (feasible AND maximal).

Writes results/optimize.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import yaml

from utils.seeds import set_seed
from src.model.insolation import diurnal_light_profile
from src.optimize.light_budget import (
    flat_harvest,
    min_light_from_fraction,
    no_cap_harvest,
    optimal_harvest,
)


def demand_weights(n_hours: int) -> np.ndarray:
    """Diurnal net-demand ("duck curve", mean 1): morning and evening peaks
    with a deep midday trough where solar supply is already abundant. This is
    the value of *harvesting* in each hour -- high at the peaks, low midday."""
    h = np.linspace(0, 24, n_hours, endpoint=False)
    w = (1.0
         + 0.55 * np.exp(-((h - 8.0) / 1.8) ** 2)
         + 0.95 * np.exp(-((h - 19.0) / 2.2) ** 2)
         - 0.55 * np.exp(-((h - 13.0) / 2.5) ** 2))
    w = np.clip(w, 0.05, None)
    return w / w.mean()


def build_light(cells_lat, doy, n_hours) -> np.ndarray:
    return np.array([diurnal_light_profile(lat, doy, n_hours) for lat in cells_lat])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))
    oc = cfg["optimize"]

    cells_lat = oc["cells_lat"]
    n_hours = oc["n_hours"]
    doy = oc["representative_doy"]
    L = build_light(cells_lat, doy, n_hours)          # W/m^2, shape (cells, hours)
    w = np.tile(demand_weights(n_hours), (len(cells_lat), 1))

    blind = no_cap_harvest(L, w)                       # upper bound (kills ecosystem)

    sweep = []
    for f_min in oc["f_min_grid"]:
        d_min = min_light_from_fraction(L, f_min)
        opt = optimal_harvest(L, w, d_min)
        flat = flat_harvest(L, w, f_min)
        sweep.append({
            "f_min": f_min,
            "optimal_value": opt.harvest_value,
            "flat_value": flat.harvest_value,
            "blind_value": blind.harvest_value,
            # fraction of the ecosystem-blind harvest value retained
            "optimal_retained": opt.harvest_value / blind.harvest_value,
            "flat_retained": flat.harvest_value / blind.harvest_value,
            # extra value the temporal schedule unlocks over the flat policy
            "temporal_gain": (opt.harvest_value - flat.harvest_value) / flat.harvest_value,
            "reserved_energy": opt.reserved_energy,
        })

    # Per-cell accept schedule at the headline f_min (for the heatmap figure).
    f_head = oc["f_min_headline"]
    head = optimal_harvest(L, w, min_light_from_fraction(L, f_head))

    result = {
        "cells_lat": cells_lat,
        "n_hours": n_hours,
        "representative_doy": doy,
        "f_min_headline": f_head,
        "demand_weights": demand_weights(n_hours).tolist(),
        "sweep": sweep,
        "headline_accept": head.accept.tolist(),
        "light_supply": L.tolist(),
    }

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "optimize.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[optimize] wrote {path}")


if __name__ == "__main__":
    main()
