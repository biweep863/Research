"""Solar-resource + land-feasibility experiment.

Writes results/resource.json:
  * total intercepted power and the TOA->surface loss cascade (ozone broken out),
  * annual-mean surface GHI per latitude band,
  * deployable-land energy vs world demand (cities / countries).
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
from src.model.insolation import (
    S0,
    band_surface_ghi,
    global_annual_mean_toa,
    global_mean_surface_ghi,
    loss_cascade,
    toa_intercepted_power,
)
from src.model import landuse
from src.model.landuse import land_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    n_doy = cfg["resource"]["doy_samples"]
    bands = cfg["resource"]["lat_bands"]
    band_ghi = [{"lat": b, "ghi": band_surface_ghi(b, n_doy)} for b in bands]

    cascade = [{"label": lv.label, "irradiance": lv.irradiance} for lv in loss_cascade()]
    lb = land_budget()

    result = {
        "solar_constant": S0,
        "intercepted_power_W": toa_intercepted_power(),
        "toa_horizontal_mean_W_m2": global_annual_mean_toa(),
        "surface_ghi_mean_W_m2": global_mean_surface_ghi(),
        "cascade": cascade,
        "band_ghi": band_ghi,
        "land": {
            "land_area_m2": lb.land_area,
            "deployable_area_m2": lb.deployable_area,
            "deployable_power_W": lb.deployable_power,
            "frac_of_primary": lb.frac_of_primary,
            "frac_of_electricity": lb.frac_of_electricity,
            "area_for_world_electricity_m2": lb.area_for_world_electricity,
            "n_million_cities": lb.n_million_cities,
            "n_country_equivalents": lb.n_country_equivalents,
            "world_electricity_W": landuse.WORLD_ELECTRICITY_POWER,
            "world_primary_W": landuse.WORLD_PRIMARY_POWER,
        },
    }

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resource.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[resource] wrote {path}")


if __name__ == "__main__":
    main()
