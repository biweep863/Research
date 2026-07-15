"""Space-based vs ground solar comparison.

Writes results/sbsp.json: the orbital collection+transmission chain, per-m^2
delivered power for ground vs orbit, capacity factors, and the collector area
each would need to meet world electricity demand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import yaml

from utils.seeds import set_seed
from src.model.sbsp import compare_ground_vs_space, sbsp_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    chain = [{"label": s.label, "power": s.power} for s in sbsp_chain()]
    c = compare_ground_vs_space()
    result = {
        "chain": chain,
        "ground_per_m2_W": c.ground_per_m2,
        "space_per_m2_W": c.space_per_m2,
        "ratio": c.ratio,
        "ground_capacity_factor": c.ground_capacity_factor,
        "space_capacity_factor": c.space_capacity_factor,
        "ground_area_for_world_m2": c.ground_area_for_world,
        "space_area_for_world_m2": c.space_area_for_world,
    }

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sbsp.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[sbsp] wrote {path}")


if __name__ == "__main__":
    main()
