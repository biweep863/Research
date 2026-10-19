"""Shared setup for the experiment entry points: config, catalogue, output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from utils.seeds import set_seed  # noqa: E402
from src.data.gaia import load_catalog  # noqa: E402


def setup(name: str):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))
    dc = cfg["data"]
    cat = load_catalog(dc["radius_pc"], dc["poe_min"], dc["cache_dir"])
    rng = np.random.default_rng(cfg.get("seed", 0))
    print(f"[{name}] catalogue: {len(cat['ra'])} stars "
          f"({cat['has_rv'].mean():.0%} with RV, {cat['is_host'].sum()} exoplanet hosts)")
    return cfg, cat, rng


def write(cfg, name: str, result: dict):
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"[{name}] wrote {path}")
