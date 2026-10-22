from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def setup(desc: str):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="configs/local_small.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    out = ROOT / cfg.get("results_dir", "results")
    out.mkdir(parents=True, exist_ok=True)
    return cfg, out


def work_dir(cfg) -> Path:
    p = ROOT / cfg["analysis"]["work_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def write(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=1, default=conv))
    print(f"wrote {path}")


def circ_diff(a: float, b: float) -> float:
    return float(abs((a - b + 180) % 360 - 180))
