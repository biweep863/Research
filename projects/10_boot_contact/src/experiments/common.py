"""Shared helpers for the experiment entry points."""
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


def load_cfg() -> tuple[dict, Path]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, out_dir


def dump(out_dir: Path, name: str, obj: dict) -> None:
    (out_dir / name).write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"[{name.split('.')[0]}] wrote {out_dir / name}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def project_root() -> Path:
    return HERE
