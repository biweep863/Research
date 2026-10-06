"""Shared helpers for the experiment entry points (same pattern as project 14)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from utils.seeds import set_seed  # noqa: E402


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg() -> tuple[dict, Path]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sanity", action="store_true", help="tiny end-to-end run")
    ap.add_argument("--only", default=None, help="comma-separated run names (train_sweep)")
    args, _ = ap.parse_known_args()
    cfg = yaml.safe_load(open(args.config))
    if args.sanity or cfg.get("sanity"):
        cfg["sanity"] = True
        _merge(cfg, cfg.get("sanity_overrides", {}))
        cfg["output_dir"] = str(Path(cfg.get("output_dir", "results")) / "sanity")
    set_seed(cfg.get("seed", 0), deterministic=cfg.get("deterministic", False))
    cfg["_only"] = args.only.split(",") if args.only else None
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return cfg, out_dir


def resolve_device(cfg: dict) -> str:
    dev = cfg.get("device", "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        print("[common] device=cuda requested but CUDA is unavailable; falling back to cpu")
        return "cpu"
    return dev


def data_root(cfg: dict) -> Path:
    return HERE / cfg["data"]["root"]


def cache_tag(cfg: dict) -> str:
    return "sanity" if cfg.get("sanity") else "full" if cfg["rooms"]["n_train"] >= 1000 else "local"


def dump(out_dir: Path, name: str, obj) -> None:
    (out_dir / name).write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"[{name.split('.')[0]}] wrote {out_dir / name}")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(type(o))
