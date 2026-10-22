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


def load_inventory(cfg: dict, log=print) -> dict:
    """OSM corridor inventory (downloaded once, cached) or the synthetic fallback,
    with optional field counts attached."""
    from src.data.corridor import build_corridor
    from src.data.counts import apply_field_counts
    from src.data.osm import load_osm
    from src.data.synthetic import synthetic_corridor

    root = project_root()
    osm = load_osm(cfg["osm"], root / cfg["osm"]["cache"], log=log)
    if osm is None:
        log("[data] no OSM data available -> synthetic corridor (synthetic_data: true)")
        inv = synthetic_corridor(cfg.get("seed", 0))
    else:
        inv = build_corridor(osm, cfg["corridor"], cfg["osm"]["spine_lonlat"])
        inv["synthetic_data"] = False
        inv["osm_source"] = osm.get("source")
        inv["osm_fetched"] = osm.get("fetched")
    side_green = 1 - cfg["control"]["g_main"][1] - cfg["control"]["lost_frac"]
    n = apply_field_counts(inv, root / cfg["demand"]["field_counts"], side_green)
    inv.setdefault("field_counts_applied", n)
    return inv


def ctm_params(cfg: dict):
    from src.sim.ctm import CTMParams
    return CTMParams(**{k: cfg["ctm"][k] for k in ("vf_kmh", "w_kmh", "kj_vpkm", "dt_s")},
                     fifo=cfg["ctm"]["fifo"][1], cap_drop=cfg["ctm"]["cap_drop"][1])


def demand_matrix(cfg: dict, base_vph, p, scale: float = 1.0, peak_len_h: float | None = None):
    """(T, R) ramp demand: trapezoidal peak profile times each ramp's peak demand."""
    from src.data.network import peak_profile
    c = cfg["ctm"]
    T = int(round(c["horizon_h"] * 3600 / p.dt_s))
    prof = peak_profile(T, p.dt_h, c["peak_start_h"], c["peak_len_h"] if peak_len_h is None else peak_len_h,
                        c["ramp_h"], cfg["demand"]["peak_factor_offpeak"])
    return prof[:, None] * np.asarray(base_vph, float)[None, :] * scale
