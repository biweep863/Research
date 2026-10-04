from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from ..data.simulate import HAS_PRA, pink_noise, simulate_scene, voiced_speech


def parse_config(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/local_small.yaml")
    if extra:
        extra(ap)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    return args, cfg


def write_json(cfg: dict, name: str, payload: dict) -> Path:
    out = Path(cfg.get("output_dir", "results")) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"simulator": "pyroomacoustics" if HAS_PRA else "free-field", **payload}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return out


def talker_clip(
    cfg: dict, az_deg, dist_m, rt60, seed, f0=None, snr_db=None, noise_dbfs=None,
    level_jitter_db=0.0, fan=False,
):
    """Lead silence (noise-model warm-up) + one utterance, simulated at the array.

    level_jitter_db: uniform +-jitter of vocal effort (people do not all speak
    at the same level - this is what breaks a pure inverse-square range cue).
    fan: add a directional pink-noise source (``sim.fan``) at the same power as
    the diffuse noise.
    """
    fs = int(cfg["audio"]["fs"])
    sim = cfg["sim"]
    rng = np.random.default_rng(seed)
    f0 = f0 or float(rng.uniform(100, 220))
    sig = voiced_speech(sim["utterance_s"], fs, f0=f0, seed=seed)
    sig = np.concatenate([np.zeros(int(sim["lead_silence_s"] * fs)), sig])
    level = sim["source_level_dbfs_1m"] + float(rng.uniform(-level_jitter_db, level_jitter_db))
    src = {"signal": sig, "az_deg": az_deg, "dist_m": dist_m, "level_dbfs_1m": level}
    fan_src = None
    if fan:
        fc = sim["fan"]
        fan_src = {"signal": pink_noise(len(sig), rng), "az_deg": fc["az_deg"], "dist_m": fc["dist_m"],
                   "height": fc["height"], "level_dbfs_1m": 0.0}
    return simulate_scene(cfg, [src], rt60=rt60, snr_db=snr_db, noise_dbfs=noise_dbfs, fan=fan_src, seed=seed)


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
