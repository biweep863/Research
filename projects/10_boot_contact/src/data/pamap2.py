"""Real ankle-IMU background from PAMAP2 (Reiss & Stricker, 2012; UCI ML repo).

PAMAP2 has a Colibri wireless IMU strapped to the dominant ankle sampled at
100 Hz, with a +-16 g accelerometer and a gyroscope, during walking (4),
running (5) and -- in the optional protocol of subjects 108 and 109 -- playing
soccer (20).  We use it only as *background*: real foot motion under which a
synthetic contact is injected.  There are no labelled fouls in PAMAP2 (nor in
any public dataset we know of).

Loading is two-stage so the pipeline reproduces offline:
1. If the raw ``PAMAP2_Dataset`` folder is present (``data/PAMAP2_Dataset`` or
   ``$PAMAP2_DIR``), extract the ankle columns for the three activities, drop
   NaN rows, cap each (subject, activity) at ``cap_s`` seconds, and write a
   small ``data/pamap2_ankle.npz`` cache (checked in, a few MB).
2. Otherwise load the cache.  If neither exists return ``None`` and the
   experiments fall back to the synthetic gait, saying so in the results.

Column layout of the .dat files (1-indexed, from the dataset readme): 1 time,
2 activity id, 3 heart rate, 4-20 hand IMU, 21-37 chest IMU, 38-54 ankle IMU.
Within an IMU block: 1 temperature, 2-4 acc +-16 g (m/s^2), 5-7 acc +-6 g,
8-10 gyro (rad/s), 11-13 magnetometer, 14-17 orientation (invalid).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

G = 9.80665
ACTIVITIES = {4: "walking", 5: "running", 20: "soccer"}
SUBJECTS = [101, 105, 106, 108, 109]
ANKLE_ACC16 = slice(38, 41)     # 0-indexed columns 38,39,40
ANKLE_GYRO = slice(44, 47)


def _raw_dir(project_root: Path) -> Path | None:
    for cand in [os.environ.get("PAMAP2_DIR"), project_root / "data" / "PAMAP2_Dataset"]:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _extract(raw: Path, cap_s: float = 120.0) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for subj in SUBJECTS:
        for part in ("Protocol", "Optional"):
            f = raw / part / f"subject{subj}.dat"
            if not f.exists():
                continue
            arr = np.loadtxt(f, dtype=np.float32)
            act = arr[:, 1].astype(int)
            for aid, name in ACTIVITIES.items():
                m = act == aid
                if m.sum() < 1000:
                    continue
                blk = np.concatenate([arr[m][:, ANKLE_ACC16] / G,
                                      np.rad2deg(arr[m][:, ANKLE_GYRO])], axis=1)
                blk = blk[~np.isnan(blk).any(axis=1)]
                blk = blk[: int(cap_s * 100)]
                key = f"s{subj}_{name}"
                if key in out:
                    out[key] = np.concatenate([out[key], blk])[: int(cap_s * 100)]
                else:
                    out[key] = blk
    return out


def load_pamap2_ankle(project_root: Path, cap_s: float = 120.0,
                      activities: tuple[str, ...] = ("walking", "running", "soccer")
                      ) -> list[np.ndarray] | None:
    """Return a list of (T, 6) float32 arrays [ax ay az (g), gx gy gz (deg/s)]
    at 100 Hz, one per (subject, activity), or None if unavailable."""
    cache = project_root / "data" / "pamap2_ankle.npz"
    if not cache.exists():
        raw = _raw_dir(project_root)
        if raw is None:
            return None
        segs = _extract(raw, cap_s)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **segs)
    z = np.load(cache)
    segs = [z[k] for k in z.files if any(k.endswith(a) for a in activities) and len(z[k]) > 200]
    return segs or None
