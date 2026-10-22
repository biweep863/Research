"""Download + cache the Gaia DR3 catalogue and exoplanet hosts; write a summary.

Writes results/catalog.json (counts and the nearest stars, as a sanity check
that alpha Cen / Barnard's star / Wolf 359 come out where they should)."""
from __future__ import annotations

import numpy as np

from src.experiments.common import setup, write
from src.model.kinematics import phase_space


def main():
    cfg, cat, rng = setup("catalog")
    pos, vel = phase_space(cat, rng)
    dist = np.linalg.norm(pos[1:], axis=1)
    speed_kms = np.linalg.norm(vel[1:], axis=1) / 1.0227121650537077
    near = np.argsort(dist)[:12]
    write(cfg, "catalog", {
        "radius_pc": cfg["data"]["radius_pc"],
        "n_stars": int(len(dist)),
        "n_with_rv": int(cat["has_rv"].sum()),
        "n_hosts": int(cat["is_host"].sum()),
        "n_supplement": int((cat["source_id"] < 0).sum()),
        "speed_kms_median": float(np.median(speed_kms)),
        "speed_kms_p99": float(np.quantile(speed_kms, 0.99)),
        "nearest": [{"source_id": int(cat["source_id"][k]), "dist_pc": float(dist[k]),
                     "G": float(cat["phot_g_mean_mag"][k])} for k in near],
    })


if __name__ == "__main__":
    main()
