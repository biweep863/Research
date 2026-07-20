"""Acceleration-scaling experiment (lossless methods).

For a range of grass-field sizes, count the ray/primitive tests needed to cast
one primary + one shadow ray per pixel under three schemes:
  * brute force           -- O(rays x triangles),
  * BVH                   -- O(rays x log triangles),
  * BVH + instancing      -- geometry stored once, rays transformed per instance.
All three produce the identical image (verified in tests); only the cost differs.
Writes results/accel.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import yaml

from utils.seeds import set_seed
from src.render.scene import bake_field, build_scene
from src.render.bvh import BVH, BruteForce, TwoLevelBVH
from src.render.tracer import count_primary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    w, h = cfg["render"]["width"], cfg["render"]["height"]
    leaf = cfg["accel"]["bvh_leaf_size"]
    counts = cfg["accel"]["blade_counts"]

    rows = []
    for n in counts:
        rng = np.random.default_rng(cfg.get("seed", 0))
        scene = build_scene(n_blades=n, n_spheres=cfg["render"]["spheres"],
                            width=w, height=h, rng=rng)
        baked = bake_field(scene, scene.modes)
        n_tris = len(baked)

        brute = BruteForce(baked)
        bvh = BVH(baked, leaf_size=leaf)
        two = TwoLevelBVH(scene.blade, scene.instances, leaf_size=leaf)

        t0 = time.time(); cb = count_primary(scene, brute, shadow=True); tb = time.time() - t0
        t0 = time.time(); cv = count_primary(scene, bvh, shadow=True); tv = time.time() - t0
        t0 = time.time(); ci = count_primary(scene, two, shadow=True); ti = time.time() - t0

        rows.append({
            "blades": n, "triangles": n_tris,
            "brute": {"tests": cb.total(), "tri": cb.tri, "time_s": tb},
            "bvh": {"tests": cv.total(), "tri": cv.tri, "box": cv.box,
                    "nodes": bvh.n_boxes, "time_s": tv},
            "instanced": {"tests": ci.total(), "tri": ci.tri, "box": ci.box,
                          "time_s": ti},
            # Scene memory: duplicated triangles vs one blade + K transforms.
            "mem_duplicated_floats": n_tris * 3 * 3,
            "mem_instanced_floats": len(scene.blade) * 3 * 3 + n * 12,
        })
        print(f"[accel] blades={n:5d} tris={n_tris:6d} "
              f"brute={cb.total():>10,} bvh={cv.total():>9,} inst={ci.total():>9,}")

    out = {"width": w, "height": h, "leaf_size": leaf, "rows": rows}
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "accel.json").write_text(json.dumps(out, indent=2))
    print(f"[accel] wrote {out_dir / 'accel.json'}")


if __name__ == "__main__":
    main()
