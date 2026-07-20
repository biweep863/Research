"""Level-of-detail experiment (lossy).

Beyond a distance threshold, blades collapse to a cheaper form (a single
triangle, or a flat billboard). Lower threshold => cheaper => more error. We
render each setting at a fixed sample count (so Monte-Carlo noise is identical
across settings and only the geometry differs) and measure cost vs SSIM/PSNR
against the full-detail image. Writes results/lod.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import yaml

from utils.seeds import set_seed
from src.render.scene import assign_lod, bake_field, build_scene
from src.render.bvh import BVH
from src.render.tracer import count_primary, render
from src.quality.metrics import psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    w, h = cfg["render"]["width"], cfg["render"]["height"]
    leaf = cfg["accel"]["bvh_leaf_size"]
    spp = cfg["lod"]["spp"]
    n = cfg["render"]["scene_blades"]

    rng = np.random.default_rng(cfg.get("seed", 0))
    scene = build_scene(n_blades=n, n_spheres=cfg["render"]["spheres"],
                        width=w, height=h, rng=rng)

    full = bake_field(scene, scene.modes)
    full_tris = len(full)
    ref_img, _ = render(scene, BVH(full, leaf), spp=spp, seed=1)
    full_cost = count_primary(scene, BVH(full, leaf)).total()

    modes = {}
    for far_mode in cfg["lod"]["modes"]:
        rows = []
        for thr in cfg["lod"]["thresholds"]:
            lod = assign_lod(scene, thr, far_mode)
            baked = bake_field(scene, lod)
            bvh = BVH(baked, leaf)
            img, _ = render(scene, bvh, spp=spp, seed=1)
            cost = count_primary(scene, bvh).total()
            rows.append({
                "threshold": thr,
                "triangles": len(baked),
                "tri_frac": len(baked) / full_tris,
                "cost_tests": cost,
                "cost_frac": cost / full_cost,
                "ssim": ssim(img, ref_img),
                "psnr": psnr(img, ref_img),
                "n_simplified": int(sum(1 for m in lod if m != "full")),
            })
            print(f"[lod] {far_mode:10s} thr={thr:.2f} tris={len(baked):5d} "
                  f"cost={cost:>8,} ssim={rows[-1]['ssim']:.3f}")
        modes[far_mode] = rows

    out = {"width": w, "height": h, "spp": spp, "full_triangles": full_tris,
           "full_cost_tests": full_cost, "modes": modes}
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lod.json").write_text(json.dumps(out, indent=2))
    print(f"[lod] wrote {out_dir / 'lod.json'}")


if __name__ == "__main__":
    main()
