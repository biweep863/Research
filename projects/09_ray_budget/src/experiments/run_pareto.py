"""The cost-quality Pareto frontier (headline).

Every technique is placed on one plane: cost (ray/primitive tests) vs quality
(SSIM against the high-sample reference). One cost currency is used throughout,

    cost(accel, spp) = tests(primary + shadow, one pass) x samples-per-pixel,

so lossless and lossy methods are directly comparable. Lossless methods (BVH,
instancing) sit at SSIM = 1 and move the frontier for free; lossy methods (LOD,
low samples, denoising) trade measurable quality for further cost cuts. The full
stack combines them. Writes results/pareto.json.
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
from src.render.bvh import BVH, BruteForce, TwoLevelBVH
from src.render.tracer import count_primary, render
from src.quality.metrics import bilateral_denoise, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    w, h = cfg["render"]["width"], cfg["render"]["height"]
    leaf = cfg["accel"]["bvh_leaf_size"]
    ref_spp = cfg["render"]["reference_spp"]
    n = cfg["render"]["scene_blades"]
    ss = cfg["sampling"]["denoise_sigma_spatial"]
    sc = cfg["sampling"]["denoise_sigma_color"]

    rng = np.random.default_rng(cfg.get("seed", 0))
    scene = build_scene(n_blades=n, n_spheres=cfg["render"]["spheres"],
                        width=w, height=h, rng=rng)
    full = bake_field(scene, scene.modes)

    brute = BruteForce(full)
    bvh = BVH(full, leaf)
    two = TwoLevelBVH(scene.blade, scene.instances, leaf)

    cp_brute = count_primary(scene, brute).total()
    cp_bvh = count_primary(scene, bvh).total()
    cp_two = count_primary(scene, two).total()

    ref, _ = render(scene, bvh, spp=ref_spp, seed=1)
    base_cost = cp_brute * ref_spp        # brute force at full quality = 1.0

    def cost(cp, spp):
        return cp * spp / base_cost

    pts = []

    def add(label, cls, cp, spp, quality):
        pts.append({"label": label, "class": cls, "rel_cost": cost(cp, spp),
                    "tests": int(cp * spp), "spp": spp, "ssim": quality})
        print(f"[pareto] {label:26s} rel_cost={cost(cp, spp):.4e} ssim={quality:.3f}")

    # Lossless: identical image to the reference (SSIM = 1 by construction).
    add("brute force", "lossless", cp_brute, ref_spp, 1.0)
    add("BVH", "lossless", cp_bvh, ref_spp, 1.0)
    add("BVH + instancing", "lossless", cp_two, ref_spp, 1.0)

    # Sample budget (BVH), with and without the guided denoiser.
    for spp in [1, 4, 16]:
        img, _, gbuf = render(scene, bvh, spp=spp, seed=2, gbuffer=True)
        add(f"BVH, {spp} spp", "sampling", cp_bvh, spp, ssim(img, ref))
        den = bilateral_denoise(img, gbuf, sigma_spatial=ss, sigma_color=sc)
        add(f"BVH, {spp} spp + denoise", "denoise", cp_bvh, spp, ssim(den, ref))

    # Level of detail (BVH over simplified geometry) at a fixed sample count.
    lod_spp = cfg["lod"]["spp"]
    for thr in [0.6, 0.3, 0.0]:
        lod = assign_lod(scene, thr, "simplified")
        baked = bake_field(scene, lod)
        lbvh = BVH(baked, leaf)
        img, _ = render(scene, lbvh, spp=lod_spp, seed=1)
        cp_lod = count_primary(scene, lbvh).total()
        add(f"LOD thr={thr}", "lod", cp_lod, lod_spp, ssim(img, ref))

    # Full stack: instancing memory + LOD geometry + few samples + denoise.
    lod = assign_lod(scene, 0.3, "simplified")
    baked = bake_field(scene, lod)
    lbvh = BVH(baked, leaf)
    img, _, gbuf = render(scene, lbvh, spp=4, seed=2, gbuffer=True)
    den = bilateral_denoise(img, gbuf, sigma_spatial=ss, sigma_color=sc)
    cp_lod = count_primary(scene, lbvh).total()
    add("full stack", "stack", cp_lod, 4, ssim(den, ref))

    out = {
        "width": w, "height": h, "reference_spp": ref_spp, "blades": n,
        "cp_brute": cp_brute, "cp_bvh": cp_bvh, "cp_instanced": cp_two,
        "lossless_speedup_bvh": cp_brute / cp_bvh,
        "lossless_speedup_instanced": cp_brute / cp_two,
        "points": pts,
    }
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pareto.json").write_text(json.dumps(out, indent=2))
    print(f"[pareto] BVH speedup={out['lossless_speedup_bvh']:.1f}x  "
          f"wrote {out_dir / 'pareto.json'}")


if __name__ == "__main__":
    main()
