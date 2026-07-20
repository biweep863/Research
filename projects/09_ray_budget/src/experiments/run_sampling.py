"""Sample-budget + denoising experiment (lossy).

Fewer samples per pixel means cheaper renders but more Monte-Carlo noise, which
falls as 1/sqrt(N). A normal/depth-guided bilateral denoiser buys back quality
at low sample counts -- the minimal idea behind production denoisers. We sweep
samples per pixel, measure error/SSIM/PSNR against a high-sample reference with
and without denoising, and fit the convergence slope. Writes results/sampling.json.
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
from src.render.scene import bake_field, build_scene
from src.render.bvh import BVH
from src.render.tracer import render
from src.quality.metrics import bilateral_denoise, psnr, ssim


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
    baked = bake_field(scene, scene.modes)
    bvh = BVH(baked, leaf)

    ref, ref_counter = render(scene, bvh, spp=ref_spp, seed=1)

    rows = []
    for spp in cfg["sampling"]["spp_grid"]:
        img, counter, gbuf = render(scene, bvh, spp=spp, seed=2, gbuffer=True)
        den = bilateral_denoise(img, gbuf, sigma_spatial=ss, sigma_color=sc)
        rmse = float(np.sqrt(np.mean((img - ref) ** 2)))
        rows.append({
            "spp": spp,
            "tests": counter.total(),
            "rmse": rmse,
            "ssim": ssim(img, ref),
            "psnr": psnr(img, ref),
            "ssim_denoised": ssim(den, ref),
            "psnr_denoised": psnr(den, ref),
        })
        print(f"[sampling] spp={spp:3d} tests={counter.total():>9,} "
              f"rmse={rmse:.4f} ssim={rows[-1]['ssim']:.3f} "
              f"ssim_dn={rows[-1]['ssim_denoised']:.3f}")

    spps = np.array([r["spp"] for r in rows], dtype=float)
    rmses = np.array([r["rmse"] for r in rows], dtype=float)
    slope = float(np.polyfit(np.log(spps), np.log(rmses), 1)[0])

    out = {"width": w, "height": h, "reference_spp": ref_spp,
           "reference_tests": ref_counter.total(), "mc_slope": slope,
           "sigma_spatial": ss, "sigma_color": sc, "rows": rows}
    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sampling.json").write_text(json.dumps(out, indent=2))
    print(f"[sampling] mc_slope={slope:.3f}  wrote {out_dir / 'sampling.json'}")


if __name__ == "__main__":
    main()
