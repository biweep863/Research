# 03 · Experiments log

All runs use `configs/local_small.yaml` (seed 0) unless noted. Times are on a
laptop CPU; the machine-independent numbers are the intersection-test counts.

## `run_accel` (~12 s)
Cast primary + shadow rays per pixel over grass fields of 50…1600 blades under
brute force, BVH, and instancing. Confirms brute grows linearly in triangle
count; BVH and instancing stay far flatter. At 1600 blades (6400 triangles):
brute 31,987,980 tests, BVH 237,949 (134×), instancing 215,277 (149×).

## `run_lod` (~70 s)
Render the field at 4 distance thresholds × 2 far-modes (simplified, billboard)
at fixed spp, SSIM/PSNR vs the full-detail image. Simplified holds SSIM ≥ 0.973
to full-collapse; billboard falls to 0.949. Cost drops only to ~80% of full —
LOD is a weak lever on intersection cost once a BVH exists (see decisions).

## `run_sampling` (~44 s)
Sweep spp ∈ {1,2,4,8,16}, RMSE/SSIM/PSNR vs a 24-spp reference, with and without
the guided denoiser. MC slope fit = −0.43 (theory −0.5). Denoiser lifts SSIM at
1 spp from 0.708 to 0.747 and helps less as spp rises (at 16 spp it is neutral).

## `run_pareto` (~62 s)
Assemble all methods on one cost–quality plane with the unified cost currency.
Lossless: BVH rel-cost 0.024 (SSIM 1), instancing 0.021 (SSIM 1) — i.e.
~41–47× cheaper than brute at identical image on the rendered scene. Lossy points
extend to rel-cost ~1e-3. Full stack: SSIM 0.908 at rel-cost 0.0036 (≈ 277×
cheaper than brute).

## Figures + macros (`plot_budget`, ~11 s)
Re-renders the scene for the beauty/cost-map figure and writes 5 figures + 22
macros to `paper/macros.tex`. Every quoted number in the paper flows from these
macros, so re-running the pipeline rewrites the prose.

## Total
`make all` ≈ 3 min end to end (run + figures + paper). `make test` ≈ 13 s.
