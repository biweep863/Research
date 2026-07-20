# 04 · Results

Headline numbers (local config; all emitted as macros in `paper/macros.tex`).

## Lossless acceleration
- Largest scene (1600 blades, 6400 triangles): brute **31,987,980** tests →
  BVH **237,949** (**134×**), instancing **215,277** (**149×**). Speedup grows
  with scene size (BVH cost ~logarithmic in primitives).
- On the rendered scene (200 blades): BVH **41×**, instancing **47×** cheaper
  than brute — at a **bit-identical** image (SSIM = 1, verified to 1e-9).
- Geometry memory (instanced vs duplicated): **3×** here, because the toy blade
  is only 4 triangles; the ratio grows toward the blade's triangle count for
  realistic vegetation.

## Level of detail (lossy)
- Single-triangle simplification of all far blades: SSIM floor **0.973**.
- Flat billboard: SSIM floor **0.949**.
- Cost only falls to **~80%** of full-detail — LOD barely moves intersection
  cost once a BVH is present (its win is memory / front-end, not traversal).

## Sample budget + denoising (lossy)
- Monte-Carlo RMSE slope **−0.43** (theory −0.5).
- Guided denoiser at 1 spp: SSIM **0.708 → 0.747** (+3.9 pts); neutral by 16 spp.

## Cost–quality frontier (headline)
- Lossless methods form a wall at SSIM 1, moving cost from 1.0 to ~2.4% of brute.
- Lossy methods extend the frontier by another order of magnitude, bending down
  in quality; low-spp+denoise Pareto-dominates LOD on this scene.
- **Full stack** (instancing + LOD + 4 spp + denoise): SSIM **0.908** at
  **277×** less cost than the brute-force reference.

## Interpretation
The priority order is legible: take the free lossless savings first (and they
grow with scene size), then spend samples and denoise, and reach for LOD only
when geometry — not noise — is the bottleneck. The opportunity cost of each
shortcut is the vertical SSIM drop you accept per horizontal step left on the
frontier. See `paper/main.pdf`.
