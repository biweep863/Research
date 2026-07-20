# 02 · Methodology

## Renderer (`src/render/tracer.py`)
A minimal Monte-Carlo path tracer:
1. **Primary ray** per pixel, jittered inside the pixel for anti-aliasing.
2. **Direct light**: Lambert shading from the sun, gated by a **shadow ray**.
3. **One indirect bounce**: a cosine-weighted hemisphere sample (Malley's
   method); with a Lambert BRDF and cosine-pdf the estimator reduces to
   `albedo × L_incoming`, where `L_incoming` is the second hit's direct light or
   the sky.
Averaging `N` samples/pixel (spp) cuts variance as `N^{-1/2}`.

## Intersections (`src/render/geometry.py`), all closed form
- **Ray/sphere**: quadratic, nearest root `t > ε`.
- **Ray/triangle**: Möller–Trumbore.
- **Ray/plane**: `t = (h − o_y)/d_y`.
- **Ray/AABB**: slab test, used by BVH nodes.
Every routine increments a `Counter` split by primitive kind — the cost currency.

## Acceleration (`src/render/bvh.py`)
- **BVH**: median split on the axis of largest centroid extent; leaves hold
  ≤ `leaf_size` triangles. Traversal tests a node's box before its subtree and
  prunes any box farther than the current best hit.
- **Two-level (instancing)**: a BLAS BVH over the canonical blade; a TLAS BVH
  over instance world-space boxes. A candidate instance transforms the ray into
  local space with `A^{-1}` (translation subtracted; direction **not**
  renormalised, so `t` is preserved) and intersects the shared blade. World
  normals use `(A^{-1})^T`. This is provably identical to duplicated geometry
  (unit-tested to floating point).

## Cost model
- **Acceleration study**: tests to cast one primary + one shadow ray per pixel
  (`count_primary`), isolating structure from sampling noise.
- **Pareto study**: `cost(method, N) = tests_1pass × N`, one currency spanning
  geometric and sampling cost, normalised so brute force at reference spp = 1.
  Lossless methods are placed at their measured `count_primary × ref_spp` with
  SSIM = 1 (identical image, so no render needed); lossy methods are rendered and
  scored.

## Metrics + denoiser (`src/quality/metrics.py`)
- **PSNR** (capped at 100 for identical images) and **windowed SSIM**, numpy.
- **Guided bilateral denoiser**: cross-bilateral weights = spatial × colour ×
  normal-agreement^8 × depth, so noise is smoothed within a surface but not
  across silhouettes. G-buffer (albedo/normal/depth) comes from the deterministic
  centre ray.

## Verification (`tests/test_model.py`, 10 tests)
Closed-form intersections vs hand-built cases; BVH nearest-hit == brute (and
never more triangle tests); instancing image == duplicated geometry to 1e-9;
Monte-Carlo RMSE slope in [−0.75, −0.25]; SSIM/PSNR properties; denoiser does not
reduce SSIM. `make test` must be green before experiments.
