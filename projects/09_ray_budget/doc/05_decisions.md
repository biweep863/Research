# 05 · Decisions & Alternatives

## A from-scratch numpy tracer instead of a real engine
We chose a self-contained path tracer over scripting Blender/Cycles or a GPU
renderer. Rationale: it gives *computable ground truth* — the BVH image is
provably identical to brute force, instancing is provably identical to
duplicated geometry, and both are unit-tested to floating point. It reproduces on
a clean machine in minutes, and it lets us report a machine-independent cost
(intersection tests) instead of noisy wall-clock times. The price is realism,
addressed honestly in the limitations.

## Cost is counted, not timed
Wall-clock time on pure-numpy CPU is meaningless for graphics conclusions, and it
would vary by machine. Counting ray/primitive intersection tests is exact,
deterministic, and portable — it is the quantity acceleration structures actually
reduce. We keep timings in the logs for sanity but never quote them as results.

## The honest LOD finding
We expected LOD to be a strong cost lever and it is not — once a BVH exists,
halving triangle count barely changes traversal cost (most tests are boxes). We
did not hide this: we report the weak cost saving and explain it (LOD's real win
is memory / rasterisation front-end, and it matters when geometry, not sampling,
is the bottleneck). This is the most interesting result precisely because it was
counterintuitive.

## Instancing memory ratio is small — and we say why
With a deliberately tiny 4-triangle blade, a per-instance transform (12 floats)
is not much smaller than the duplicated geometry (36 floats), so the memory ratio
is only ~3×. We keep the toy blade for speed and explain that the ratio scales
toward the blade's triangle count — for the hundreds-of-triangles blades Nintendo
actually drew, instancing is the difference between prohibitive and trivial.

## One bounce, diffuse only
The tracer does direct sun + one diffuse bounce. That is enough to exhibit both
geometric cost (what BVH/instancing attack) and Monte-Carlo noise (what
sampling/denoising attack) while keeping `make all` at ~3 minutes. Reflection,
refraction, and deep paths are natural extensions gated by the config.

## Unified Pareto cost currency
The frontier needs one x-axis spanning geometry and sampling. We use
`tests_per_pass × spp`, normalised so brute at reference spp = 1. Lossless points
are placed analytically at SSIM = 1 (the image is identical by construction, so
rendering brute at full spp — which is slow — is unnecessary). This is documented
in the paper's method section so the mixed nature of the axis is explicit.
