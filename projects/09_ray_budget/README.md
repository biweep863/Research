# Project 9 — Ray Tracing on a Budget: the Zelda grass trick, measured

## Question
Ray tracing is expensive because every ray is tested, independently, against the
scene's geometry — which is why photoreal rendering wants very capable GPUs.
Games don't pay that bill in full: *Breath of the Wild* did not treat every
blade of grass as unique geometry; it replicated one blade and instanced it, and
faded distant detail to a cheaper form. This project builds a from-scratch path
tracer and measures four ways to trace fewer rays against an **exact reference
image**, so each technique's speedup *and its quality price* are quantified.

## The four techniques
1. **BVH** — a bounding-volume hierarchy. Lossless: identical image, orders of
   magnitude fewer intersection tests than brute force.
2. **Instancing** (the Zelda grass trick) — one canonical blade, many transforms,
   a two-level BVH. Lossless: identical image, scene memory and build cost cut ~K×.
3. **Level of detail / impostors** — distant blades collapse to a single triangle
   or a billboard. Lossy: cheaper, but the image now differs — a measured curve.
4. **Sample budget + denoising** — fewer samples per pixel (Monte-Carlo noise
   falls as 1/√N), rescued by a normal/depth-guided bilateral denoiser. Lossy,
   measured.

The headline is the **cost–quality Pareto frontier**: lossless methods move the
frontier for free; lossy methods buy further speedups at a price you can read off.

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/09_ray_budget
make test           # intersections vs closed forms; BVH == brute; instancing == duplicated
make run            # accel, LOD, sampling, and Pareto experiments -> results/*.json
make figures        # regenerate figures + paper/macros.tex
make paper          # compile paper/main.pdf
```

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
Larger images, more samples, bigger grass fields, finer sweeps — no code changes,
everything is config-gated. (Pure numpy, CPU-only: no GPU needed.)

## Files
- `src/render/geometry.py` — ray/sphere, ray/triangle (Möller–Trumbore), ray/AABB + a test counter
- `src/render/scene.py` — procedural grass field, spheres, sun, camera, LOD variants
- `src/render/bvh.py` — median-split BVH and two-level BVH for instances
- `src/render/tracer.py` — numpy path tracer (primary + shadow + 1 bounce) and G-buffer
- `src/quality/metrics.py` — PSNR, SSIM, and a guided bilateral denoiser (numpy)
- `src/experiments/run_{accel,lod,sampling,pareto}.py` — write `results/*.json`
- `src/figures/plot_budget.py` — 5 figures + LaTeX macros
- `paper/main.tex` — two-column paper
- `doc/` — methodology, experiments log, decisions

## Headline
Lossless structure (BVH + instancing) buys 100×+ fewer intersection tests at a
*bit-identical* image — the grass trick is not an approximation, it is free. Only
after that do you trade quality for speed, and the LOD and sample-budget curves
say exactly how much you pay and where.
