# 00 · Overview

**Question.** Ray tracing is expensive because every ray is tested against the
scene's geometry. Games (famously *Breath of the Wild*'s grass) don't pay that
bill in full: they instance one blade thousands of times and fade distant detail.
This project builds a from-scratch path tracer and *measures* four ways to trace
fewer rays against an exact reference, so each technique's speedup and its
quality price are quantified.

**Four techniques.**
1. BVH — lossless spatial acceleration.
2. Instancing (the Zelda grass trick) — lossless, geometry stored once.
3. Level of detail / billboards — lossy, distant blades simplified.
4. Sample budget + guided denoising — lossy, fewer Monte-Carlo samples.

**Headline.** Lossless structure (BVH + instancing) buys 40–150× fewer
intersection tests (growing with scene size) at a *bit-identical* image; only
after that do LOD and sample budgeting trade measurable SSIM for further
speedups. The full stack reaches SSIM ~0.91 at ~277× less cost than brute force.

**Cost currency.** Machine-independent ray/primitive intersection tests, counted
by the renderer — not wall-clock time.

**Reproduce.**
```bash
cd projects/09_ray_budget
make test      # ~13 s: intersections vs closed forms, BVH==brute, instancing==duplicated
make all       # ~3 min: experiments -> results/*.json, 5 figures, macros, paper/main.pdf
```

See `01_dataset.md` (the procedural scene), `02_methodology.md` (derivations and
the cost model), `03_experiments_log.md`, `04_results.md`, `05_decisions.md`.
