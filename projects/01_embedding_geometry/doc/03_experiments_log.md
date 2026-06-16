# 03 · Experiments Log

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-06-15 | `local_small.yaml` | `python -m src.experiments.run_geometry_sweep --config configs/local_small.yaml` | OK; 8 docs × 4 ctx × 13 layers, ≈90 s total on M-series CPU. Output: `results/sweep_gpt2.json`. |
| 2026-06-15 | — | `python -m src.figures.plot_geometry --results results --out paper/figures --macros paper/macros.tex` | OK; 5 PDF figures + 7 macros generated. |
| 2026-06-15 | — | `latexmk -pdf -interaction=nonstopmode paper/main.tex` | OK; 431 KB `paper/main.pdf`, two-column, all refs resolved. |

## Observations
- GPT-2 middle layers show strong drop in TwoNN from ~8.6 at ctx=128 to ~0.9
  at ctx=1024 — interpretable as severe collapse but very small model. The
  GPU configuration (Llama-3-8B / Mistral-7B) is needed to test whether the
  trajectory is qualitatively the same on production-scale models.
- Effective rank drop is negative in the recorded value because the metric is
  reported as `er[short] − er[long]` and the long-context cloud has more
  effective directions (more tokens contribute to the covariance). This is
  consistent: TwoNN measures local geometry, effective rank measures global
  variance distribution.
