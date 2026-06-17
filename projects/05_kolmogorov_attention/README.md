# Project 5 — Kolmogorov Complexity vs Attention Density

## Question
Does the gzip-based Kolmogorov-complexity proxy of a prompt predict
the per-layer attention density of a transformer LM?

## Quick start
```bash
source ../../.venv/bin/activate
make test
make all       # ~30 s on CPU for GPT-2
```

## Files
- `src/data/prompts.py` — 7 controlled prompt families
- `src/analysis/complexity.py` — gzip proxy + attention entropy / Gini / sparsity
- `src/experiments/run_complexity.py` — main sweep
- `src/figures/plot_complexity.py` — figures + macros
- `paper/main.pdf` — two-column paper
