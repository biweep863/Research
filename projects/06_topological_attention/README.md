# Project 6 — Topological Persistence in Attention

## Question
Do truthful and hallucinated continuations of a factual prompt give
statistically distinguishable persistent-homology summaries of the
per-layer attention graph?

## Quick start
```bash
source ../../.venv/bin/activate
make test
make all     # ~2 s on CPU for the 10-pair pilot
```

## Scaling
```bash
make all CONFIG=configs/gpu_full.yaml   # Llama-3-8B + TruthfulQA, N=200 pairs
```

## Files
- `src/analysis/topology.py` — Vietoris–Rips persistence + Betti curves (ripser)
- `src/experiments/run_tda.py` — sweep
- `src/figures/plot_tda.py` — figures + macros
- `paper/main.pdf` — pilot paper with explicit power-analysis
