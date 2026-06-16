# Project 1 — Embedding Geometry & Information Collapse in Long Contexts

## Question
Does the residual-stream manifold of a decoder-only transformer undergo a
measurable phase transition as the context length grows?

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/01_embedding_geometry
make test           # sanity-check the metric implementations
make run            # run the geometry sweep (GPT-2 by default)
make figures        # regenerate figures + paper/macros.tex
make paper          # compile paper/main.pdf
```

## Scaling up
Switch to a dedicated GPU by passing the GPU config:
```bash
make run     CONFIG=configs/gpu_full.yaml
make figures CONFIG=configs/gpu_full.yaml
```
No code changes are needed — context lengths up to 131,072 tokens and
Llama-3-8B / Mistral-7B are gated entirely behind the config.

## Files
- `src/data/load_long_docs.py` — corpus loader (HF or local fallback)
- `src/analysis/{intrinsic_dimension,anisotropy,curvature}.py` — metrics
- `src/experiments/run_geometry_sweep.py` — main entry point
- `src/figures/plot_geometry.py` — figures + LaTeX macros
- `paper/main.tex` — two-column scientific paper
- `doc/` — step-by-step methodology, experiments log, decisions
