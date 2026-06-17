# Research Program (Six Papers)

Six LLM research projects, each with a runnable pipeline, real
experimental results, figures, two-column scientific paper, and full
methodology documentation.

## Layout
```
projects/
├── shared/                       env, utils, LaTeX preamble
├── 01_embedding_geometry/        Geometry of long-context residual stream
├── 02_mechanistic_logic/         Surface-form-invariant logic representations
├── 03_quantization_math/         Bit-width scaling laws for math reasoning
├── 04_neural_ode_depth/          Continuous-depth transformers via Neural ODE
├── 05_kolmogorov_attention/      Gzip complexity ↔ attention density
└── 06_topological_attention/     Persistent homology of attention graphs
```

## Setup
```bash
cd <repo root>
bash projects/shared/env/setup.sh
source .venv/bin/activate
```

## Per-project workflow
```bash
cd projects/<NN_name>
make test     # unit tests
make all      # run experiments, build figures, compile paper
```

## Scaling to a real GPU
Every project ships two configs:
- `configs/local_small.yaml` — runs on Mac CPU/MPS in seconds-to-minutes.
- `configs/gpu_full.yaml`     — Llama-3-8B / Mistral-7B / Pythia-2.8B /
  Qwen-Math-7B at full context length and dataset size.

Switch with:
```bash
make all CONFIG=configs/gpu_full.yaml
```
No code changes required — all model choices, dataset sizes, context
lengths, bit-widths, ODE solver budgets, and TDA filtration grids are
config-driven.

## Headline findings (local pilot)
| # | Topic | Headline |
|---|-------|----------|
| 1 | Embedding geometry | GPT-2 mid-layer intrinsic dim drops 8.61→0.89 across 128→1024 ctx |
| 2 | Logic interpretability | English probe 1.000, conlang 0.959, cross-surface 0.67 |
| 3 | Quantization math | Qwen-0.5B accuracy 0.69→0.31→0.0 at FP32→INT4→ternary; MLP 6× more fragile than attn |
| 4 | Neural ODE depth | ODE loss range 0.05 vs discrete 22.9 across solver budgets |
| 5 | Kolmogorov attention | R²=0.68 at layer 5 (entropy ~ gzip ratio) |
| 6 | Topological attention | Pipeline validated; pilot underpowered (p=0.85, N=10); GPU run sized for N≥200 |
