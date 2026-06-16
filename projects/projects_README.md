# Research Program

Three LLM research projects, each with a runnable pipeline, real
experimental results, figures, two-column scientific paper, and full
methodology documentation.

## Layout
```
projects/
├── shared/                      env, utils, LaTeX preamble
├── 01_embedding_geometry/       Geometry of long-context residual stream
├── 02_mechanistic_logic/        Surface-form-invariant logic representations
└── 03_quantization_math/        Bit-width scaling laws for math reasoning
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
- `configs/local_small.yaml` — runs on Mac CPU/MPS in minutes
- `configs/gpu_full.yaml` — Llama-3-8B / Pythia-2.8B / Qwen-Math-7B, full
  context lengths, large eval sets

Switch with:
```bash
make all CONFIG=configs/gpu_full.yaml
```
No code changes required — all model choices, dataset sizes, context
lengths, and bit-widths are config-driven.
