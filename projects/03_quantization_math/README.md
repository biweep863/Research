# Project 3 — Quantization Robustness on Math Reasoning

## Question
How does residual error in non-linear function approximation by an
LLM scale with parameter bit-width, and which submodules contribute most
to that error?

## Quick start
```bash
source ../../.venv/bin/activate
make test       # quantizer unit tests
make eval       # full sweep {fp32,fp16,int8,int4,ternary}
make sens       # submodule (attn vs mlp) sensitivity at 4-bit
make figures    # figures + LaTeX macros
make paper      # paper/main.pdf
```

## Scaling
```bash
make all CONFIG=configs/gpu_full.yaml  # TinyLlama-1.1B + Qwen2.5-Math-7B + GSM8K + MATH
```

## Files
- `src/data/load_math.py` — synthetic arithmetic + GSM8K loader
- `src/experiments/quantize_models.py` — per-channel symmetric quantizer
- `src/experiments/run_math_eval.py` — sweep + logit MSE/KL
- `src/experiments/run_layerwise_sensitivity.py` — attn vs mlp
- `src/figures/plot_quantization.py` — figures + macros
- `paper/main.pdf` — two-column scientific paper
- `doc/` — methodology, experiments log, decisions
