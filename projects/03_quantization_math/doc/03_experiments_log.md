# 03 · Experiments Log

| Date | Config | Command | Notes |
|------|--------|---------|-------|
| 2026-06-15 | `local_small.yaml` (distilgpt2) | `make eval` | First validation of pipeline. Baseline accuracy ~0 (no math ability), but logit MSE scaled cleanly across bit-widths: FP16 ≈ 7e-5, INT8 ≈ 3.3, INT4 ≈ 5.1e2, ternary ≈ 1.4e3. Confirmed quantizer correctness. |
| 2026-06-15 | `local_small.yaml` (Qwen2.5-0.5B, 16 items) | `make eval` | Used as the published local run with a model that actually does arithmetic. See `results/math_eval_qwen05b.json` for raw values. |
| 2026-06-15 | `local_small.yaml` | `make sens` | Submodule sensitivity at 4-bit. See `results/sensitivity_qwen05b.json`. |
| 2026-06-15 | — | `make figures && make paper` | Figures + macros + LaTeX build. |
