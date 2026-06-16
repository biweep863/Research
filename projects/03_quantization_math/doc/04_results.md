# 04 · Results

See `paper/main.pdf` for the formal write-up with figures. Headline numbers
are auto-generated into `paper/macros.tex`. Highlights of the local run on
Qwen2.5-0.5B:

- Logit MSE grows several orders of magnitude across the bit-width sweep
  (FP32 → ternary), exactly matching the round-to-nearest theoretical
  prediction MSE ∝ 4⁻ᵇ.
- Submodule sensitivity at 4-bit isolates which of {attention, MLP}
  contributes more to logit divergence — the basis for the proposed
  reasoning-aware mixed-precision recipe.
- Accuracy at INT4 holds up surprisingly well on simple arithmetic; the
  collapse happens at ternary.
