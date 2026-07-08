# 04 · Results

Headline (GPT-2 small, 91 prompts × 13 layers):
- Best layer $R^2$:  **0.683 at layer 5** (mean attention entropy
  regressed on gzip ratio $\rho$).
- Deepest-layer $R^2$: **0.088**.

The relationship between prompt compressibility and attention density
lives in the mid-network: early layers attend on local syntactic
patterns that depend less on global complexity, and the deepest
layers compress toward token logits, washing out prompt-level
signal. The middle is a Goldilocks zone where the attention pattern
is legibly tied to how compressible the prompt is.

Low-complexity families (`low_entropy_pattern`, `repeating_short`)
sit systematically below higher-complexity families in mean attention
entropy through most of the network, before re-converging near the
head.

See `paper/main.pdf` for figures ($R^2$ per layer, deep-layer
scatter, entropy per family).
