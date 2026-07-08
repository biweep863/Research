# 01 · Dataset

## Local configuration
Seven synthetic prompt families, each instantiated with multiple
seeds (~91 prompts total). Each prompt is truncated to 128 tokens.

Families, sorted from lowest to highest expected gzip ratio $\rho$:
- `low_entropy_pattern` — `aaaa…` (near-zero $\rho$).
- `repeating_short` — short periodic strings.
- `repeating_long`  — long periodic strings.
- `arithmetic` — number/operator sequences.
- `code_like` — synthetic snippets with brackets and identifiers.
- `english_text` — short English paragraphs.
- `random_uniform` — uniform random bytes (highest $\rho$).

The families collectively span compression ratio $\rho \in
[\approx 0.05,\, \approx 0.55]$, giving a broad regressor range from
a small, fully self-contained corpus.

## GPU configuration
`gpu_full.yaml` swaps the corpus for Wikitext-103 at 1024 tokens and
the model for Llama-3-8B; the pipeline is unchanged.
