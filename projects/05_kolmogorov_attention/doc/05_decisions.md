# 05 · Decisions
- **gzip vs zlib vs zstd.** gzip used because it's universally available and
  the proposal explicitly mentions Kolmogorov complexity in a context where
  any reasonable compressor would do.
- **Forced eager attention.** Newer transformers default to SDPA which
  returns an empty attentions tuple; we override to `eager` so
  `output_attentions=True` actually populates.
- **7 prompt families.** Chosen to span the (0.05, 0.55) compression-ratio
  range. Adding more natural prompts via Wikitext is gated behind
  `gpu_full.yaml`.
