# 00 · Overview

This project studies the geometry of the residual stream of an autoregressive
language model as the input context grows. The hypothesis is that
long-context degradation has a geometric signature: an intrinsic-dimension
drop, an anisotropy rise, and a curvature flattening, all in the middle
layers.

We compute three lightweight metrics — TwoNN intrinsic dimension, mean
pairwise cosine similarity (anisotropy), and a local-PCA curvature proxy —
across a context-length sweep on GPT-2 (local) and Llama-3-8B / Mistral-7B
(GPU config).
