# 05 · Decisions & Alternatives

## TwoNN vs. MLE-Levina-Bickel
We picked TwoNN because the proposal in `mainideas.txt` flags it explicitly
and because it is robust in d ≥ 100. The Levina-Bickel estimator was
considered as a cross-check; it can be added in `analysis/intrinsic_dimension.py`
without changing the public API.

## Local-PCA curvature vs. true Ricci/sectional curvature
A genuinely Riemannian curvature would require a metric tensor estimate at
each point. The local-PCA ratio is a proxy that's cheap and visually
interpretable, but it should not be over-interpreted as Ricci curvature.

## GPT-2 as local default
GPT-2 small fits on CPU and runs the full sweep in ~90 s, so iteration is
fast. Pythia-410M is the natural next step; Llama-3-8B is the GPU target.

## Why no normalisation of activations
Anisotropy is scale-invariant by construction (cosine similarity), and the
local-PCA proxy is normalised internally. TwoNN is scale-invariant because
it only sees neighbour-distance ratios. Pre-normalising would lose
information about how the manifold sits inside $\mathbb{R}^d$.
