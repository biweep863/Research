# 02 · Methodology

## Activation extraction
We use `output_hidden_states=True` on a standard HuggingFace causal LM and
collect the residual-stream tensor at every block (embedding included).

## Metrics

### Intrinsic dimension (TwoNN, Facco et al. 2017)
Given the ratio μ = r₂/r₁ of second- to first-nearest-neighbour distances,
the MLE for the intrinsic dimension is d̂ = N / Σ log(μ). We truncate the
upper 10 % tail of μ.

### Anisotropy (Ethayarajh, 2019)
Mean cosine similarity between M=2000 random token pairs. Values ≈0 mean
isotropic; values close to 1 mean a narrow cone.

### Local-PCA curvature proxy
For each anchor point, PCA on its k=20 nearest neighbours; report
ratio (Σ_{j>m} σ²_j) / (Σ_{j≤m} σ²_j) with m=3.

### Effective rank
exp(H(p)) of singular-value distribution — a smooth complement to TwoNN.

## Pipeline
```
load_long_docs.py
        │
        ▼
run_geometry_sweep.py — for each (model, ctx_length, layer):
        │                 collect activations, compute metrics
        ▼
results/sweep_<short>.json
        │
        ▼
plot_geometry.py — figures + LaTeX macros
        │
        ▼
paper/main.pdf
```
