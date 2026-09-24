# 02 · Methodology

```
features X, comparison graph ─► hodge/ (B, C, projectors P_grad/P_curl/P_harm)
                               ─► kernels/ (closed-form ReLU NTK → edge NTK Θ_E; empirical NTK via torch.func)
                               ─► Hodge rates κ_c = tr(P_c Θ_E P_c)/dim_c, spectral mass, linearised flow
models/ (scalar, antisym, GPM) ─► train.py (GD, squared or BT loss; per-component fit, held-out metrics)
experiments/run_* ─► results/*.json ─► figures/plot_hodge.py ─► paper/figures + macros.tex ─► paper/main.pdf
```

## Hodge (`src/hodge`)
- **Operators.** (B s)_ij = s_j − s_i; (C y)_ijk = y_ij + y_jk − y_ik over triangles.
- **Projectors.** P_grad projects onto im B (via SVD), P_curl onto im Cᵀ, and P_harm = I − P_grad − P_curl.
- **Tests.** Orthogonality, idempotence and completeness; a rock-paper-scissors triangle is pure curl, and a triangle-free 4-cycle is pure harmonic.

## Kernels (`src/kernels`)
- **ReLU NTK, NTK parameterisation.** σ_w² = 2 on every layer (the input layer included) and σ_b² = 0.1. Arc-cosine recursion (Cho & Saul).
- **Scalar edge NTK.** Θ_E = B K Bᵀ.
- **Antisymmetric edge NTK.** K(z,z') − K(z,z̃') − K(z̃,z') + K(z̃,z̃') on concatenated inputs.
- **GPM.** Empirical kernel only, since the bilinear form makes it initialisation-dependent.
- **Analytic vs empirical.** They agree to 4.5 % (width 2048) and 16 % (width 128), with error falling like 1/√width.
- **Linearised flow.** r_t = (I − ηΘ/E)^t y, with g_0 = 0 (networks are centred: α(net_θ − net_θ0)).

## Training (`src/train.py`)
- **Optimiser.** Full-batch GD with η = 0.25 E / λ_max(Θ_E), divided by α² for the lazy/rich sweep.
- **Bradley–Terry loss.** BCE on vote counts with 4× the step (the curvature at g = 0 is 1/4).
- **Logged metrics.** Fitted fraction per component, 1 − ‖P_c(y − g)‖²/‖P_c y‖², against the *clean* truth; held-out sign accuracy against clean truth, or vote accuracy for Arena.

## Metrics
- Hodge rates and their ratios.
- t50 and t90 per component, and the ratios t(cycle)/t(grad).
- Maximum deviation of the network from the linearised prediction.
- Noise energy fraction in gradient space vs (|V|−1)/|E|.
- Best-step vs final held-out accuracy; accuracy at the NTK-predicted t* (95 % gradient fit).
- Kernel drift ‖Θ_T − Θ_0‖/‖Θ_0‖.
- For Arena: cyclic fraction vs a BT null (votes simulated from weighted HodgeRank scores with the same counts), a p-value, and held-out log-loss and accuracy.

## Lemma 2 (noise fraction)
For z ~ N(0, σ² I_E), ‖P_grad z‖²/‖z‖² ~ Beta((|V|−1)/2, (|E|−|V|+1)/2), with mean (|V|−1)/|E|. This follows from rotational invariance and rank(P_grad) = |V| − 1 for a connected graph.

## Verification (`tests/test_model.py`, 9 tests, ~2 s)
- Projector properties.
- RPS is pure curl and the square is pure harmonic.
- The scalar kernel is blind to cycles.
- The antisymmetric net and GPM are exactly antisymmetric.
- Analytic vs empirical NTK < 8 % at width 4096.
- The linearised formula matches explicit iteration.
- The noise fraction matches theory.
- HodgeRank recovers a planted ranking.
- The synthetic task has all three components.
