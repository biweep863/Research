# 05 · Decisions & Alternatives

## Topic choice
We weighed three candidates: Hodge spectral bias in preferences, GP–NTK posterior as a failure predictor, and a Siamese/contrastive pair NTK. The first unites all four themes (NTK, pairs, GD, GP) and had no direct prior work; the other two overlap with busy literatures (NTK-GP uncertainty, SSL kernel theory). User chose the first.

## Edge kernel instead of item kernel
Preference models are trained on pairs, so the operative kernel lives on edges. Only the edge kernel can be projected onto Hodge subspaces. The item kernel hides the effect: for the scalar model the edge kernel is B K Bᵀ, and its cycle blindness is invisible at item level.

## Mean Hodge rate as the summary statistic
κ_c = tr(P_c Θ P_c)/dim_c is the expected Rayleigh quotient of a random flow in the component. It is target-independent and cheap. Target-specific Rayleigh quotients are also recorded (`run_dynamics`) and agree in sign and order.

## Antisymmetric net on concatenated inputs
This is the simplest universally antisymmetric architecture. GPM is included because it is the model the alignment literature actually uses. We rejected a GNN-style edge model because it would blur the comparison with TopoNTK.

## Centred networks
α(net_θ − net_θ0) starts at g = 0. This removes the random initial function, so fitted fractions start at 0 and match the linearised theory exactly. It is also the standard way to make lazy vs rich comparable.

## Synthetic truth as a function of features
Components defined by projection on the training graph would have no held-out targets. A feature-level truth (utility + skew bilinear) makes generalisation measurable; the actual energies are then measured by projection.

## Arena at model level with one-hot items
Prompt-level analysis would need text embeddings and a far sparser graph. Model level gives a dense, well-sampled graph where the intransitivity question is testable. This is stated as a limitation.

## Transitive null instead of bootstrap for Arena cycles
Parametric bootstrap around the observed flow double-counts sampling noise. Simulating from the best transitive fit with the same counts is the correct test of "more cycles than noise would create".

## Keeping failed predictions
We keep three predictions that failed: curl vs harmonic ordering, the NTK stopping time t*, and "noise is cyclic" for vote flips. Each teaches something (the interaction-part mechanism, signal cycles, shrinkage) and is reported in the paper.
