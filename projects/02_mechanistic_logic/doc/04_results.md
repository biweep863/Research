# 04 · Results

## Local pilot (GPT-2)
Probes recover validity at high accuracy (English near-perfect, conlang
≈0.96) at mid-network layers. Cross-surface transfer is meaningful but
moderate (≈0.67 / 0.59), suggesting the model has a partially
language-independent representation of validity that is not perfectly
shared between English and the OOD conlang. Per-head ablation reveals no
purely logic-specific head at GPT-2 scale; the most-damaging head also
hurts language modelling cross-entropy.

## GPU-scale (Pythia-2.8B, 10 templates, N=5120 probe items, 1024-head ablation)
Probe accuracy is still near-perfect for both surfaces (1.000 English,
0.997 conlang) — a bigger model still linearly encodes validity cleanly in
both surface forms. But **cross-surface transfer is weaker at 2.8B than at
GPT-2 scale** (0.58/0.51 vs 0.67/0.59), with cl→en barely above chance.
Taken at face value, this says the *linearly decodable* representation
recovered by a probe does not become more surface-form-invariant with
scale — if anything, the opposite in this comparison. This is reported
as-is rather than reconciled with an intuition that "bigger models
generalize better": a probe's cross-surface transfer measures whether the
*same linear direction* works on both surfaces, and a larger model could
easily encode validity more accurately per-surface while using two more
different directions to do it, which is exactly compatible with what is
observed here (near-perfect within-surface accuracy, weaker cross-surface
transfer).

Ablation shows the effect of any single head is far smaller at 2.8B scale
(|ΔAcc| ≤ 0.025 across all 1024 heads, vs GPT-2's clearer −0.033 worst
case) and uncorrelated with language-modelling damage (r≈0.03, vs a
visible link at GPT-2 scale) — consistent with logical-validity behaviour
becoming more distributed/redundant across heads as the model grows,
rather than concentrating in a small identifiable circuit.
