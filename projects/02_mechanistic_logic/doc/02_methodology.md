# 02 · Methodology
- Probes: 5-fold stratified logistic regression on last-token residual
  activations at every block (embedding + 12 transformer blocks).
- Cross-surface: train on one surface, evaluate on the other.
- Head ablation: zero out the `head_dim` slice of attention output
  projection input for one head at a time; measure ΔAcc (logic) and ΔLM
  (cross-entropy on a neutral 50-token snippet).
