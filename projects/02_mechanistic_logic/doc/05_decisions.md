# 05 · Decisions

- **Manual hooks vs TransformerLens.** Chose pure HF forward hooks for the
  local pipeline to avoid an extra dependency; TransformerLens drop-in is a
  follow-up.
- **Zero ablation vs mean ablation.** Zero ablation is closer to a
  causal-effect intervention and avoids leaking statistics from other
  inputs into the patched head. Mean ablation is exposed as a config flag
  (`ablation.baseline: mean`).
- **Why exhaustive enumeration of 144 heads.** GPT-2-small is small enough
  that the full grid runs in ~2.5 minutes; this avoids any selection bias
  that would come from a screening heuristic.
- **Probe model.** Logistic regression with L2 only. More expressive probes
  (MLPs) inflate accuracy spuriously on small training sets.
