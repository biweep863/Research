# 05 · Decisions
- **ripser over gudhi/giotto-tda.** ripser has the cleanest distance-matrix
  API and is fast for small dense matrices.
- **Distance d = 1 - max(A, A^T).** Several alternatives (1 - A^T A,
  -log A) were considered; this one preserves the [0,1] range and is
  symmetric by construction.
- **H0 over H1 at this scale.** Short prompts give degenerate 1-complexes;
  H0 is the only stable signal at the local pilot scale.
- **N=10 deliberately.** Pilot's job is effect-size estimation for power
  analysis, not significance.
