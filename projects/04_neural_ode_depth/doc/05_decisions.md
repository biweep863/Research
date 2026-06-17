# 05 · Decisions
- **Backward-through-solver, not adjoint.** Simpler; memory cost is fine
  at this size.
- **CPU default** because `torchdiffeq.odeint` uses float64 tolerances
  internally and MPS doesn't support float64.
- **Synthetic copy task** rather than wikitext — gives an irreducible
  cross-entropy floor that makes loss-range comparisons interpretable.
- **No adjoint comparison** — out of scope; would be a separate paper.
