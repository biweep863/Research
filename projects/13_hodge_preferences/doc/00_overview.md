# 00 · Overview

**Origin.** Research idea "Neural tangent kernel, pairwise relations, gradient
descent, Gaussian process". NTK alone is crowded (alignment, edge of stability,
grokking). The open niche that joins all four themes: the *learning dynamics*
of pairwise-preference models seen through the Hodge decomposition of the
edge-level NTK.

**Literature gap (checked Sep 2026).**
- HodgeRank decomposes preference flows but has no networks or training.
- BT vs GPM / HRC (2025–26) treats expressivity only, with no dynamics.
- Neural dueling bandits use the NTK only for confidence bounds.
- TopoNTK (2026) is the closest work: a topological spectral bias for GNN message passing on simplicial complexes. It has no preference models, no BT loss, no noise and no early stopping.

**Question.** In which order does gradient descent learn the transitive (gradient),
local-cycle (curl) and global-cycle (harmonic) parts of pairwise preferences?

**Hypotheses and outcome.**
| | Hypothesis | Result |
|---|---|---|
| H0 | the scalar edge NTK annihilates cycles | proved; measured ≤ 3e-16 |
| H1 | antisymmetric NTK: κ_grad > κ_curl ≥ κ_harm | gradient ≫ cycles (4–123×) holds; curl vs harm **fails** (ratio 0.8–1.7) |
| H2 | training fits the ranking first | 12/12 runs; cycles 8–29× later; linearised theory within 0.02; survives feature learning (24/24) |
| H3 | noise is cyclic; NTK-timed early stop filters it | lemma exact (±0.005); **vote flips are mostly transitive** (shrinkage); t* too early; early stopping still +4 to +12 pts at 40 % flips |
| H4 | real data shows the order | yes; Arena cycles (15.3 %) ≤ transitive null (17.5 %), p = 0.96: sampling noise |

**Reproduce.**
```bash
cd projects/13_hodge_preferences
make test      # ~2 s, 9 tests
make all       # ~13 min on a laptop CPU (over the 5-min template target): 5 experiments, 6 figures, 144 macros, paper/main.pdf
```
