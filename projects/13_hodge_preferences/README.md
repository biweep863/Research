# Project 13 — Transitive first: does gradient descent learn the ranking before the cycles?

## Question
Pairwise preferences split, on the comparison graph, into a ranking (gradient),
local cycles (curl, rock-paper-scissors) and global cycles (harmonic) — the
Hodge decomposition behind HodgeRank. Scalar Bradley–Terry reward models can
only represent the ranking; skew-symmetric models (antisymmetric nets, GPM) can
represent cycles. **In which order does gradient descent learn these parts?**
We answer with the neural tangent kernel *between edges*, projected on the Hodge
subspaces, its Gaussian-process limit, and training runs on synthetic data and
on Chatbot Arena votes.

## Hypotheses
- **H0.** The scalar edge NTK is `B Θ_V Bᵀ`, so it annihilates every cycle (proved).
- **H1.** The antisymmetric edge NTK ranks gradient above curl above harmonic.
- **H2.** Training follows the spectrum: ranking first, cycles later.
- **H3.** i.i.d. noise is mostly cyclic, a fraction (|V|−1)/|E| lands in gradient space, and early stopping filters it.
- **H4.** Real Arena votes show the same order.

## Outcome
- **H0: confirmed.** The scalar model's cycle rate is ≤ 3e-16.
- **H1: gradient vs cycles confirmed; curl vs harmonic refuted.** The antisymmetric net's gradient rate exceeds its cycle rate by 4× to 123× across 96 graphs, and the ratio grows with the graph's mean degree. Curl and harmonic cycles are learned at the same rate.
- **H2: confirmed.** In 12 of 12 runs the net fits the ranking first and the cycles 8–29× later. This holds under both the squared and the Bradley–Terry losses, and across all 24 width and lazy/rich runs.
- **H3: partly confirmed.**
  - The noise lemma holds within 0.005.
  - Vote flips mostly shrink the transitive signal instead of adding cyclic noise.
  - Stopping when the ranking is fitted is too early.
  - Early stopping still gains up to 12 accuracy points at 40 % flipped votes.
- **H4: the order holds.** Arena's 15.3 % cyclic energy is below a transitive null (17.5 %), so the cycles are sampling noise and a scalar reward predicts held-out votes best.

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/13_hodge_preferences
make test           # 9 tests (~2 s)
make run            # 5 experiments -> results/*.json (~13 min on a laptop CPU)
make figures        # 6 figures + paper/macros.tex
make paper          # compile paper/main.pdf
```
Arena votes are cached in `data/arena_counts.npz` (125 KB, checked in). To
rebuild it from the raw 184 MB CSV (`lmarena-ai/arena-human-preference-55k`,
Apache-2.0), set `ARENA_CSV=/path/to/train.csv`.

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
More graph sizes, feature dims and depths, finer width and α sweeps, 10 seeds, longer training. CPU is enough.

## Files
- `src/hodge/{graph,decompose}.py`: comparison graphs, B / C / L1, orthogonal Hodge projectors, HodgeRank
- `src/kernels/{ntk_relu,edge_ntk,empirical}.py`: closed-form ReLU NTK, edge kernels, Hodge rates, spectral mass, linearised flow, empirical NTK (torch.func)
- `src/models/nets.py`: NTK-parameterised scalar BT, antisymmetric and GPM models, with centring and output scale α
- `src/data/{synthetic,arena}.py`: controlled-Hodge synthetic preferences, votes with flips, Arena loader and cache
- `src/train.py`: full-batch training with per-component tracking
- `src/experiments/run_{spectrum,dynamics,noise,width,arena}.py`: write `results/*.json`
- `src/figures/plot_hodge.py`: 6 figures + 144 LaTeX macros
- `paper/main.tex`: two-column paper (7 sections)
- `doc/`: overview, data, methodology, experiments log, results, decisions
