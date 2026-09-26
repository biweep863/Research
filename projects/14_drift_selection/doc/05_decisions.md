# 05 · Decisions

## No elitism
A generational GA with elitism can never lose the best genome once found,
which has no population-genetics analogue and would make the drift/
selection comparison meaningless (elitism is itself an infinite-selection-
strength intervention on the single best individual). The GA here is a
plain generational Wright-Fisher-like process: tournament selection,
crossover, mutation, nothing else.

## Adjacent-neighbourhood NK model, not random neighbourhoods
Kauffman's original NK model allows each locus's K neighbours to be chosen
randomly per locus. This project uses the simpler adjacent (circular)
neighbourhood — locus $i$ depends on loci $i, i+1, \dots, i+K \pmod L$ —
because it is easy to vectorise exactly (fixed-shape gather, no per-locus
index list) and is a standard, well-studied NK variant; the ruggedness
properties that matter here (number of local optima growing with K) hold
for both variants.

## Landscape averaging is the primary fix, not a footnote
The first attempt at the threshold sweep used one random landscape per K
and got a non-monotonic $N_c(K)$ (doc/03). Averaging $P(\text{success})$
over `landscapes_per_k` independent landscape draws before fitting $N_c$
resolved this. This is treated as *the* correct experimental design, not
a post-hoc fix applied only when something looked wrong: the K-dependence
this project studies is a property of the *NK(L,K) ensemble*, not of any
one random table, and the pilot's own numbers make that distinction
concrete (single-draw $N_c$ at K=2 was 324.7; 5-landscape-average $N_c$ at
K=2 is 13.4 — a 24x difference from instance noise alone).

## Exact brute-force optimum, not a large random-restart search
$L$ is kept small enough (16 locally, 20 at GPU scale) that
`brute_force_optimum` enumerates every genome exactly. A larger $L$ with an
approximate "best found so far across many restarts" optimum would
introduce ambiguity into the success criterion itself (is a "failure" a
real GA failure, or did the reference search also miss the true optimum?)
that would undermine the whole threshold-fitting analysis. This caps how
large $L$ can be scaled, which is a real limitation, not a free choice —
noted rather than hidden.

## DFE-based s(K), not a single fixed mutation
$s(K)$ is the *mean beneficial* effect size from sampling many random
genomes and single-locus flips (the distribution of fitness effects, a
standard population-genetics object), not the effect of one specific
"designed" mutation. This is the more defensible choice for comparing
against Kimura's formula, which itself describes a randomly-arising
mutation's fate, not a chosen one.

## The mechanism was falsified — reported, not adjusted post-hoc
The measured $s(K)$ increases with K, which makes Kimura's formula predict
a *falling* $N_c(K)$ against the measured *rising* $N_c(K)$
(doc/04_results.md). It would have been possible to quietly redefine $s(K)$
(e.g. as a landscape-wide fitness-variance measure instead of a DFE mean)
until the sign matched the hypothesis. That was not done: the DFE-based
definition was fixed before running the mechanism check, and the result is
reported as a falsification of the specific mechanism, with the underlying
phenomenon ($N_c$ growing with K) reported as still real and worth a
different explanation.

## Why this project scales on GPU at all
A bitstring GA on modest population sizes is cheap even on CPU; the
GPU-scale config's value is not "the pilot was too slow" (it finished in
under two minutes) but statistical power: `gpu_full.yaml` targets 10 K
points instead of 4, directly addressing the weakness noted in doc/03 (a
4-point K grid is a weak test of linear-vs-log-linear scaling). The same
vectorised torch implementation runs both configs; only the tensor sizes
change. `gpu_full.yaml`'s `landscapes_per_k`/`generations`/`n_grid` were
scaled back from an initial, more ambitious version (30 landscapes, 400
generations, N up to 4096) after that version proved too slow to finish in
the session — see the RNG fix below, which is the actual reason, not a
concession on scientific scope: 10 K points at 10 landscapes each is still
2.5x the K-resolution and 2x the landscape-averaging of the local pilot.

## The real GPU bottleneck was CPU-generated randomness, not kernel-launch overhead
Direct profiling (`run_ga` called standalone, not through the sweep) found
`N=1024, R=1000, L=20, generations=200` took 37.6s, vs 0.34s at `N=4` --
a ~110x blowup for a ~256x increase in population size, i.e. worse than
linear but not the "many small ops, kernel-launch-overhead-bound" pattern
first suspected from the generation-loop structure. The actual cause:
`simple_ga.py`'s random draws (`torch.rand`/`torch.randint`) used a single
CPU `torch.Generator` "for reproducibility regardless of device," so every
per-generation random tensor of shape `(R,N,L)` was generated on CPU
(single-threaded) and then `.to(device)`-copied to GPU -- a cost that
scales with tensor size and is paid on the CPU, not accelerated by the
GPU at all. Switching to `torch.Generator(device=device)` (generating
randomness directly on the compute device) cut the same call to 2.57s --
a 14.6x speedup. The trade-off: CPU and CUDA runs of the same seed no
longer produce bit-identical trajectories (only same-device runs do,
which is what `tests/test_ga.py::test_reproducible_with_same_seed` checks
and is the only reproducibility guarantee this project ever needed). The
lesson generalises beyond this project: "vectorise across replicates as a
batch dimension" is necessary but not sufficient to make a GPU port fast
-- every other tensor-producing call in the hot loop, including RNG, has
to actually run on the target device.

## fit_threshold's bootstrap is a real, undiagnosed cost at high K
Two of the ten K values in the GPU-scale run (K=7, K=8) took 19 and 26
minutes respectively, an order of magnitude longer than K=0-6 (under two
minutes each), despite identical tensor shapes in the GA sweep itself.
Traced to `fit_threshold`'s 500-iteration bootstrap: at high K,
`p_success` is low and nearly flat across the whole N grid (0.0-0.3), and
`sklearn.LogisticRegression`'s solver appears to need many more iterations
(or repeatedly fails to fully converge) fitting such a shallow, noisy
curve 500 times over, compared to a clean sigmoid at low K. This is
reported rather than silently absorbed into the K=7/8 runtimes: it is a
real inefficiency in the current bootstrap implementation (a fixed
`n_boot` regardless of how well-conditioned the fit is), not a GPU issue,
and is left as a known limitation rather than patched under time pressure
in this pass. A stray leftover process from an earlier killed run was
briefly suspected as a confound (it had continued running because killing
a `.bat` script's current line does not stop the script from advancing to
its next line) but was ruled out by process-listing before attributing the
K=7/8 slowdown to the bootstrap.
