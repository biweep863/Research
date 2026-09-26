# 03 · Experiments log

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-09-26 | local_small | `make test` | 15 passed, ~3 s |
| 2026-09-26 | local_small | `run_threshold_sweep.py`, single landscape per K (first attempt) | Non-monotonic, noisy $N_c$: 1.8, 324.7, 55.0, 716.3 at K=0,2,4,6. Diagnosed as landscape-instance variance dominating the K-dependence at a single draw per K; fixed by averaging over `landscapes_per_k` independent draws (see Observations). |
| 2026-09-26 | local_small | `run_threshold_sweep.py`, 5 landscapes/K (fixed) | Monotonic $N_c$: 2.7, 13.4, 212.3, 469.4 at K=0,2,4,6 (~175x over the sweep). AIC prefers **linear** over log-linear (37.2 vs 41.9) — the log-linear hypothesis is not supported at this K-grid resolution (4 points). 1:17 total. |
| 2026-09-26 | local_small | `run_effective_size.py`, 5 landscapes/K | s(K): 0.038, 0.075, 0.102, 0.118 (K=0,2,4,6) — **increasing**, not decreasing as hypothesized. $N_c^{\text{Kimura}}$: 26.4, 13.4, 9.8, 8.5 — **decreasing**, opposite the directly-measured $N_c$ trend. Ne_median ≈ 28–36 at N=100 (same order of magnitude as N, as expected for a simple GA without elitism). 5 s total. |
| 2026-09-26 | local_small | `make figures` | 3 figures, 24 macros, no errors. |

## Observations
- **Single-landscape-per-K is not the same as K-dependence.** The first
  threshold-sweep attempt used one random NK table per K and got a
  non-monotonic, high-variance $N_c(K)$ (even negative-looking trends
  between adjacent K). This is landscape-instance noise: at fixed K, two
  random NK tables can differ substantially in how rugged they actually
  are. Averaging `p_success` over `landscapes_per_k` independent draws
  before fitting $N_c$ fixed this and gave a clean monotonic trend. This
  is the primary reason `gpu_full.yaml` spends most of its extra compute
  budget on `landscapes_per_k` (5→30) rather than just a finer N grid.
- **The mechanism is falsified, the phenomenon is not.** $N_c$ growing
  with K is real and substantial (~175x from K=0 to K=6). But feeding the
  *measured* $s(K)$ into Kimura's formula predicts $N_c$ should *fall*
  with K, because $s(K)$ itself grows with K in these landscapes — the
  opposite of the "epistasis dilutes single-mutation effect size" story
  this project's hypothesis rested on. Reported as a real, informative
  negative result on the mechanism (doc/04_results.md), not smoothed over.
- **AIC prefers linear at N=4 K-points.** With only 4 K values, a
  2-parameter linear fit and a 2-parameter log-linear fit are both weakly
  constrained; the gpu_full run's 10-point K grid is a much stronger test
  of which functional form actually fits (see the GPU-scale rows below).

## GPU-scale run (2026-09-26)
| Config | Command | Outcome |
|--------|---------|---------|
| gpu_full (original scope: landscapes_per_k=30, generations=400, N up to 4096) | `run_threshold_sweep.py --limit 2` (smoke test, 2 of 10 K) | Killed after 15+ min still on K=0/1 -- far too slow to project a 10-K run within the session. |
| gpu_full (reduced: landscapes_per_k=10, generations=200, N up to 1024) | `run_threshold_sweep.py --limit 1` smoke test | Still >15 min for a single K, no clean improvement -- root cause not yet found. |
| gpu_full (reduced, no `--limit`) | Direct profiling: `run_ga(N=4)` 0.34s, `run_ga(N=64)` 2.43s, `run_ga(N=1024)` **37.6s** at generations=200, R=1000, L=20 | Diagnosed: `run_ga`'s random draws used a CPU `torch.Generator`, so every `torch.rand`/`torch.randint` call for a `(R,N,L)` tensor generated on CPU (single-threaded) then `.to(device)`-copied to GPU every generation -- cost scaled with tensor size, not with GPU compute. Not a kernel-launch-overhead problem as first suspected. |
| gpu_full (reduced) | Fixed `simple_ga.py` to use `torch.Generator(device=device)` throughout (no CPU generator, no host-to-device copy) | Re-profiled: `run_ga(N=1024)` **2.57s** (was 37.6s) -- a 14.6x speedup. Cross-device bit-identical reproducibility is no longer guaranteed (only same-device reproducibility, which is what the tests actually check), a deliberate trade documented in `doc/05_decisions.md`. |
| gpu_full (reduced) | Relaunched full chain; separately, an **earlier killed run's second stage (`run_effective_size`) had continued running** because killing the first stage's Python process doesn't stop a `.bat` script from proceeding to its next line -- it was competing for GPU with a fresh smoke test at the same time, producing misleadingly slow numbers before this was noticed and both were killed cleanly. | Root-caused via `Get-CimInstance Win32_Process` showing two unrelated experiment scripts running simultaneously; not a code bug, a background-job hygiene issue. |
| gpu_full (reduced, clean single run) | `run_threshold_sweep.py` (all 10 K), then `run_effective_size.py` | K=0-6 fast (59-105s each) but **K=7 took 1157.9s (19.3 min) and K=8 took 1534.5s (25.6 min)** -- both far slower than K=0-6 despite identical tensor shapes. Traced to `fit_threshold`'s 500-iteration bootstrap: at high K, `p_success` is very low and flat (0.0-0.3 across the whole N grid), and `sklearn.LogisticRegression` takes many more solver iterations to converge (or fails to and keeps iterating) on such a shallow, noisy curve than on a clean sigmoid -- a real, reproducible cost, not GPU contention (GPU was near 100% util from the still-running GA sweeps in parallel with an idle-looking CPU, which briefly looked like a red herring but wasn't the actual K=7/8 slowdown). Not fixed in this pass (`n_boot` could be reduced at high K, or the bootstrap could early-stop on convergence failure) -- noted as a real inefficiency for a future pass. |
| gpu_full (reduced) | `make figures` + `make paper` | 49 macros, 4-page PDF, no undefined refs. |

## GPU-scale headline numbers
- $N_c(K)$, $K=0..9$: 3.7, 9.8, 40.1, 367.5, 1374.6, 14490.6, 2409.2, 3591.2, 25350.6, 32506.0 -- an ~8785x increase from $K=0$ to $K=9$, non-monotonic at $K=5\to6$ even after 10-landscape averaging.
- Linear vs log-linear AIC: 181.5 vs 194.7 -- **linear preferred**, replicating the local pilot's finding at more than double the K-resolution (not an artefact of an underpowered 4-point pilot).
- $s(K)$, $K=0..9$: 0.0336, 0.0519, 0.0583, 0.0691, 0.0776, 0.0884, 0.0938, 0.1017, 0.1066, 0.1149 -- **monotonically increasing across all 10 points**, cleaner than the local pilot's 4-point version. $N_c^{\text{Kimura}}(K)$ falls correspondingly (29.7 → 8.7). The mechanism falsification from the local pilot replicates, more cleanly, at scale.
- $N_e$ at fixed $N=500$ grows from ~101 ($K=0$) to ~179 ($K=9$) -- still the same order of magnitude as $N$ throughout.
