# 03 · Experiments log

All runs use `configs/local_small.yaml` (seed 0) on a laptop CPU.

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-09-21 | local_small | `make test` | 9 passed, 2 s |
| 2026-09-21 | local_small | `run_spectrum` (14 s) | antisym κ_grad/κ_cyc 4.0–123 over 96 graphs, grows with mean degree; curl/harm 0.80–1.71; scalar cyc ≤ 3e-16; empirical ≈ analytic (4.5 % at w=2048); GPM ratio 4–6 |
| 2026-09-21 | local_small | `run_dynamics` (43 s) | 12/12 order holds; cycles fitted 25×/29× later (antisym sq), 19×/22× (BT), GPM 12×/19×; linearised within 0.02; held-out acc scalar 0.681, antisym 0.882, GPM 0.840 |
| 2026-09-21 | local_small | `run_noise` (~10 min, background) | lemma within 0.0048 in 9 graphs; Gaussian noise 0.20 in grad (theory 0.21); vote-flip perturbation 0.60 in grad; early-stop gain at 40 % flips: antisym 4.3, GPM 12.0, scalar 4.1 pts; t* too early (final run, 2 seeds) |
| 2026-09-21 | local_small | `run_width` (48 s) | order holds 24/24; t90(curl)/t90(grad) 66× lazy → 17× rich; drift 0.03 → 6.6 |
| 2026-09-21 | local_small | `run_arena` (2.5 min) | 63 models / 640 pairs / 52 200 votes; cyclic 15.3 % vs transitive null 17.5 % (p = 0.96); order ratio 44× (antisym), 160× (GPM); held-out log-loss scalar 0.6282 < GPM 0.6306 < antisym 0.6317 |
| 2026-09-21 | local_small | `make run` from clean, torch pinned to 1 thread | 5 result files in 13 min 22 s (first attempt without pinning: 16 min); noise sweep trimmed to 2 seeds × 3 levels per noise type, Arena to 2 seeds; all headline numbers unchanged |
| 2026-09-21 | local_small | `make figures`, `make paper` | 6 figures, 144 macros, 6-page paper, no undefined refs |

## Observations
- **Analytic NTK was off by ~2× at first.** The input layer also carries σ_w² in the NTK parameterisation. After the fix, analytic vs empirical agree at 4 % (width 4096), and the error falls like 1/√width.
- **The first learning rate was too big to see anything.** At η = E/λ_max everything converged in a handful of steps. Using 0.25 E/λ_max with log-spaced logging resolves the dynamics.
- **Curl vs harmonic was a wrong hypothesis.** Both are seen only through the interaction part of the kernel, and their rates match (mean ratio 1.02).
- **Vote flips are not cyclic noise.** Flipping votes pulls log-odds toward 0 in proportion to the signal, so 60 % of the perturbation is transitive. Lemma 2 applies to i.i.d. measurement noise.
- **The NTK stopping time t\* is too early.** Stopping when the ranking is 95 % fitted also discards the true cycles (0.750 vs 0.872 accuracy). The best step moves earlier with noise (1974 → 236 steps).
- **Bootstrap on Arena flows is biased toward cycles.** Parametric resampling adds a second layer of sampling noise. We therefore test genuine intransitivity with a transitive null that has the same counts; the bootstrap is reported only for spread.
- **Arena cycles are noise.** The observed cyclic energy is below the null. The flexible models' late training phase is overfitting, and early stopping recovers it.
