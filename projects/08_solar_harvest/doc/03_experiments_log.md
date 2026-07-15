# 03 · Experiments log

| Date | Experiment | Config | Outcome |
|------|------------|--------|---------|
| 2026-07-14 | `run_resource` | local_small | TOA 173,549 TW; surface GHI 184 W/m²; desert-belt band the resource max; deployable 6.1 Mkm² → 244 TW (81× electricity) |
| 2026-07-14 | `run_sbsp` | local_small | ground 40 W/m², orbit 261 W/m² (6.5×); advantage traced to capacity factor (99% vs 25%), not atmospheric bypass |
| 2026-07-14 | `run_optimize` | local_small | temporal water-filling beats flat reservation by 17.9% at f_min=0.3, up to 45.4% at f_min=0.6 |

## Observations
- **Sanity checks all pass.** Intercepted power ≈ 174,000 TW, global TOA mean =
  S0/4, surface GHI ≈ 184 W/m² — the model reproduces the standard energy budget.
- **First `run_optimize` (mild demand curve) gave only 1–14% temporal gain.**
  The gain is bounded by how much harvest value varies across *daylight* hours.
  Switching to a realistic solar-grid "duck curve" (deep midday value trough)
  raised the gain to 5–45% and is more physically defensible (midday energy is
  genuinely low-value when solar supply saturates the grid). See `05_decisions.md`.
- **The LP oracle in the test was initially written with the reserve inequality
  flipped** (reserve ≤ d_min instead of ≥), making the LP look better than the
  greedy optimum. The greedy solver was correct; the test was fixed and the two
  now agree to 1e-6.
- **Temporal gain is monotone in f_min** — the more light the ecosystem needs,
  the more scheduling matters. This is the paper's central quantitative message.

## Scaling
`make all CONFIG=configs/gpu_full.yaml` refines to 5° latitude bands, 365-day
means, half-hour resolution, and a finer f_min sweep. Still CPU-only; runs in
well under a minute.
