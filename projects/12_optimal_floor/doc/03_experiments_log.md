# 03 · Experiments log

All runs use `configs/local_small.yaml` (seed 0) on a laptop CPU.

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-09-18 | local_small | `make test` | 8 passed, 0.2 s |
| 2026-09-18 | local_small | `run_analytic` (1 s) | N=20: h* = 11 (α=0, last), 7 (α=0, lobby), 2 for α ≥ 0.25 with lobby parking; middle-floor regret 46 % (α=0.5) → 163 % (α=0.9) → 231 % (α=1) |
| 2026-09-18 | local_small | `run_traffic` (6 s) | validation 2.3 % (lobby) / 2.6 % (last); h* = 2 up to 2/min, 11 at 4/min (lobby), 16 (last); middle-floor regret 83 → 38 → 22 → 10 → 4 % |
| 2026-09-18 | local_small | `run_stairs` (4 s) | best 49 → 38 s; walked 10 % (7 % descend-only); h* = 2 in all variants; middle regret 22 → 51 % |
| 2026-09-18 | local_small | `run_fleet` (5 s) | best 65 / 52 / 43 s for 1 / 2 / 4 cars; middle regret 16 / 19 / 30 % |
| 2026-09-18 | local_small | `run_robust` (9 s) | 16 of 24 cells h* = 2; α = 0: 11 → 13 with load; α = 0.25: 5 → 11, band 2–12 at 4/min; archetypes residential/office/hospital h* = 2/2/2, bands 2–2 / 2–2 / 2–4 |
| 2026-09-18 | local_small | `make figures` | 6 figures, 128 macros |
| 2026-09-18 | local_small | `make paper` | 6 pages, no undefined references |

`make all` ≈ 1 min end to end.

## Observations
- **Stop convention mattered for validation.** The first closed form counted
  two door cycles per trip; the simulator records boarding and alighting at
  the car's *arrival*, so a trip contains exactly one dwell. Aligning the
  convention made the per-trip comparison exact (1e-6) and the run-level
  mean agree to ~2 %, the residue being the car caught mid-return to its
  parking floor.
- **Independent seeds per floor made h\* noisy.** With a different traffic draw
  for each home floor the flat curves at high load produced spurious optima
  (h* jumping between 4 and 12). Common random numbers (same background per
  replication for all floors) fixed it; the curves became smooth and the
  ordering stable.
- **Saturation inverts the low-floor advantage.** At 4/min on one car the wait
  *decreases* with height up to floor 16 (148 → 98 s) because the down sweep
  serves low floors last; this was unexpected and is explained in the paper
  with the wait/ride decomposition. The p90-optimal floor stays low.
- **The stairs asymmetry hypothesis failed.** Symmetric 12/12 s stairs and a
  descend-only resident move the optimum exactly as much as 15/10 s stairs;
  the upward shift comes from stairs absorbing short trips that a lobby-parked
  car served badly, not from the cost of climbing.
- **More cars steepen, not flatten.** Removing waiting leaves the ride, which
  is pure distance; the relative regret of the middle floor grows with fleet
  size even though every floor gets faster.
- A CUSUM-like "does the car stay put" question is answered empirically: under
  traffic the stay-put car is at the lobby 25–30 % of the time and otherwise
  roughly uniform — a mixture of the lobby and random policies.
