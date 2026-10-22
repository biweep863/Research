# 03 · Experiments log

| Date | Experiment | Config | Outcome |
|------|------------|--------|---------|
| 2026-10-16 | `fetch_data` (v1, astroquery async) | local_small | **truncated**: 11,067 of 48,809 rows, no error. Switched to chunked TAP sync + count check |
| 2026-10-16 | `run_static` (raw parallax cut) | local_small | r_50 = 2.27 pc; Sun pocket of 6 objects; prompted the brown-dwarf check |
| 2026-10-16 | `run_timedep` (raw cut, unconditional RV fill) | local_small | 62% reduction at 1 Myr, **but** the star count inside 15 pc grew 57% by 0.5 Myr, which is unphysical. Results discarded |
| 2026-10-16 | diagnosis | — | no-RV stars had median v_t = 2.6 km/s; 8,586 rows not in GCNS (spurious parallaxes). Fixed with the GCNS filter and a conditional RV fill |
| 2026-10-16 | `run_static` (GCNS) | local_small | r_10/50/90 = 2.12/2.16/2.30 pc; CI 2.15–2.17; Poisson 2.07; Gaia-only pocket bridge 2.27 pc; +20% brown dwarfs → 2.02 |
| 2026-10-16 | `run_timedep` (GCNS) | local_small | r_50 reduction 12/20/35/52/58% at 50/100/200/500/1000 kyr; count inside 15 pc stays at 99–104% through 0.5 Myr |
| 2026-10-16 | `run_routing` | local_small | fusion τ* 184 yr (static) / 88 yr (moving); greedy success 8/49/99.5% at 2.5/3/4 pc; host-only refuelling needs a 3.4× tank; sails reach 39% at a 1,000-yr hop budget |
| 2026-10-16 | `run_exploration` | local_small | 2-opt 7–9% shorter than NN; 1 probe 29.8 kyr vs 8 probes 4.3 kyr at 0.1c |

## Observations
- **Ties in tests.** The frozen-star equivalence test first failed because the
  median bottleneck equals an MST edge exactly. Choosing r between two
  distinct MST edges fixed it; the code was correct.
- **The threshold is local-density-limited, not boundary-limited.** r_50 is
  identical (2.16 pc) for evaluation radii of 10, 15, 20 and 25 pc.
- **Brown dwarfs are first-order for the Sun itself.** Luhman 16 (1.99 pc)
  and WISE 0855 (2.28 pc) change the Sun's pocket from 4 objects bridged at
  2.27 pc to fully connected at the field threshold.
- **Total local runtime** is about 3 minutes: timedep ~90 s (3 bootstrap draws),
  static ~25 s, routing ~7 s, exploration ~3 s.

## Scaling
`make all CONFIG=configs/gpu_full.yaml` (100 pc, 60 pc evaluation, 2 Myr). Not
yet run. The time-dependent stage is the expensive one: candidate pairs grow
as N·V·T·r², so expect ~20M pairs and tens of minutes per bootstrap draw.
