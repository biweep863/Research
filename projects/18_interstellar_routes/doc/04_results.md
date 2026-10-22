# 04 · Results (local_small: 50 pc GCNS, 40,229 objects)

## Frozen stars
| Quantity | Value |
|---|---|
| r_10 / r_50 / r_90 / r_99 (≤25 pc, 5,117 stars) | 2.12 / **2.16** / 2.30 / 2.62 pc |
| r_50 95% CI (30 astrometric draws) | 2.15–2.17 pc |
| median nearest-neighbour distance | 1.18 pc |
| Poisson null r_50 (same density) / analytic r_c | 2.07 / 2.04 pc (real field ~5% harder) |
| Gaia-only: Sun pocket, bridge | Sun + Proxima + Barnard's + Ross 154, bridged by **2.27 pc** |
| + unseen brown dwarfs 10/20/30% | r_50 = 2.08 / 2.02 / 1.97 pc |
| route stretch / hops at r_90 | 2.5× / 28 hops (median) |

## Moving stars (≤15 pc, 1,070 stars, ships at 0.1c)
| Horizon | r_50 | Reduction vs frozen |
|---|---|---|
| frozen | 2.15 pc | — |
| 100 kyr | 1.73 pc | **20%** |
| 200 kyr | 1.40 pc | 35% |
| 500 kyr | 1.04 pc | **52%** |
| 1 Myr | 0.91 pc | 58% (conservative: boundary losses) |

- Effective front speed: 22 pc/Myr (21 km/s) at r = 1 pc; 76 pc/Myr at 1.5 pc.
- Settle delay at r = 1.5 pc, T = 200 kyr: 65% reached (0 kyr), 44% (10 kyr), 2% (100 kyr).
- Ship speed at the same point: 65% (0.1c), 45% (0.01c), 3% (0.001c).

## Engineering
| Technology (R = 10) | v_cruise | τ* frozen | τ* moving (≤500 kyr wait) |
|---|---|---|---|
| chemical | 5 km/s | 417,000 yr | 200,000 yr |
| electric | 115 km/s | 18,400 yr | 8,800 yr |
| fusion | 11,500 km/s (3.8% c) | **184 yr** | **88 yr** |
| antimatter | 0.36 c | 19 yr | 9 yr |

- Greedy pilot success: 8% (2.5 pc), 49% (3 pc), 99.5% (4 pc). A* expands
  10% (2.5 pc) to 2% (4 pc) of Dijkstra's nodes.
- Refuelling only at 310 hosts (≤30 pc, hops ≤3 pc): a 7.5 pc tank reaches
  half the stars, 3.4× the 2.18 pc needed when any star refuels.
- Sails (0.2c launch): median brake speed 1,225 km/s; only 21% of targets
  can stop a sail faster than 0.01c. Reach is 39% / 83% at 1,000 / 3,000-yr
  hop budgets, vs 100% for a rocket. Reaching half the stars takes
  3,934 yr vs 336 yr.

## Exploration (572 stars ≤12 pc, 0.1c)
2-opt is 7–9% shorter than nearest-neighbour for fleets of 1–16. NN is 26%
and 2-opt 15% above the MST bound. Makespan is 29.8 kyr for 1 probe and
4.3 kyr for 8 probes.
