# 02 · Methodology

## Phase space
Positions in pc, velocities in pc/Myr (1 km/s = 1.0227 pc/Myr), heliocentric
ICRS axes, with the Sun as node 0 at rest. Only relative motion enters, so the
frame does not matter. Stars move on straight lines.

Missing RVs: a Gaussian velocity ellipsoid N(μ, Σ) is fit to stars with RVs
(speed < 200 km/s). Each missing RV is drawn from the conditional Gaussian
given the star's measured tangential components.

## Exp. 1: frozen stars (`run_static.py`)
- Minimax path = MST path. The EMST comes from a k-NN graph (k doubled until
  connected); it equals the dense MST in tests.
- A star's bottleneck is the largest MST edge on the Sun → star path. Reach(r)
  is the fraction of stars in the evaluation ball with bottleneck ≤ r. r_q is
  the q-quantile of the bottleneck.
- Extras: astrometric bootstrap (Gaussian draws of parallax, PM and RV from
  Gaia errors, plus a fresh RV fill); an evaluation-radius sweep; a Poisson
  null (same N, uniform ball; analytic r_c from n·(4/3)πr³ = 2.7348); a Gaia-only
  variant; injection of unseen brown dwarfs (10/20/30% of N, velocities borrowed
  from real stars).

## Exp. 2: moving stars (`run_timedep.py`)
- Pair (i, j) separation: d(t) = |p + w t|. The window where d ≤ r_eff is one
  interval from a quadratic, with r_eff = r(1 − |w|/v) so that the flown
  intercept path is ≤ r.
- Candidate pairs come from time-sliced KD-trees. The query radius is
  r_max + V·dt/2, with V = 2 × the 99.9th percentile of speed.
- Earliest arrival: Dijkstra over arrival times. Departure is
  max(ready + settle, a), feasible if ≤ b, and arrival = departure + intercept
  time. This is exact because the network is FIFO when waiting is allowed.
  It is checked against a Bellman–Ford fixed point.
- One Dijkstra per r gives reach at all horizons T. r_q(T) is interpolated
  on the r grid. T = 0 is the frozen-star answer.

## Exp. 3: engineering (`run_routing.py`)
- Rocket cruise speed: v = c·tanh(v_e ln R / 2c). Survival per hop to
  percolate: τ* = r_50 / v.
- Pilots on static range graphs at 2.5, 3 and 4 pc: greedy geographic routing
  (success rate, stretch) and A* vs Dijkstra node expansions.
- Fuel bottleneck with refuelling only at hosts: multi-source Dijkstra from
  the stations, an MST on the station-to-station distance matrix, then
  need(s) = min over stations h of max(b_h, dist(h, s)). It reduces to the
  minimax when all nodes are stations and to straight distance when none are.
  Both cases are tested.
- Sails: the hop i→j flies at min(v_launch, v_brake(L_j)), with v_brake ∝ L^0.4
  calibrated to 13,800 km/s at α Cen A. Luminosity comes from G with zero
  bolometric correction.

## Exp. 4: exploration (`run_exploration.py`)
k-means on sky directions gives one sector per probe. Each probe flies an
open path from the Sun: nearest-neighbour first, then 2-opt. The bound is the
MST of the sector.
