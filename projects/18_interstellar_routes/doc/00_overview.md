# 00 · Overview

**Question.** Treat the stars as a graph with their real 3D positions and
velocities. How short can a ship's single hop be and still reach the stars?
How much does waiting for stars to drift closer help? What does that mean for
propulsion, refuelling, sails and exploration fleets?

**Answer, in one line.** With frozen stars, reach percolates at a sharp
threshold: about 2.16 pc per hop reaches half the stars within 25 pc. Letting
colonies wait for stellar motion to close gaps lowers that by 20% within
100 kyr and 52% within 500 kyr. Below the static threshold, the stars
themselves carry the settlement front at about 21 km/s.

**Four experiments, one graph.**
1. **Frozen-star percolation**: exact minimax reach via the Euclidean MST,
   bootstrap over Gaia errors, Poisson null model, unseen brown-dwarf
   injection, and the Sun's local "pocket".
2. **Moving stars (headline)**: an exact earliest-arrival search on a temporal
   graph whose edges are open on one closed-form time window each.
3. **Engineering**: propulsion → hop range → survival time per hop; greedy
   pilot vs A*; refuelling only at exoplanet hosts; photon sails that brake on
   the target's light.
4. **Exploration**: Bjørk-style multi-probe tours on real stars, NN vs 2-opt.

**Novelty.** Prior work (Bjørk 2007, Cotta & Morales 2009, Forgan et al. 2013,
Carroll-Nellenback et al. 2019, GTOC X) used synthetic or static star fields,
or modelled the settlement front statistically. None routed the measured
Gaia 6D neighbourhood with exact time-dependent shortest paths. That is the
gap filled here.

**Why it's honest.** Every optimiser is checked by `tests/` against a
brute-force reference. The kinematics are checked against astropy and the
published Gliese 710 encounter, and every number in the paper is generated
from `results/` by `src/figures/plot_routes.py`.
