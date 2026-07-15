# 00 · Overview

**Question.** How much of the Sun's energy can humanity harvest, how does it
compare to demand, and how do we take it without starving the ecosystems that
the same sunlight sustains?

**Answer, in one line.** The Sun is not the bottleneck (Earth intercepts
~174,000 TW; a conservative desert-and-rooftop footprint already beats world
electricity demand ~81×). The binding constraint is the biosphere's own light
budget — and because that budget is a *daily integral*, harvesting can be
*scheduled* around it, recovering up to ~45% more energy at zero ecological cost.

**Four threads, one framework.**
1. **Resource ceiling** — first-principles cascade from the solar constant
   (1361 W/m²) to surface irradiance (~184 W/m²), with the ozone layer's share
   broken out; resource mapped by latitude band (desert belt is the maximum).
2. **Land feasibility** — deployable area under a "do no ecological harm" filter
   (deserts + rooftops), converted to power and compared to real demand.
3. **Orbital alternative (SBSP)** — a collector "layer" above the atmosphere;
   quantified as a capacity-factor play, not free energy.
4. **Ecosystem-aware harvesting (headline)** — a constrained optimisation whose
   optimum is water-filling: reserve the ecosystem's light in low-value hours,
   harvest the peaks.

**Why it's honest.** Everything is first-principles physics with published
constants — no fitted datasets. The insolation geometry, the energy-budget
cascade, and the optimisation optimum are all computable ground truth, checked
in `tests/` against numerical integration and an exact LP solver.

See `04_results.md` for the headline numbers and `paper/main.pdf` for the
write-up.
