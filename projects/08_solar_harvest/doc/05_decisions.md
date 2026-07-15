# 05 · Decisions & Alternatives

## First-principles model instead of gridded datasets
We chose a self-contained physical model (published constants, closed-form
insolation geometry, standard energy-budget fractions) over downloading NASA
POWER / land-cover / population grids. Rationale: the numbers become computable
ground truth (testable against numerical integration and an exact LP), the
project reproduces on a clean machine in seconds, and the *structure* of the
argument — not decimal-place fidelity — is what carries the message. The
`gpu_full` config and the limitations section flag the data-swap path.

## The ecosystem constraint is a daily integral, not an instantaneous floor
This is the whole point of the paper. If ecosystems needed a fixed fraction of
light *every hour*, there would be no scheduling freedom and the flat policy
would be optimal. Because photosynthetic light requirements are naturally a
daily-integral (daily light integral, DLI) quantity, the harvest schedule has
temporal slack — and exploiting it is the contribution.

## Water-filling, proved optimal and LP-checked
Rather than an RL agent (as in project 07), the ecosystem problem has an exact,
transparent optimum: reserve the ecosystem's quota from the lowest-value hours.
We keep the greedy solver *and* validate it against `scipy.optimize.linprog` so
the "optimal" label is earned, not asserted.

## Duck-curve demand weight
The temporal gain depends on how much harvest value varies across daylight
hours. A mild two-peak demand curve gave only 1–14% gain; we switched to a
realistic solar-grid "duck curve" with a deep midday value trough (midday energy
is genuinely low-value when solar already saturates the grid), giving 5–45%.
This is a modelling choice, documented honestly: the qualitative claim
(scheduling helps, monotonically in f_min) holds under any non-flat demand; the
magnitude scales with demand variability.

## SBSP framed as capacity factor, not "free energy above the atmosphere"
The user's intuition was that an orbital layer captures light "before the
atmosphere" for more power. We quantified this and corrected it: atmospheric
absorption is a small part of the story (microwave downlink is ~95%
transmissive); the real orbital advantage is the ~99% capacity factor vs ~25%
on the ground. We report the full conversion chain so the 6.5× is not oversold.

## Conservative deployable-land filter
Only 20% of desert plus urban rooftops count as deployable — a floor, not a
maximiser. Even this floor beats demand 81×, which is precisely why the paper
pivots from "can we?" to "how do we share the light with life?"
