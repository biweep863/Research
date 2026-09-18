# 05 · Decisions & Alternatives

## Closed form first, simulator second
The closed form makes the mechanism legible (parking position + distance) and
gives an exact target for validating the simulator at zero traffic. Rejected:
simulation only (no ground truth), or an M/M/1-style queueing model of the
car (does not capture collective control's sweep behaviour, which turned out
to be the interesting effect under load).

## Chained outbound/return trips
Alternative: independent random (o, d) pairs. Chaining is physically right
(you come back), makes the OD distribution exactly symmetric, and is the only
way the stay-put parking policy is well defined without traffic.

## Floor-by-floor cars with a per-run start penalty
Alternative: exact trapezoidal runs between stops with mid-run re-planning.
At 1.6 m/s a single floor reaches rated speed, so the linearised time equals
the trapezoid; the floor-by-floor model lets moving cars pick up calls in
passing (collective control) with a simple event loop. `run_trapezoid` is kept
for tests and for high-speed configs where the two differ.

## One dwell per trip in the timing convention
Boarding and alighting are stamped at the car's arrival. The trip therefore
contains the origin dwell but not the destination door opening, on both sides
of the comparison. Documented in the closed-form docstring after the first
mismatch.

## Common random numbers across floors
Independent seeds per floor produced noisy optima on flat curves. Sharing the
background draw across floors within a replication is the standard
variance-reduction choice for comparing alternatives and is stated in the
paper's method.

## Collective control + nearest idle car, not destination dispatch
Destination dispatch and learned parking would change the numbers but not the
structure; the simple controller keeps every effect attributable and the
simulator validatable. Listed as a limitation and a `gpu_full` extension.

## Stairs decided on expected elevator time
The resident does not know where the car is; deciding on the closed-form
expectation (same parking policy, zero traffic) is the honest model of a
person's rule of thumb. A traffic-aware resident is a natural extension.

## Regret and indifference band as the "does it matter?" metric
Reporting only h\* would hide that the curve is flat in some regimes and
steep in others. The 5 % band and the middle/top-floor regrets make the
answer to the user's second question explicit.

## Archetypes are illustrative
Residential / office / hospital parameters (α, rate, cars, parking) are
plausible round numbers, not measured; they exist to show three regimes of
the map on concrete curves.

## Reporting the failed hypothesis
We expected the up/down stairs asymmetry to move the optimum up. It does not
(symmetric stairs give the same shift). Kept and explained rather than
removed.
