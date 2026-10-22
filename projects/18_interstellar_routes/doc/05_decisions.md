# 05 · Decisions & Alternatives

## Range as the single ship parameter
A coasting ship's reach per hop is cruise speed × survival time, so one
number, r, captures any propulsion + lifetime pair. Exp. 3 maps technologies
back onto it. This keeps the graph problems exact (minimax / earliest arrival)
instead of heuristic multi-objective searches.

## Exact algorithms over heuristics
The user asked for heuristics. The finding is that the core questions have
exact, cheap answers: the MST bottleneck, and FIFO earliest-arrival Dijkstra
on single-window temporal edges. Heuristics are then *evaluated* against the
exact optimum: the greedy pilot fails near threshold, A* saves 90–98% of
expansions, and 2-opt beats NN by 7–9%. That turns "use heuristics" into a
measured statement.

## GCNS filter instead of quality cuts
RUWE / excess-noise cuts would also remove genuine binaries, which are real
stepping stones. The GCNS is a curated, published classifier for exactly
this volume, so we inherit its validation instead of inventing cuts.

## Conditional RV fill
Unconditional draws created artificial infall (the count inside 15 pc tripled
within 1 Myr in the no-RV subset). Conditioning on the measured tangential
velocity keeps the velocity field coherent, and a unit test checks it is
unbiased.

## Linear motion, ≤1 Myr locally
Galactic-potential orbits (e.g. galpy) are an easy swap in
`src/model/kinematics.py`, but they do not matter at ≤1 Myr and ≤50 pc. The
catalogue boundary is the binding limit. The full config moves to 100 pc and
2 Myr, and `retention` in `timedep.json` reports how many stars remain.

## Not used: LKH / elkai, galpy
2-opt (numpy, ~1 s) already reproduces the literature ~10% effect. LKH would
tighten the MST gap but does not change any conclusion. Both are noted as
optional upgrades and kept out of the dependency list.

## Sail braking law
v_brake ∝ L^0.4 follows from energy balance at a closest approach proportional
to stellar radius (R ∝ L^0.2), calibrated to one published case (Heller,
Hippke & Kervella 2017). It is an order-of-magnitude scaling. The qualitative
result (dim M dwarfs cannot stop fast sails) is robust to the exponent.
