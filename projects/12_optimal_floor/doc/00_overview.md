# 00 · Overview

**Question.** Which floor of a building should you be based on to minimise
the time you spend moving around it — the middle, as the median argument
says, or does it not matter? The elevator is a *mechanism*, not a distance:
the car has to come from wherever it was parked, most trips go to or from the
lobby, other passengers add stops, and for a floor or two the stairs beat
waiting (and going down is cheaper than going up).

**Hypothesis.** The middle-floor answer holds only when no trips go to the
lobby and the car waits where you left it. With lobby trips and lobby
parking the optimum drops to the lowest residential floor; traffic flattens
the curve; stairs pull the optimum up (we expected the up/down asymmetry to
drive this); more cars make the floor matter less.

**Falsification.** If the closed form at α = 0 with stay-put parking did not
give the median floor, or if the simulator did not match the closed form at
zero traffic within a few percent, the model would be wrong. If the optimum
under lobby trips were not monotone toward the ground, the mechanism story
would be wrong. Two of our own expectations were falsified (see results):
the stairs asymmetry does not matter, and more cars make the floor matter
*more*.

**Five stages.**
1. Closed form: one car, no traffic, four parking policies, α and N sweeps.
2. Discrete-event simulation with Poisson background traffic (validated
   against the closed form): mean, p90, wait/ride/stops decomposition, and the
   measured stationary car position under stay-put parking.
3. Stairs with up/down asymmetry and climb limit.
4. Fleet size with load scaling per car.
5. The (α, traffic) map with indifference bands, and three archetypes.

**Headline.** Median floor only at α = 0 with stay-put parking (floor 11 of
20). With a quarter of trips to the lobby and lobby parking the optimum is
floor 2 and stays there; at α = 0.9 the middle floor costs 163 % more, the
top 381 %. Traffic flattens the curve (middle-floor regret 83 % → 10 % at
2/min) and only at saturation drifts the optimum up (floor 11 at 4/min: low
floors are served last on the down sweep). Stairs cut the best time 49 → 38 s
and move a lobby-free building's optimum from floor 5 to 7 — symmetric stairs
do the same. One to four cars: middle-floor regret 16 % → 30 %. Map: 16 of 24
cells put the optimum on floor 2.

**Reproduce.**
```bash
cd projects/12_optimal_floor
make test      # ~1 s: 8 tests
make all       # ~1 min: 5 experiments -> results/*.json, 6 figures, 128 macros, paper/main.pdf
```
