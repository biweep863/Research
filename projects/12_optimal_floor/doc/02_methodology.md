# 02 · Methodology

```
configs/*.yaml ─► src/model (kinematics, trips, stairs)
                     ├─► src/analytic/closed_form  E[T(h)] exact, regret, band
                     └─► src/sim/elevator (DES) ─► src/sim/evaluate (resident probes)
src/experiments/run_{analytic,traffic,stairs,fleet,robust} ─► results/*.json
src/figures/plot_floor ─► paper/figures/*.pdf + paper/macros.tex ─► latexmk ─► paper/main.pdf
```

## Closed form (`src/analytic/closed_form.py`)
T(o,d | c) = run(|c−o|) + stop + run(|o−d|), counted from the button press to
the car's arrival at the destination (one dwell inside; the destination door
opening is not counted on either side of the sim/analytic comparison). Car
position c by policy: lobby (1), middle (⌈(N+1)/2⌉), last (= o, the resident's
own previous destination), random (uniform 1..N). E[T(h)] = Σ p(o,d)·T with
the exact chained-trip distribution. With stairs: per pair, min(elevator,
stairs) if the climb is allowed. Outputs: h*, best, regret of the middle /
worst / top floor (s and %), 5 % indifference band.

## Simulator (`src/sim/elevator.py`)
Event heap with passenger arrivals and car events. Cars step one floor per
H/v seconds; a run from standstill adds v/a. Stop decision at a floor: car call
there; hall call in the car's direction (capacity permitting); any hall call if
no calls ahead (reversal floor); terminal floors; parking target. Stop handles
alighting (t_alight recorded at arrival), boarding in the car's direction
(t_board recorded at arrival), and dwells t_door + t_pass × transfers. After a
dwell: continue if calls ahead, reverse if calls behind, otherwise go to the
nearest pending hall call, else park (lobby/middle) or idle in place (last).
New hall calls wake the nearest idle car. Bookkeeping per passenger: wait,
ride, intermediate stops while on board.

## Resident evaluation (`src/sim/evaluate.py`)
The resident is inserted every ~trip_interval seconds after a warm-up; the
trip chain is the same generator as the closed form; stairs are chosen on the
*expected* elevator time (closed form, same parking policy, zero traffic);
otherwise the resident becomes a passenger and the sim advances until they
alight. Records mean, p90, std, mean wait/ride/stops, fraction walked, and
all cars' positions at each call.

## Metrics
mean trip time; p90; regret(h) = E[T(h)] − min E[T] in % of the best; 5 %
indifference band = floors within 1.05 × best; h*_p90 = argmin of p90.

## Verification (`tests/test_model.py`, 8 tests, < 1 s)
Trapezoid vs linearised kinematics; trip chaining and α limits; closed form
gives the median at α = 0 with stay-put parking and floor 2 at α = 1 with
lobby parking; simulator equals the closed form per trip to 1e-6 at zero
traffic and within 6 % in mean over a run; every background passenger is
served with consistent timestamps and traffic adds stops; stairs rule and
asymmetry; regret bookkeeping and bootstrap CI.
