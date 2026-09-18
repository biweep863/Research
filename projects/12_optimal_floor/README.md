# Project 12 — Up or down? The optimal floor when the elevator has to come and get you

## Question
Which floor should you be based on to minimise the time you spend moving
around a building — the middle, as the median argument says, or does it not
matter? The elevator is a mechanism: it has to come from where it was parked,
most trips go to or from the lobby, other passengers add stops, and for a floor
or two the stairs win (and down is cheaper than up).

## The five stages
1. **Closed form** — one car, no traffic; four parking policies (lobby, middle,
   stay-put, random); sweep lobby-trip fraction α and building height.
2. **Traffic** — discrete-event simulator (collective control, nearest-car
   dispatch, capacity) with Poisson background passengers, validated against
   the closed form; wait/ride/stops decomposition; measured car position.
3. **Stairs** — up/down asymmetric costs and a climb limit.
4. **Fleet** — 1, 2, 4 cars with load scaling per car.
5. **Map** — optimal floor, 5 % indifference band and heuristic regret over
   (α, load); residential / office / hospital archetypes.

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/12_optimal_floor
make test           # 8 tests (<1 s)
make run            # 5 experiments -> results/*.json (~30 s)
make figures        # 6 figures + paper/macros.tex
make paper          # compile paper/main.pdf
```

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
Finer α/rate grids, taller buildings, up to 8 cars, 12 h runs, 10 replications.
Pure Python/numpy on CPU.

## Files
- `src/model/{kinematics,trips,stairs}.py` — run/stop times, chained trip generator + exact OD distribution, stairs rule
- `src/analytic/closed_form.py` — E[T(h)] per parking policy, regret and indifference band
- `src/sim/{elevator,evaluate}.py` — discrete-event simulator and resident probe
- `src/experiments/run_{analytic,traffic,stairs,fleet,robust}.py` — write `results/*.json`
- `src/figures/plot_floor.py` — 6 figures + 128 LaTeX macros
- `paper/main.tex` — two-column paper (7 sections)
- `doc/` — overview, data, methodology, experiments log, results, decisions

## Headline
The middle floor is optimal only when no trip goes to the lobby and the car
waits where you left it. With lobby trips and lobby parking the optimum is the
lowest residential floor, and at a residential mix the middle floor costs 163 %
more time (top floor 381 %). Traffic flattens the curve (middle regret 83 → 10 %
at 2/min) and only at saturation drifts the optimum up. Stairs cut the best time
49 → 38 s and shift a lobby-free building's optimum up — but symmetric stairs do
the same. More cars make the floor matter *more* (middle regret 16 → 30 %).
Down, unless you never leave.
