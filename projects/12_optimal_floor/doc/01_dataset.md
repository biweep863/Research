# 01 · "Dataset" — a parametric building

There is no external dataset. Everything is generated from the configuration
and a seed, so every number reproduces on a clean machine.

## Building and car (`configs/local_small.yaml`)
- Floors 1..N (1 = lobby), N ∈ {10, 20, 40} in the closed form, 20 in most
  simulations, 15 for the (α, rate) map, 12 for the hospital archetype.
- Kinematics (CIBSE Guide D mid-rise values): rated speed 1.6 m/s,
  acceleration 1.0 m/s², floor height 3.3 m, door cycle 4 s, transfer 1.2 s
  per passenger. Run time n·H/v + v/a; at 1.6 m/s a single floor reaches
  rated speed so this equals the trapezoid exactly (`Kinematics.trapezoid_error`).
- Capacity 10 passengers per car.

## Trips of the resident (`src/model/trips.py`)
Chained outbound/return trips: outbound to the lobby with probability α,
otherwise to a uniform other floor ≥ 2; return home. `od_distribution` gives
the exact (origin, destination) probabilities used by the closed form; the
generator reproduces them in simulation.

## Background traffic (`src/sim/elevator.py::poisson_background`)
Poisson arrivals at λ per minute with the same α: up from or down to the
lobby (half each), else between two distinct floors ≥ 2.

## Sampling plan (local config)
- Closed form: 3 heights × 6 α × 4 parking policies, all 2..N home floors.
- Traffic: 2 parking × 5 rates × 19 home floors × 4 replications of 4 h with
  a resident probe every ~2 min (~115 trips per run).
- Stairs: 6 variants × 19 floors × 4 reps. Fleet: 3 fleets × 19 × 4.
- Map: 6 α × 4 rates × 14 floors × 3 reps of 3 h; archetypes × 6 reps.
- Common random numbers: replication r uses seed `seed + r` for every home
  floor so the background is identical across floors.

`configs/gpu_full.yaml` widens every grid (7 heights, 12 α, 9 rates, up to 8
cars), lengthens runs to 12 h, and uses 10 replications.
