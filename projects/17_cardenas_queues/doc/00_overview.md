# 00 · Overview

## Origin
A conversation with another LLM estimated that Av. Lázaro Cárdenas (Monterrey ↔
San Pedro) has "80–100 accesses" and "130–160 feeder lanes" that empty into
"5–6 lanes per direction", and offered that ratio as the explanation for its
congestion. Those numbers had no source. This project (a) measures them from
OpenStreetMap and (b) tests whether the ratio explains anything.

## Question
Where does the peak-hour queue form on Lázaro Cárdenas, and is that place
predicted by the ratio of feeder lanes to corridor lanes, or by the utilisation
(demand / capacity) of each merge?

## Hypothesis
A lane count is not a demand. Most feeder lanes are one-lane residential streets
and driveways that carry a small fraction of a lane's saturation flow. The queue
should form where demand exceeds capacity at a merge, i.e. where ρ > 1, and back
up from there (spillback).

## Falsification rule (set before running)
Across a Monte Carlo over every unmeasured assumption, if the lane ratio finds
the CTM's principal bottleneck at least as often as ρ does, the hypothesis fails.

## Mechanism (intuition)
- Queueing: a server with utilisation ρ < 1 has finite queue; ρ ≥ 1 during a
  peak gives a fluid queue that grows at rate λ − μ (Newell).
- On a corridor, servers are in tandem: the first saturated merge meters the
  flow downstream and its queue spills upstream across feeders and exits.
- Lanes enter capacity (denominator) and demand (numerator) very differently:
  a corridor lane is worth ~1900 veh/h; a residential feeder lane ~100.

## Outcome (local run)
- OSM confirms the counts: 99 street approaches, 133 feeder lanes (243 with
  110 driveways) into 5 lanes per direction. The LLM's ranges were right.
- Feeder lanes carry ~150 veh/h/lane (~8 % of saturation) at base rates.
- The local lane ratio never locates the bottleneck (0 %); ρ does in 100 % of
  congested draws and classifies congestion perfectly.
- The location is geometric: a 5 → 3-lane merge at 4.4–4.6 km between the
  Alfonso Reyes overpass and the Arroyo Seco bridge. Filling that gap moves the
  queue (see caveat below).
- +1 lane over 600 m at the merge: −37 % median delay, better in 100 % of runs.
- Retornos (17 U-turns / median crossings) count as accesses of the receiving
  direction and exits of the giving one.
- Roadside estimate at Alfonso Reyes (both lanes towards San Pedro; 570–770
  veh/h vs 2,100 assumed): westbound ρ 1.04 → 0.88; no westbound queue.
- Working day 7–20 h, tidal: WB never queues (ρ 0.88 at 7:00) — the observer
  confirms little or no morning queue towards San Pedro; EB queues from ~15:45
  and is still queued at 20:00 (no field check yet).
- Caveat: with the lane gap filled, the lane ratio beats ρ (70 % vs 52 %, n = 23).

See `04_results.md` and `paper/main.pdf`.
