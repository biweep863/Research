# 17 · Where the queue forms on Av. Lázaro Cárdenas (Monterrey)

## Question
Where does the peak-hour queue form on Av. Lázaro Cárdenas (Garza Sada ↔ Gómez
Morín, ~10 km), and is that place predicted by the ratio of feeder lanes to
corridor lanes ("130–160 lanes into 5–6"), or by the utilisation ρ = demand /
capacity of each merge?

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/17_lazaro_cardenas_queues
make test        # 19 tests (<1 s, no network)
make data        # OSM extract (cached in data/) -> results/inventory.json
make run         # 7 experiments -> results/*.json (~90 s)
make figures     # 8 figures + paper/macros.tex
make paper       # paper/main.pdf
```
Without network the pipeline falls back to a synthetic corridor and marks
`synthetic_data: true`.

## Field counts
`field_counts.csv` replaces the assumed demand of a counted junction (vehicles per
minute on the cross street's green and red, ranges allowed, plus lanes with a
continuous turn; one row per direction fed; format in `src/data/counts.py`). It
holds one roadside estimate at Av. Alfonso Reyes: two lanes join towards San Pedro
(WB); on red only the continuous-turn lane moves (7–10 veh/min), on green both
(14–18) → 570–770 veh/h (≈2,100 had been assumed).
Add rows and rerun `make all`.

## A working day (7:00–20:00)
`configs/*.yaml → daily`: hourly demand multipliers per direction (tidal: at 7:00
towards San Pedro, evening towards Garza Sada). `run_daily.py` reports hourly ρ
against the merge's maximum capacity (~5,400–5,500 veh/h), queue length, when the
queue starts and whether it clears.

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml   # 400 MC draws, 200 intervention draws, finer grids (CPU)
```

## Files
- `src/data/osm.py` — Overpass / tiled OSM API download, cache
- `src/data/corridor.py` — corridor ways, cross-section, approaches, retornos, internal links, junctions
- `src/data/network.py` — cells, capacities, feeder queues, exit balancing, interventions
- `src/data/{synthetic,counts}.py` — offline fallback; field counts (`field_counts.csv`)
- `src/analysis/` — M/M/c, M/D/1, Kingman, tandem; capacity, Webster, HCM; fluid queue
- `src/sim/ctm.py` — Cell Transmission Model (priority merge, partial-FIFO diverge, capacity drop, ALINEA)
- `src/sim/des.py` — vehicle-level DES (multi-server FIFO, fixed-time signal)
- `src/experiments/run_{inventory,analytic,validation,peak,montecarlo,interventions,daily}.py`
- `src/figures/plot_corridor.py` — 8 figures + 195 LaTeX macros
- `paper/main.tex` — two-column paper (7 sections)
- `doc/` — overview, data, methodology, experiment log, results, decisions

## Headline
OSM confirms the lane count (99 street approaches, 133 feeder lanes, 5 lanes per
direction) but feeder lanes carry ~8 % of a lane's saturation flow. The local lane
ratio never finds the bottleneck (0 %); ρ finds it in 100 % of congested draws. The
bottleneck is a 5 → 3-lane merge at ~4.4 km between the Alfonso Reyes overpass
and the Arroyo Seco bridge (check on site!). +1 lane there: −37 % median delay.
With the Alfonso Reyes estimate, westbound peaks at ρ = 0.88 at 7:00 and does not
queue (ρ 1.04 with assumed demand) — matching the field report of little or no
morning queue towards San Pedro. Eastbound queues from ~15:45 and has not cleared
at 20:00. If the lane drop is not real, the lane ratio locates the queue better
than ρ (70 % vs 52 %, 23 runs).
