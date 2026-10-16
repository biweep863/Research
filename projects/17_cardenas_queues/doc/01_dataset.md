# 01 · Data

## OpenStreetMap extract
- Source: `api.openstreetmap.org/api/0.6/map`, 25 tiles of 0.01° along a
  hand-placed spine (config `osm.spine_lonlat`, taken from the trunk ways named
  "Avenida Lázaro Cárdenas"). Overpass is tried first (it was saturated on
  2026-10-12, so the tiled API was used).
- Cached in `data/osm_raw.json` (git-ignored, ~2.5 MB, 5,841 highway ways,
  29,628 nodes). Delete it to refetch. License: ODbL, © OpenStreetMap contributors.
- Fetched timestamp is stored in `results/inventory.json` (`osm_fetched`).

## Corridor extraction (`src/data/corridor.py`)
| Item | Rule |
|---|---|
| Corridor ways | name contains "Lázaro Cárdenas", not "General/Prolongación/Calle", highway trunk/primary/secondary/tertiary(+links), median offset ≤ 120 m from spine |
| Express vs frontage | `trunk`/`primary` = express; others = lateral (frontage) |
| Direction | one-way travel direction along chainage: WB = towards San Pedro, EB = towards Garza Sada |
| Approach | non-corridor street way touching a corridor node whose one-way rules let it *deliver* vehicles there |
| Internal links | chains of unnamed / `_link` ways that only touch the corridor (52 removed) |
| Demand class | major (motorway–secondary + links), tertiary, minor (residential/unclassified), service; links take the class of the best road they reach |
| Lanes | `lanes:forward/backward`, else `lanes` (÷2 if two-way), else class default per direction |
| Signals | `highway=traffic_signals` on corridor nodes (only 4, at Alfonso Reyes) + inferred at frontage junctions fed by major or ≥2-lane tertiary |

## Inventory (local run)
- 10.0 km; 3 express + 2–3 frontage lanes per direction (median 5, range 2–7).
- 209 approaches = 99 streets + 110 service (driveways, plazas). 33 crossing, 176 terminal.
- Feeder lanes: 133 streets (major 30, tertiary 46, minor 57) + 110 service = 243.
- Lane tags: 100 % of corridor ways, 22 % of approaches (rest = defaults).

## Known gaps
- Signals under-mapped → inferred (assumption, see decisions).
- The 5 → 3 merge at 4.36–4.55 km is a mapped feature (`secondary_link`
  "Lateral Avenida Lázaro Cárdenas" joins the trunk) — **check on site**.
- Traffic counts: only one roadside estimate (below). `field_counts.csv`
  (project root, versioned) overrides the demand model for a counted junction
  (`src/data/counts.py`).

## Retornos
Internal chains whose origin is on one carriageway and destination on the other
(U-turns, median crossings): 17. Each is an approach (class `retorno`, 20–200
veh/h/lane) of the receiving direction and an exit of the giving one.
Express↔frontage transfers of the same direction stay internal.

## Field estimate (2026-10-12)
Av. Alfonso Reyes arrives from the south with 4 lanes; 2 join Lázaro Cárdenas
towards San Pedro (WB). On Alfonso Reyes' red only one moves (continuous turn):
7–10 veh/min; on green both: 14–18 veh/min. With the side green share 0.35:
570–770 veh/h (model had ~2,100, mostly a two-lane slip ramp rated as a major
feeder). The observer also reports little or no morning queue towards San Pedro.
Matched within 80 m (named street + unnamed ramps, retornos excluded). Counts
are what entered: demand only if no queue remains on Alfonso Reyes after green.

## Synthetic fallback
`src/data/synthetic.py`: 10 km, 3+2/3 lanes, junctions every 1.2 km, Poisson
minor/service accesses. Used automatically when OSM is unreachable; results then
carry `synthetic_data: true`. Unit tests always use it.

## Local vs full config
| | local_small | gpu_full |
|---|---|---|
| Monte Carlo draws | 48 | 400 |
| Intervention draws | 24 | 200 |
| DES vehicles × reps | 40k × 5 | 200k × 20 |
| Demand grid | 5 scales × 3 peaks | 9 × 5 |
Both CPU-only (numpy); "gpu_full" is just the large configuration.
