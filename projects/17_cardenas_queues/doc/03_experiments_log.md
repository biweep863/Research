# 03 · Experiments log

| Date | Config | Command | Outcome |
|---|---|---|---|
| 2026-10-12 | local_small | `make data` | Overpass busy (406/timeouts) → OSM API, 25 tiles, 5,841 ways |
| 2026-10-12 | local_small | `run_inventory` (v1) | 196 approaches but 17–23k veh/h side demand: express↔frontage links counted as feeders |
| 2026-10-12 | local_small | `run_inventory` (v2) | connected-component rule removes 52 internal links; links inherit class of reached road |
| 2026-10-12 | local_small | `run_validation` | DES vs Erlang C/M/D/1 3.2 %; signal vs Webster 8 %; CTM = fluid (Newell) |
| 2026-10-12 | local_small | `run_peak` (FIFO=1) | EB delay 51,700 veh h vs fluid 11,000 → exits blocked by FIFO dominate |
| 2026-10-12 | local_small | `run_peak` (partial FIFO φ) | φ ∈ [0,1] added; EB 3,200–51,700 veh h; head location unchanged |
| 2026-10-12 | local_small | `run_montecarlo` | ρ hit 99 %, lane ratio 0 %, cumulative 70 % → added feeder-only and capacity-only predictors |
| 2026-10-12 | local_small | `run_montecarlo` | capacity alone 100 % → geometric bottleneck; added gap-filled scenario |
| 2026-10-12 | local_small | `run_interventions` (v1) | metering +34 %, consolidation +213 % (demand dumped on next access) |
| 2026-10-12 | local_small | `run_interventions` (v2) | capacity drop (8 %) + collector consolidation; lane −34 %, metering +6 %, consolidation −3 % |
| 2026-10-12 | local_small | `make all` | 18 tests, ~70 s run, 7 figures, 152 macros, 7-page PDF |
| 2026-10-12 | local_small | `make all` + `field_counts.csv` (v1, all WB) | Alfonso Reyes WB estimate 570–770 veh/h (assumed 2,100): WB ρ 1.06 → 0.90, WB queue gone at base, WB congested in 21 % of draws (was 52 %); EB unchanged; ρ hit 100 %, lane +1 −36 % |
| 2026-10-12 | local_small | retornos + per-direction Alfonso Reyes + `run_daily` | 17 retornos; WB ρ 0.83 (no queue all day), EB queue 15:44 → still at 20:00; ρ hit 100 %; gap-filled: lane ratio 91 % vs ρ 45 % (n = 22); +1 lane −38 %; ~90 s |
| 2026-10-12 | local_small | Alfonso Reyes corrected: both lanes WB (continuous turn on red) | WB ρ 0.88 at 7:00, no queue (matches field); EB queue 15:44 → 20:00+; ρ hit 100 %; gap-filled lane ratio 70 % vs ρ 52 % (n = 23); +1 lane −37 % |

## Observations
- The main bottleneck sits where OSM maps the frontage road merging into the
  express lanes (4.36–4.55 km). Verified in OSM tags: `secondary_link`
  "Lateral Avenida Lázaro Cárdenas" ends on the trunk; frontage resumes after
  4.55 km; trunk bridge over "Canal del Arroyo Seco" at 4.8–5.1 km.
- Alfonso Reyes does cross the corridor (3.85 km), with the only mapped signals.
- With strict FIFO the eastbound queue fills the whole upstream corridor and
  blocks every exit; spillback, not the merge itself, makes most of the delay.
- The equal-rates counterfactual congests rarely (5/96): spreading the same total
  side demand over all lanes evenly loads the merge less.
- Gap-filled runs reverse the ranking: without the lane drop the queue head sits
  where many accesses (incl. retornos) join in a short stretch; steady-state ρ
  points to the neighbouring segment. Small sample (~23 runs, one direction).
