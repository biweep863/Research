# 04 · Results (local_small, Alfonso Reyes estimate, retornos included)

Formal write-up: `paper/main.pdf`. All numbers below come from `results/*.json`.

## Inventory vs the LLM claims
| | LLM (unsourced) | OSM 2026-10-12 |
|---|---|---|
| accesses | 80–100 | 99 street approaches (97 clustered) + 110 driveways |
| feeder lanes | 130–160 | 133 street lanes; 243 with driveways |
| lanes per direction | 5–6 | median 5 (3 express + 2–3 frontage) |
| length | 9–10 km | 10.0 km |

## Demand vs lanes
Feeder lanes deliver ~140 (WB) / ~157 (EB) veh/h per lane at base rates — 7–8 % of
saturation. At saturation the same lanes would deliver ~211k / 251k veh/h.

## Where the queue forms
- ρ > 1 only at the 5 → 3 merge: EB 1.36 at 4.5 km. WB peaks at 0.88 there
  (1.04 with assumed demand at Alfonso Reyes).
- CTM 2-h peak: EB 15,240 veh h, queue 6.0 km; WB no queue (assumed demand:
  1,360 veh h, 2.1 km).
- Share of Monte Carlo draws that congest: WB 23 %, EB 100 %.
- Working day 7–20 h (tidal): WB ρ 0.88 at 7:00, no queue (MC: 12 % of days;
  field: little or no morning queue — consistent);
  EB queue from 15:44, 6 km, still present at 20:00; merge capacity ~5,400–5,500 veh/h.
- Diverge blocking φ: EB 4,170 (φ=0) – 61,510 (φ=1) veh h.
- Gap filled: WB no queue; EB head moves to 2.3 km.

## Predictor test (48 draws × 2 directions)
| predictor | hit rate, OSM | hit rate, gap filled |
|---|---|---|
| local lane ratio | 0 % | 70 % |
| cumulative lane ratio | 81 % | 0 % |
| feeder lanes alone | 0 % | 0 % |
| capacity alone | 100 % | 0 % |
| ρ | 100 % | 52 % |
Gap-filled: only 23 congested runs, all EB, mostly one segment.
max ρ ≥ 1 classifies congested runs: accuracy 100 %, AUC 1.00. Hypothesis not falsified.

## Interventions (26 congested draw-directions, median change in total delay)
| intervention | median | better | worse |
|---|---|---|---|
| +1 lane, 600 m at merge | −37 % | 100 % | 0 % |
| ramp metering (ALINEA) | +15 % | 35 % | 65 % |
| consolidate minor accesses | −4 % | 77 % | 4 % |
| retime signals | ≈0 % | 8 % | 8 % |
Metering shifts delay from the corridor into side streets (EB base: corridor
8,920 → 4,340, feeders 6,310 → 10,130 veh h).

## Caveats
One roadside estimate only; absolute delays depend on assumed rates and on φ. Robust: location,
predictor ranking, sign of the lane intervention.
