# Project 8 — Planetary Solar Harvest: how much Sun can we take without turning off the lights on life

## Question
Earth intercepts ~174,000 TW of sunlight — far more than humanity uses — so why
isn't solar "solved"? Because that light is already running the biosphere. This
project builds a first-principles budget from the solar constant to the surface,
compares deployable-land energy and an orbital collector against demand, and
(the headline) shows how to schedule harvesting around each ecosystem's daily
light budget to unlock more energy at zero ecological cost.

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/08_solar_harvest
make test           # closed-form physics vs numeric integration; water-filling vs LP
make run            # resource, SBSP, and optimisation experiments
make figures        # regenerate figures + paper/macros.tex
make paper          # compile paper/main.pdf
```

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
Finer latitude bands, 365-day means, half-hour resolution, wider `f_min` sweep —
no code changes, everything is config-gated. (CPU-only: no GPU needed.)

## Files
- `src/model/insolation.py` — solar constant → surface irradiance (geometry + cascade)
- `src/model/landuse.py` — deployable area, panel power, demand coverage
- `src/model/sbsp.py` — orbital collection chain vs ground panels
- `src/optimize/light_budget.py` — ecosystem-aware harvesting (water-filling + LP check)
- `src/experiments/run_{resource,sbsp,optimize}.py` — write `results/*.json`
- `src/figures/plot_solar.py` — 5 figures + LaTeX macros
- `paper/main.tex` — two-column scientific paper
- `doc/` — methodology, experiments log, decisions

## Headline
Potential dwarfs demand (81× world electricity on deserts + rooftops); the Sun
is not the bottleneck. The biosphere is — and because its light need is a daily
*integral*, scheduling harvest around it (water-filling) recovers up to ~45%
more energy at identical ecological cost.
