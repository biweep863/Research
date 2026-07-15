# 02 · Methodology

## Pipeline
```
configs/*.yaml
      │
      ▼
src/model/{insolation,landuse,sbsp}.py   ← first-principles physics
      │
      ├── run_resource.py  → results/resource.json   (cascade, bands, land, demand)
      ├── run_sbsp.py      → results/sbsp.json        (orbital chain, comparison)
      └── run_optimize.py  → results/optimize.json    (3 policies × f_min sweep)
                                   │
                                   ▼
                 src/figures/plot_solar.py
                     → paper/figures/*.pdf  +  paper/macros.tex
                                   │
                                   ▼
                 paper/main.tex (all prose numbers come from macros)
```

## Solar geometry (closed form, tested)
Daily extraterrestrial insolation on a horizontal surface:
`H0 = (86400/π)·S0·E0·[cos φ cos δ sin ωs + ωs sin φ sin δ]`,
with declination `δ(doy)`, eccentricity factor `E0(doy)`, and sunset hour angle
`ωs = arccos(−tan φ tan δ)` (0 at polar night, π at midnight sun). Validated in
`tests/` against direct numerical integration of the instantaneous sun-angle to
< 1e-3 relative, and the global area/time mean recovers `S0/4` to < 0.5 W/m².

## Loss cascade
The global-mean surface irradiance is obtained by applying the published
reflection/absorption fractions to the TOA mean, kept as an ordered waterfall so
the ozone term is explicit. Per-band surface GHI uses a latitude transmittance
profile (equatorial cloud belt low, subtropical desert belt highest).

## Ecosystem light-budget optimisation (headline)
For cells `i` and hours `h`: maximise harvest value `Σ w·a·L` subject to each
cell reserving a daily light integral `Σ(1−a)L ≥ f_min·ΣL`. Because reserving a
unit of light in hour `h` forfeits value `w[h]`, the optimum reserves from the
lowest-value hours first — greedy **water-filling**, proved optimal for the
per-cell LP and checked against `scipy.optimize.linprog` to 1e-6.

Three policies: **no-cap** (upper bound, ecosystem-blind), **flat** (uniform
`f_min` every hour — feasible but wasteful), **optimal** (water-filling). Flat
and optimal reserve the *same total* light, so their difference is pure
scheduling gain. Harvest value is weighted by a diurnal net-demand ("duck
curve") so that midday harvest is worth less than the demand peaks.

## Reproducibility
`make test` (physics vs numeric integration, water-filling vs LP) must pass
before `make run`. All figures + prose numbers regenerate from the JSON via
`plot_solar.py`; nothing is hand-typed into the paper.
