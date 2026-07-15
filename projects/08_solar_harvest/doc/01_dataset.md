# 01 · "Dataset": the physical model and its constants

There is no downloaded dataset. The "data" is a first-principles physical model
built from published constants, so every number is reproducible on a clean
machine and serves as computable ground truth.

## Constants (with sources)
- **Solar constant** `S0 = 1361 W/m²` (Kopp & Lean 2011, TSI).
- **Earth radius** `R = 6.371e6 m`; surface area `4πR²`, land fraction `0.292`.
- **Obliquity** `23.45°` (drives the declination / seasonal geometry).
- **Energy-budget cascade** (Trenberth et al. 2009), as fractions of the TOA
  mean (S0/4 ≈ 340 W/m²): cloud/aerosol reflection `0.23`, **ozone absorption
  `0.02`** (broken out on request; mostly UV), water/aerosol absorption `0.21`
  → surface transmittance `0.54` → surface GHI ≈ 184 W/m².
- **Land cover** fractions (forest 0.31, grassland 0.28, desert/barren 0.19,
  cropland 0.12, ice/rock 0.09, urban 0.01).
- **Panels**: efficiency `0.20`, performance ratio `0.80`; desert GHI 250 W/m².
- **SBSP chain**: orbit availability `0.99`, space-cell efficiency `0.30`,
  DC→RF `0.80`, atmospheric microwave transmission `0.95`, rectenna `0.85`.
- **Demand**: world primary ~18 TW, electricity ~3 TW, population 8e9,
  195 countries, a "one-million city" reference.

## Generated quantities
- `src/model/insolation.py` — daily extraterrestrial insolation `H0(lat, doy)`
  (Duffie & Beckman closed form), instantaneous irradiance, band transmittance,
  and the diurnal light profile `L[h]` used by the optimiser.
- `src/model/landuse.py` — deployable area, panel power density, demand coverage.
- `src/model/sbsp.py` — orbital chain and ground-vs-space comparison.

## Deliberately conservative choices
Only 20% of desert is treated as usable, and only rooftops on urban land — the
"deployable" number is a floor, not a maximiser. The point is that even this
floor dwarfs demand, so the analysis then moves to the real constraint (the
biosphere) rather than resting on an optimistic potential.
