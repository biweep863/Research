# 04 · Results

Headline numbers (local config; all emitted as macros in `paper/macros.tex`).

## Resource ceiling
- Intercepted power: **173,549 TW** (≈ 174,000 TW).
- TOA horizontal mean: **340 W/m²** (= S0/4, recovered from the geometry).
- Global-mean surface GHI: **184 W/m²**.
- Cascade: cloud/aerosol reflection 23%, **ozone 2%** (mostly UV), water/aerosol
  absorption 21% → surface transmittance 54%.
- Best latitude band (subtropical desert belt): **247 W/m²** annual-mean GHI.

## Land feasibility vs demand
- Deployable area (deserts@20% + rooftops): **6.1 Mkm²**.
- Deployable power: **244 TW** = **81× world electricity**, **13.6× primary energy**.
- All world electricity fits on **75,000 km²** of panel — a **274 km square**.

## Space-based solar (SBSP)
- Ground desert panel: **40 W/m²** delivered; orbital collector: **261 W/m²**
  → **6.5×**. Driver: capacity factor **99% vs 25%**, not atmospheric bypass
  (microwave downlink already ~95% transmissive).

## Ecosystem-aware harvesting (headline)
Retained harvest value vs ecosystem reserve fraction `f_min`:

| f_min | flat retained | optimal retained | temporal gain |
|-------|---------------|------------------|---------------|
| 0.1   | 90%           | 94.8%            | 5.3%          |
| 0.2   | 80%           | 88.8%            | 11.0%         |
| 0.3   | 70%           | 82.6%            | **17.9%**     |
| 0.4   | 60%           | 75.7%            | 26.1%         |
| 0.5   | 50%           | 67.0%            | 34.0%         |
| 0.6   | 40%           | 58.2%            | **45.4%**     |

Flat and optimal reserve the *same* light for the ecosystem, so the temporal
gain is pure scheduling — obtained by yielding light to the biosphere at
low-value midday hours and harvesting the demand peaks.

## Interpretation
Raw potential, land, and even orbit are not the binding constraint on solar
energy — the biosphere is. But that constraint is a daily *integral*, so it is
schedulable: respecting *when* (not just how much) light an ecosystem needs
unlocks up to ~45% more harvestable energy at zero ecological cost. See
`paper/main.pdf`.
