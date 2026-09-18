# 04 · Results

Headline numbers (local config; all emitted as macros in `paper/macros.tex`).

## Closed form (one car, no traffic, 20 floors)
| α | lobby | middle | last | random |
|---|---|---|---|---|
| 0.00 | h*=7 (band 3–10) | 10 | **11** (median) | 11 (band 8–13) |
| 0.25 | **2** | 10 | 8 | 9 |
| 0.50 | **2** (middle +46 %) | 10 | 2 | 6 |
| 0.90 | **2** (middle +163 %, top +381 %) | 2 | 2 | 2 |
| 1.00 | **2** (middle +231 %) | 2 | 2 | 2 |
40 floors, α = 0.5, lobby parking: h* = 2, middle +67 %; α = 0, random: h* = 21.

## Traffic (one car, lobby parking, α = 0.7, 20 floors)
- Validation vs closed form at zero traffic: **2.3 %** mean relative error.
- Rate 0 / 0.5 / 1 / 2 / 4 per min: best **20 / 35 / 49 / 83 / 160 s**; h* =
  **2 / 2 / 2 / 2 / 11**; middle-floor regret **83 / 38 / 22 / 10 / 4 %**;
  p90-optimal floor 9 / 4 / 4 / 4 / 4.
- At 4/min: wait 148 s (floor 2) → 98 s (floor 16) → 166 s (floor 20); ride
  26 → 80 s; up to 5.0 intermediate stops.
- Stay-put car is at the lobby 25 % (0.5/min) to 30 % (4/min) of calls.

## Stairs (α = 0.7, 1/min)
- Best 49 → **38 s**; **10 %** of trips walked (7 % descend-only, 10 %
  symmetric 12/12); h* = 2 throughout; middle-floor regret 22 → 51 %.
- Closed form, α = 0, 15 floors, lobby parking: h* 5 → **7** with stairs; the
  same 7 with symmetric and descend-only stairs.

## Fleet (α = 0.7, 1.5 passengers/min per car)
- 1 / 2 / 4 cars: best **65 / 52 / 43 s**; middle-floor regret **16 / 19 / 30 %**;
  band 2–4 / 2–3 / 2–4.

## Map (15 floors, one car, lobby parking, stairs on)
- **16 of 24** (α, rate) cells have h* = 2 — every cell with α ≥ 0.5, each with
  a one-floor band.
- α = 0: h* 11 → 13 as load rises; α = 0.25: 5 → 11, band 2–12 at 4/min.
- Middle-floor regret: 2 % (α=0, no traffic) … 27 % (α=0.5) … 96 % (α=0.9,
  1/min; top floor 223 %).

## Archetypes (stairs on)
| | floors | α | rate | cars | h* | p90 h* | band | mid regret | top regret |
|---|---|---|---|---|---|---|---|---|---|
| residential | 20 | 0.9 | 1 | 1 | 2 | 2 | 2–2 | 126 % | 288 % |
| office | 20 | 0.5 | 3 | 2 | 2 | 2 | 2–2 | 35 % | 102 % |
| hospital | 12 | 0.2 | 3 | 2 | 2 | 4 | 2–4 | 8 % | 42 % |

## Interpretation
Count how often you leave the building against how often you move within it.
If leaving dominates, the lowest floor wins by a wide margin, the parking
policy is the next most important factor, and stairs make the low floor
better still. If moving within dominates and the cars are busy, anywhere from
the middle up is within a few percent. See `paper/main.pdf`.
