# 04 · Results

Headline numbers (local config; all emitted as macros in `paper/macros.tex`).

## Detection (300 windows/class, 70/30 split)
- Boosting macro-F1, impact-rated sensor: **0.947 / 0.969 / 0.969 / 0.936** at
  100 / 200 / 500 / 1000 Hz. Tuned threshold rule: **0.55–0.67**.
- Phone-grade ±16 g sensor clips **66 %** of ball impacts: F1 **0.879** at
  100 Hz, recovering to **0.961** at 500 Hz — rate matters more than range.
- Real PAMAP2 ankle background at 100 Hz: **0.908** (sport), **0.874**
  (consumer); threshold **0.630**. Synthetic gait is not making it easy.
- Only systematic confusion: *both → ball* (9 %) when the player pulse is faint.

## Timing and ordering
- Ball onset RMSE: **5.8 / 4.1 / 2.2 / 0.52 ms** at 100 / 200 / 500 / 1000 Hz
  (matched filter beats the sample grid at high rates).
- Player onset: robust spread **2.1 ms** at 1 kHz; RMSE 14.8 ms from ~3 %
  gross errors (faint contact vs foot strike).
- Boot first-onset + ball sensor (σ_ball = 1.2 ms): combined σ **1.3 ms** at
  1 kHz, **5.9 ms** at 100 Hz. P(correct order) = **0.88** at 10 ms and ≥ 0.9
  at **20 ms** (1 kHz); **0.64** at 10 ms (100 Hz); chance below 2–3 ms.

## The overlap (120 windows/gap, 1 kHz)
- Separator finds two pulses in ≥ **0.96** of windows at every gap.
- Order accuracy: **1.00 / 0.96 / 0.88 / 0.73 / 0.58** at ±20 / 15 / 10 / 6 /
  3 ms; 0.63 at 0 (chance). Gap RMSE **6 ms** at ≥ 20 ms, **15 ms** inside the
  overlap.
- Boosting labels 0.84–0.96 of these windows *both* at every gap — it detects
  two contacts but cannot order them.

## The probabilistic call (750 evaluation cases, base rate 0.59)
| evidence | acc | Brier | ECE | acc single | acc overlap |
|---|---|---|---|---|---|
| prior only | 0.667 | 0.218 | 0.020 | 0.656 | 0.677 |
| boot only | 0.809 | 0.123 | 0.029 | 1.000 | 0.616 |
| boot + prior | 0.825 | 0.112 | 0.035 | 0.995 | 0.653 |
| + ball timing | 0.951 | 0.049 | 0.046 | 0.971 | 0.930 |
| + separator (naive Bayes) | 0.927 | 0.055 | 0.054 | 0.915 | 0.938 |
| **all, stacked** | **0.944** | **0.038** | **0.031** | 0.942 | 0.946 |

- Stacked weights: boot 1.68, ball 0.41, separator 1.29, prior 0.83.
- Late challenge (prior 0.20): prior accuracy 0.78 → fused 0.96.
- Equal costs: 5.6 % error; with a ±0.15 review band, **5 %** referred and
  **3.4 %** error among decided. 4:1 false-foul cost → threshold 0.20, false
  fouls 4.1 % → 0.7 %, missed 1.5 % → 6.7 %.

## Localisation (20 runs, 6 s at 6 m/s, UWB σ = 0.10 m)
- Horizontal RMSE, sport IMU: free **3.3 m**, ZUPT **0.77 m**, + UWB 1 / 5 / 10 /
  20 Hz **0.27 / 0.15 / 0.12 / 0.09 m**. Consumer: free 7.6, ZUPT 1.98, 20 Hz
  0.12 m.
- Penalty-area call at 10 Hz: P(wrong side) **1.7 %** at 25 cm, **20 %** at
  10 cm, < 0.01 % at 50 cm; ZUPT alone: 26 % at 50 cm.

## Interpretation
The boot can say what it hit and, with the ball's clock, in what order — down
to about one ball-pulse duration (~10 ms), below which the physics gives no
answer and the calibrated posterior correctly returns ≈ 0.5. The decisive
evidence is the ball sensor's timing; the knowledge-base prior matters most
where it is most wrong. See `paper/main.pdf`.
