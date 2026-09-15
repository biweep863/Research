# 02 · Methodology

```
configs/*.yaml ─► src/sim (gait + events + sensor) ──► windows (N,T,6)
                      │                                  │
       data/pamap2_ankle.npz (real background)           ▼
                                        src/detect: features → threshold | boosting
                                                    onset (threshold, matched filter)
                                                    separate (2-template NNLS)
                                        src/decide: calibrate → fuse → decide
                                        src/localize: INS + ZUPT + UWB Kalman
                                                          │
                            src/experiments/run_* ──► results/*.json
                            src/figures/plot_contact ─► paper/figures/*.pdf + paper/macros.tex
                            latexmk ──► paper/main.pdf
```

## Features (`src/detect/features.py`, 16 per window)
peak |a|; peak jerk; widths at 50 % and 10 % of the baseline-removed peak;
high-frequency (>80 Hz) energy ratio and spectral centroid in a ±60 ms
Hann window; zero crossings of the 30 ms post-peak tail (ringing); direction
cosines of the peak; gyro magnitude peak and |gz| peak; kurtosis; mean excess
40–120 ms after the peak (push); background RMS; number of secondary peaks
above half maximum.

## Threshold rule (`threshold.py`)
contact ⇔ peak > τ; ball ⇔ width₅₀ < w else player; both ⇔ >1.5 comparable
peaks. (τ, w) grid-searched (37 × 29) on training macro-F1.

## Boosting (`ml.py`)
`HistGradientBoostingClassifier(max_iter=200, lr=0.08, max_leaf_nodes=15)`.
`BinaryCalibrator` = isotonic regression, clipped to [1e-4, 1−1e-4].

## Onset (`onset.py`)
Anchor = argmax of the *horizontal* excess |a_xy| − median (foot strikes are
vertical). Threshold: walk back from the anchor while excess > max(6σ_MAD,
0.08·peak); linear interpolation of the crossing. Matched filter: normalised
correlation of the clean template's rising edge over lags in [peak−3L,
peak+L]; parabolic refinement of the best lag.
Theory used for ordering: P(correct) = Φ(Δ / (σ√2)) for two independent
onsets with error σ each.

## Separator (`separate.py`)
y = |a| − baseline. Fit A·T_ball(t−k₁) + B·T_player(t−k₂), A,B ≥ 0, on a lag grid
(1 ms step, [peak−60, peak+30] ms). Two-column NNLS in closed form: the
unconstrained 2×2 solution if positive, else the better clamped single fit;
vectorised over the (k₁,k₂) grid with Gram matrices. Model selection: two
pulses accepted iff residual/σ² + 2·2·ln n < best single residual/σ².
Templates = noise-free mean |a| of 40 injected pulses at the working rate,
aligned to onset (`clean_templates`).

## Decision (`decide/bayes.py`)
Sources: boot = P(ball) + ½P(both); ball = 1 − Φ((t̂_ball − t̂_boot,first)/(σ√2))
(0 if the ball saw nothing); sep = 0.9 / 0.1 for single models, else
Φ(|Δ̂|/(σ√2)) oriented by the order. Each isotonically calibrated on the
calibration half. Naive-Bayes: logit π_s + Σ_k [logit p_k − logit p̄]. Stacked:
logistic regression on [logit p_boot, logit p_ball, logit p_sep, logit π_s].
Decision: no-foul iff P > c_miss/(c_false + c_miss); review iff |P − thr| < band.
Metrics: accuracy, Brier, ECE (10 bins), log-loss; split by single/overlap.

## Localisation (`localize/ins.py`)
Truth: 6 m/s run, 2.8 strides/s, stance 35 % of the stride with v = 0, min-jerk
swing with 15 cm lift, 60° yaw cut mid-run. IMU: body-frame specific force and
angular rate with white noise and a constant gyro bias. Estimator: gyro
orientation integration (re-orthonormalised) → world acceleration → 6-state
[p, v] Kalman filter; ZUPT (v=0, R=1 cm/s) when |ω| < 40°/s and ||f| − g| < 3
m/s²; UWB position fix (σ = 0.10 m) every 1/rate s. Error = RMS over the second
half of the run. P(wrong side) = Φ(−d/σ_h).

## Verification (`tests/test_model.py`, 12 tests, ~5 s)
Half-sine and low-pass shapes; clipping and 1 g at rest; gait has strikes; ball
harder/shorter than player; gap sign and ball-sensor behaviour; feature vector
finite; detectors beat chance (thr > 0.4, GB > 0.7); ball onset < 4 ms RMSE at
1 kHz for both estimators; ordering theory limits; separator order ≥ 9/12 at
±25 ms; fusion lowers Brier, decision thresholds, timing-score limits; INS with
ZUPT and UWB beats free integration and stays < 0.5 m.
