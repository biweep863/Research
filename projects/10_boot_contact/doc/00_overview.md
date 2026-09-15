# 00 · Overview

**Question.** The connected match ball detects *when* it is struck from an
inertial sensor at its centre. A foul turns on the other half of the event:
*what* the boot hit and *in which order*. Can a boot-mounted IMU — the same
idea as the ball's "did the acceleration vector change suddenly?" — supply
that half, precisely enough to say who touched first, and can the answer be
given as an honest probability rather than a verdict?

**Hypothesis.** A ball impact (short, hard, forward, rings) and a player
contact (longer, softer, oblique, pushes) are separable from the boot signal;
the ball onset can be timed to about a sample; combining with the ball's own
sensor orders the two contacts once they are separated by more than the timing
error; and inside one pulse width (~10 ms) no estimator can order them — a
physical floor that the decision layer must report as P ≈ 0.5.

**Falsification.** If the classifier were no better than the threshold rule,
if the ordering curve did not follow Φ(Δ/(σ√2)) with the measured σ, or if the
separator did no better than chance above one pulse duration, the hypothesis
would be wrong. If the fused probability were poorly calibrated (ECE ≫ 0.05)
the "honest probability" claim would be wrong.

**Five stages.**
1. Detection — {none, ball, player, both} vs sampling rate, sensor grade, and
   synthetic vs real (PAMAP2) background; threshold rule vs gradient boosting.
2. Timing and ordering — onset RMSE vs rate; P(correct order) vs separation
   with the boot + ball sensor.
3. The anomalous overlap — two-template NNLS separation vs signed gap.
4. The probabilistic call — knowledge-base priors + boot + ball + separator,
   naive-Bayes vs stacked fusion; calibration; cost-weighted decision with a
   review band.
5. Where — foot INS with ZUPT + sparse UWB fixes; P(wrong side of a line).

**Headline.** Boosting separates the four classes at macro-F1 ≈ 0.97 (0.91 on
real background); the ball onset is timed to 0.5 ms at 1 kHz; ordering is ≥ 90 %
correct at 20 ms separation and at chance below 3 ms; the separator recovers
the order in 88 % of overlaps at 10 ms; stacked fusion gives P(ball first) with
accuracy 0.944 and Brier 0.038 against a base-rate prior at 0.667/0.218; ZUPT +
10 Hz UWB places the boot to 0.12 m.

**Reproduce.**
```bash
cd projects/10_boot_contact
make test      # ~5 s: 12 tests
make all       # ~2 min: 5 experiments -> results/*.json, 6 figures, 109 macros, paper/main.pdf
```

See `01_dataset.md` (simulator + PAMAP2), `02_methodology.md`,
`03_experiments_log.md`, `04_results.md`, `05_decisions.md`.
