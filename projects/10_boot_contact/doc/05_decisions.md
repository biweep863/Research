# 05 · Decisions & Alternatives

## Simulate rather than collect
No public boot-IMU foul dataset exists and collecting one (instrumented boots,
staged tackles, ground truth from high-speed video) is a study in itself. A
physics-based simulator with published kicking parameters gives computable
ground truth for onsets, gaps and order — which is exactly what the ordering
question needs — and the real PAMAP2 background guards against a synthetic
gait that is too clean. The price is realism, stated in the limitations.

## Half-sine pulse templates
Rejected: measured kick waveforms (none public with boot-frame IMU), a
Hertz-contact model (more parameters, same qualitative shape). A half-sine with
a ringing tail for the ball and a push tail for the player captures the three
properties the detectors exploit — duration, hardness, direction — with ranges
from Nunome 2006 / Lees & Nolan 1998.

## Two sensor grades, not one
The obvious deployment question is "does a phone-grade ±16 g part suffice?".
Answer: it clips two thirds of ball impacts and loses at 100 Hz, but recovers
by 500 Hz because plateau width, ringing and direction survive clipping. That
finding needed both grades in the sweep.

## Threshold rule as the baseline, tuned
The connected ball's logic is a threshold on the change of acceleration. We
give it every chance (both thresholds grid-searched on macro-F1) so that the
boosting gain is not an artifact of a weak baseline.

## Gradient boosting on features, not a 1-D CNN
Considered a small PyTorch CNN on raw windows. Rejected for the local pilot:
the interpretable features already reach 0.97 F1, boosting trains in seconds,
and the feature importances tell a physical story. A CNN is the natural
`gpu_full` extension.

## Onset anchored on the horizontal component
The global |a| peak is often a foot strike when the player contact is faint.
Foot strikes are vertical, contacts are not; anchoring on |a_xy| is a one-line
physical prior that removed most gross errors. A learned anchor (use the
classifier's window) was rejected as circular.

## Matched filter instead of CUSUM as the second onset estimator
CUSUM never resets on a gait background; documented in the log. The matched
filter's parabolic refinement gives genuine sub-sample precision and is the
textbook estimator for a known pulse shape.

## Two-template NNLS with closed-form 2×2 solution
Alternatives: sparse deconvolution (L1), a learned two-onset regressor. NNLS
over a lag grid is exact for this model, has one interpretable penalty, and in
closed form is fast enough to run inside the decision loop. The BIC-like
penalty prevents fitting a second pulse to noise on single contacts.

## The knowledge base is a config table
Base rates by situation (standing tackle 0.70, sliding 0.55, 50/50 0.50, late
challenge 0.20 ball-first) are *illustrative*: they encode that the prior is
informative but not decisive, and that it can be badly wrong (late
challenge). A deployed system learns them from adjudicated matches. Keeping
them in YAML makes the paper's claim explicit and swappable.

## Naive Bayes and stacking, both reported
Naive-Bayes fusion is the transparent baseline, and its failure when adding
the (correlated) separator is a result worth showing. Stacked logistic
regression on calibrated log-odds is the minimal fix and is what the decision
rule uses. The best model is chosen by Brier on the evaluation half — same
data for selection and reporting, noted as a mild optimism.

## Cost-weighted threshold with a review band
The point of a probability is that policy (cost of a false foul vs a missed
foul, how much to refer to VAR) becomes a threshold and a band rather than a
retraining. We sweep both and report false/missed rates and review load.

## Localisation as a decision, scored over time
A 6-state linear KF with ZUPT and UWB is the minimal foot-INS; a full
error-state filter with orientation covariance was rejected as more machinery
than the question needs. Error is scored over the second half of the run
rather than at one instant because the phase of the UWB fixes relative to a
single instant produced a non-monotone (and misleading) rate curve. The
"wrong side of the line" probability turns the RMSE into the referee's
quantity.

## Cost is machine-independent
No wall-clock numbers are quoted as results; all metrics are F1, ms, m and
probabilities, deterministic given the seed.
