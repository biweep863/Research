# 04 — Results (simulation pilot, `local_small.yaml`)

Hardware results: **pending**. Formal write-up: `paper/main.pdf`.

## Angle of arrival (8 angles × 2 RT60, talker at 1.5 m; pooled per clip)
MAE in degrees at **0 dB SNR** (RT60 0.3 / 0.6 s):

| estimator                         | diffuse noise | fan + diffuse |
|-----------------------------------|---------------|---------------|
| SRP-PHAT + silence model (full)   | 3.2 / 3.0     | 4.2 / 2.6     |
| − noise subtraction               | 3.3 / 2.3     | 7.8 / 20.1    |
| − speech-presence bin weights     | 19.4 / 26.5   | 4.8 / 4.6     |
| plain SRP-PHAT                    | 4.5 / 8.3     | 68.6 / 75.7   |
| GCC-PHAT + least squares          | 21.7 / 28.6   | 25.7 / 34.3   |
| + F0 harmonic weights             | 3.2 / 2.8     | 4.0 / 2.4     |

At 20 dB SNR everything is within 0.9–2.1°. The two uses of the silence model
are complementary: **subtraction** removes a directional noise source,
**bin weighting** stops noise-only bins from voting under diffuse noise.
Together they turn a 69–76° failure into 3–4°. F0 harmonic weighting adds
nothing on the synthetic talker (already perfectly harmonic) — retest on
real speech.

Moving-talker demo (7 positions, 0.5–3 m, RT60 0.45 s, streaming tracker):
median azimuth error **1.0°**, 90th percentile 7.0°.

## Distance (108 clips, 0.5–3 m, 3 RT60s, ±4 dB vocal effort)
Spearman with log-distance (pooled over rooms / within each room):
CDR −0.77 / −0.90…−0.95; level −0.61; SNR −0.48; tilt +0.13.

| model (MAE, m)          | leave-angle-out | leave-room-out |
|-------------------------|-----------------|----------------|
| predict mean            | 0.75            | —              |
| inverse-square prior    | 0.90            | —              |
| CDR                     | 0.44            | —              |
| CDR + level             | 0.45            | 0.55           |
| all 4 features          | 0.59            | 0.51           |
| **CDR + RT60**          | **0.24**        | **0.33**       |

Loudness alone is a poor range cue once people speak at different levels.
The CDR is the cue, but it is room-relative; giving the model the room RT60
(the firmware estimates one) halves the cross-room error. The demo's
streaming range (same room as training) has a median error of 0.18 m and
under-estimates far talkers (3 m → ~2.1 m).

## Selective cancellation (target 0°, competing talker, SIR 0 dB, 1.2 m)
Output SINR gain vs mic 1, DOA estimated by onset subtraction (≤ 9.6° error):

| interferer | 45° | 90° | 135° | 180° |
|------------|-----|-----|------|------|
| MVDR, RT60 0.3 | 5.6 | 9.8 | 10.8 | 10.7 |
| MVDR, RT60 0.6 | 2.6 | 6.7 | 6.8 | 6.0 |
| delay-and-sum  | 0.0–0.5 for all cases |||| 

Mean MVDR gain **7.4 dB**; delay-and-sum 0.2 dB. Gain drops at 45°
separation (where the DOA error was also largest, 9.6°) and in the more
reverberant room (the interferer's reverberation is diffuse and cannot be
nulled by 4 mics).

## Verdict (simulation)
H1 holds, H2 holds only with the room covariate, H3 holds. The hardware
session decides whether these transfer.
