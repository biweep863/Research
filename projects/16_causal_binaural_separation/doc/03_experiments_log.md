# 03 · Experiments log

| Date | Config | Command | Outcome |
|---|---|---|---|
| 2026-10-04 | — | `pytest tests` | 47 pass: WOLA perfect reconstruction (4 window pairs), streaming == offline for 6 model configs, no future leak, non-causal twin leaks, metric analytic checks, BRIR T60, mixer SNR/level |
| 2026-10-04 | local_small | `prepare_data` | 69 s. Real data throughout (`synthetic: false`). 4.0 h train speech (30 speakers), 300 test + 150 held-out × 3 s, KEMAR train/test and CIPIC-003 held-out, RT60 0.24–0.79 s |
| 2026-10-04 | local_small | `run_oracle_sweep` (1st attempt) | Killed: scoring took ~1.2 s per utterance (32× padded-IFFT ITD), ~30 min projected |
| 2026-10-04 | local_small | `run_oracle_sweep` | ~22 min. 6 latencies × {sym, asym} × {oracle ear, oracle common, Wiener} |
| 2026-10-04 | local_small | `run_oracle_sweep --floor-only` | ITD estimator floor: 3.8 µs at 20 dB SNR, 1.1 µs at 30 dB |
| 2026-10-04 | local_small `--sanity` | `train_sweep` (1st) | Conv-TasNet started at −30…−43 dB SI-SDRi (random basis) |
| 2026-10-04 | local_small `--sanity` | `train_sweep` (2nd, ± basis init) | TasNet still diverged in the first steps (SNR −29 dB after 5 steps) |
| 2026-10-04 | local_small `--sanity` | `train_sweep` (3rd, warm-up + 0.1× basis lr) | All 7 runs stable; pipeline end-to-end OK |
| 2026-10-04 | local_small | `train_sweep` | 19 runs × 2000 steps, ~1–5 min each on an RTX 3060. 6 of 8 sigmoid-headed runs stalled (saturated bias init) |
| 2026-10-04 | local_small | `train_sweep` (rerun) | Bias +1; 8 sigmoid-headed runs retrained, plus 3 `long_check` cells at 12k steps. TasNet at 4 and 32 ms still ~0.3 dB (ill-conditioned square basis) |
| 2026-10-04 | local_small | `train_sweep` (TasNet rerun) | Orthonormal basis; all 7 TasNet runs train (val 1.8–2.7 dB) |
| 2026-10-04 | local_small | `evaluate`, `bench_latency` | see 04_results.md |

## Observations
- **The window costs little when the analysis window is asymmetric.** The
  oracle IRM (per ear) gains 6.88 dB at 1 ms and 7.88 dB at 32 ms with
  asymmetric windows. With symmetric windows it gains 6.15 dB at 1 ms. The
  long analysis window recovers about 0.7 dB of the 1.7 dB that a short
  symmetric window loses, at the oracle level.
- **The oracle doesn't reach the cue JNDs either.** Per-ear oracle |ΔITD|
  is 17–27 µs (common gain 19–32 µs), against 3.8 µs for the estimator
  floor. Leftover late reverberation and noise move the broadband ITD even
  under an ideal mask. |ΔILD| stays at 0.06–0.11 dB, far below the JND.
- **The causal Wiener baseline barely helps**: +0.6…1.5 dB on test and
  −0.5…+0.5 on held-out, ESTOI ≈ unprocessed. It can't follow DEMAND's
  nonstationary noise or late reverberation.
- **Held-out unprocessed ESTOI (0.697) is higher than test (0.577)**: the
  held-out noise environments (park, meeting, square) are easier maskers.
  Held-out numbers are compared through improvements, never raw levels.
- **Local training is far from converged**: about 2.6–2.8 dB SI-SDRi after
  2000 steps, against about 7 dB for the oracle. Hence the extra long-budget
  check (`long_check`, 6000 steps on three AsymDF cells).
