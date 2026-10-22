# 04 · Results (local_small: 1 session, 320 scenes, Whisper-small)

## Stage 1: exact SNR gain over raw ch1 (median over 320 scenes)
| Front end | SNR gain | Gain vs separation (0–45 / 45–90 / 90–135 / 135–180°) |
|---|---|---|
| mute noisiest mic | +0.4 dB | 0.5 / 0.8 / −0.1 / 0.5 |
| soft mic weighting | +0.5 | |
| mean of 4 | +0.4 | |
| delay-and-sum | +0.4 | |
| MVDR, far-field DOA | +2.9 | 1.6 / 2.6 / 5.2 / 3.0 |
| GSC (NLMS) | +3.7 | 3.3 / 4.3 / 4.0 / 3.3 |
| **MVDR, RTF (GEVD)** | **+4.3** | 3.7 / 4.7 / 4.9 / 4.0 |
| oracle MVDR | +9.1 | 8.5 / 10.9 / 9.0 / 9.2 |

DOA from the mixture (project 15's estimator) against the clean-take label:
median error 65/58/11/5° at −5/0/5/10 dB. This is why DOA steering trails
RTF steering.

## Stage 2: Whisper WER (%) at 0 dB input SNR, 80 scenes
| Front end | no DFN | 6 | 12 (home2) | 18 | 24 | no limit | home2 chunks @12 |
|---|---|---|---|---|---|---|---|
| raw ch1 | 74 | 74 | 77 | 81 | 84 | 90 | 83 |
| mute | 73 | 74 | 80 | 81 | 84 | 91 | – |
| MVDR DOA | 54 | 53 | 59 | 61 | 65 | 74 | – |
| GSC | 54 | 51 | 54 | 56 | 62 | 72 | – |
| **MVDR RTF** | **50** | 50 | 50 | 52 | 57 | 68 | 62 |
| oracle | 20 | 19 | 22 | 23 | 25 | 30 | – |

Clean speech-only take: 15 %. Akane close-talk: 15 %.

Paired differences (Wilcoxon over scenes):

| Comparison | Δ WER | p |
|---|---|---|
| RTF-MVDR vs raw | −24.4 pts | 5e-9 |
| mute vs raw | −0.8 | 0.47 |
| GSC vs RTF-MVDR | +4.5 | 0.04 |
| DFN 12 vs off (raw / RTF) | +2.7 / +0.8 | 0.46 / 0.48 |
| DFN no limit vs off (raw / RTF) | +16.1 / +18.5 | 9e-9 / 5e-12 |
| home2 chunks vs continuous @12 (raw / RTF) | +6.3 / +11.4 | 3e-5 / 1e-9 |
| RTF + DFN12 vs raw + DFN12 | −26.3 (also −26.3 at ≥ 90°, 38 % rel.) | 1e-8 |

ESTOI against the Akane rises monotonically with the limit for every front
end. For the raw mic it goes 0.27 (off), 0.30, 0.32, 0.33, 0.34, 0.35 (no
limit). The perceptual and recognition metrics disagree.

By interferer (raw → RTF-MVDR, no DFN):
| Interferer | raw | RTF-MVDR |
|---|---|---|
| second talker | 97 | 60 |
| cafeteria | 84 | 52 |
| fan | 83 | 60 |
| pink | 78 | 53 |
| music | 29 | 22 |

Pink on the raw mic goes 78 → 94 with DFN at 12.

## Stage 3: live takes (WER %, front end only / + DFN 12 continuous)
| Front end | 2nd talker from L | cafeteria from L |
|---|---|---|
| **chip ch0 (home2 today)** | 200 / 200 | 39 / 48 |
| raw ch1 | 148 / 44 | 35 / 48 |
| mute | 167 / 207 | 42 / 48 |
| MVDR DOA | 11 / 7 | 16 / 16 |
| GSC | 11 / 15 | 16 / 16 |
| MVDR RTF | 11 / 11 | 16 / 16 |

Exact live SNR gains (talker take): RTF 2.9, DOA 3.3, GSC 2.4, mute −0.5 dB.

## Recommendations for home2
1. Publish ch1–4 (the plan in project 15's `doc/06_home2_integration.md`)
   and run an RTF-MVDR whose noise covariance is updated whenever
   `voice_detection` reports silence.
2. Keep `DF_ATTEN_LIM_DB` ≤ 12. Never set it to 0 (that means no limit) and
   never leave it unlimited.
3. Run DFN on a continuous stream with persistent state (DFN's streaming
   API / `libDF`), not as independent 64 ms `enhance()` calls. Or skip DFN
   in front of Whisper altogether: it did not lower the WER at any limit.

---

# Full run (gpu_full: 320 scenes × 4 SNRs, Whisper large-v3-turbo, 22,096 transcripts)

Run on 2026-10-19 (about 5 h on the RTX 3060 + CPU). Outputs:
`results/full/`, `paper/figures_full/`, `paper/macros_full.tex`.
Clean take: 3 % WER; Akane: 3 %. Turbo is far stronger than `small` (15 %), so absolute
WERs are lower than in the table above; the direction of every result holds.

## Stage 1: exact SNR gain (median) at −5 / 0 / 5 / 10 dB
| Front end | −5 | 0 | 5 | 10 |
|---|---|---|---|---|
| mute / mean / delay-and-sum | 0.4 | 0.4 | 0.4 | 0.4 |
| MVDR DOA | 3.1 | 2.6 | 2.9 | 2.9 |
| GSC | 2.7 | 3.5 | 4.0 | 4.3 |
| **MVDR RTF** | 3.1 | **4.1** | **4.8** | **5.0** |
| oracle | 8.2 | 9.1 | 9.4 | 9.4 |

DOA error 65/58/11/5° (same as local).

## Stage 2: mean WER (%) by input SNR (DFN continuous; last column home2 chunks @12)
| SNR | Front end | off | 3 | 6 | 12 | 18 | 24 | 36 | no limit | home2 @12 |
|---|---|---|---|---|---|---|---|---|---|---|
| −5 | raw ch1 | 83 | 82 | 82 | 83 | 83 | 86 | 95 | 93 | 86 |
| −5 | MVDR RTF | 65 | 62 | 60 | 64 | 67 | 72 | 77 | 80 | 72 |
| −5 | oracle | 17 | 16 | 16 | 19 | 23 | 27 | 34 | 35 | 28 |
| 0 | raw ch1 | 55 | 53 | 54 | 60 | 62 | 66 | 74 | 75 | 63 |
| 0 | GSC | 29 | 28 | 30 | 32 | 34 | 40 | 45 | 50 | 40 |
| 0 | **MVDR RTF** | **27** | 26 | 27 | 27 | 31 | 36 | 41 | 44 | 40 |
| 0 | oracle | 5 | 5 | 5 | 6 | 7 | 8 | 9 | 11 | 9 |
| 5 | raw ch1 | 15 | 15 | 17 | 19 | 23 | 29 | 37 | 39 | 26 |
| 5 | MVDR RTF | 13 | 12 | 11 | 12 | 15 | 15 | 20 | 22 | 17 |
| 10 | raw ch1 | 5 | 5 | 6 | 7 | 9 | 12 | 14 | 15 | 8 |
| 10 | MVDR RTF | 4 | 4 | 4 | 5 | 6 | 8 | 10 | 13 | 6 |

Paired tests (Wilcoxon, all 320 scenes unless noted):
| Comparison | Δ WER | p |
|---|---|---|
| RTF-MVDR vs raw (off) | −12.5 pts (−28.0 at 0 dB) | 8e-18 |
| GSC vs RTF-MVDR | −0.3 | 0.09 |
| mute vs raw | −1.7 | 0.12 |
| DFN 12 vs off, raw | +2.6 | 4e-6 |
| DFN 12 vs off, RTF | −0.0 | 0.92 |
| DFN 6 vs off, RTF at −5 dB | −4.9 | 1e-4 (only significant DFN win) |
| DFN no limit vs off (raw / RTF) | +15.9 / +12.7 | 7e-36 / 1e-30 |
| home2 chunks vs continuous @12 (raw / RTF) | +3.5 / +6.6 | 7e-9 / 2e-19 |
| RTF + DFN12 vs raw + DFN12 | −15.1 | 2e-21 |

## Stage 3: live takes, re-scored with turbo (WER %, off / + DFN 12)
| Front end | 2nd talker from L | cafeteria from L |
|---|---|---|
| chip ch0 (home2 today) | 222 / 130 | 35 / 26 |
| raw ch1 | 115 / 122 | 35 / 29 |
| mute | 119 / 126 | 32 / 42 |
| MVDR DOA | 7 / 7 | 16 / 16 |
| GSC | 7 / 0 | 16 / 16 |
| MVDR RTF | 7 / 4 | 16 / 13 |

## What changed vs local_small
- Conclusions hold. The spatial front end is the only large win; DFN never helps
  much and hurts above 12 dB of limit; home2's chunk mode is significantly worse.
- With turbo, GSC ties RTF-MVDR (it was 4.5 pts worse with `small`).
- The WER optimum of `atten_lim_db` sits at **3–6 dB**, not 12. Only at −5 dB after
  the beamformer is the gain significant (−4.9 pts at 6).
- The 12 dB home2 setting still costs +2.6 pts on the raw mic (now significant).
