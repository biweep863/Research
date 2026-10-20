# 00 · Overview

## Why
The home2 robot reads ch0 of its ReSpeaker (the chip's processed beam),
cleans it with DeepFilterNet3 (`atten_lim_db = 12`) and sends it to
Faster-Whisper. The four raw mics are discarded. The user wanted to know:
1. whether the array can cancel noise from one direction, starting with the
   intuitive idea of switching off the mic that faces the noise;
2. how DFN3's knob changes the output.

Localization on this array is project 15. This project uses its estimator
and gives it hardware numbers back.

## Hypotheses (set before the session)
- **H1.** Muting or selecting mics gains ≤ 1 dB. MVDR/GSC gains ≥ 5 dB when
  the noise is ≥ 45° from the talker.
- **H2.** DFN3 alone handles stationary noise but not speech-like
  interferers. The spatial front end helps most on babble and on a second
  talker.
- **H3.** WER has an optimum `atten_lim_db`, which moves lower with a
  beamformer in front.
- **H4.** home2's independent 64 ms DFN chunks are worse than continuous
  processing.
- **Falsification.** If MVDR→DFN does not beat DFN alone by ≥ 2 dB SI-SDR
  or ≥ 10 % relative WER at noise angles ≥ 90°, the array front end is not
  worth adding to home2.

## Status (2026-10-19)
- **Session:** recorded, 29 takes (doc/01).
- **Stages 1–3** (spatial, DFN + ASR, live) and figures are done. The paper
  compiles to 7 pages.

**Outcome by hypothesis:**

| Hypothesis | Outcome |
|---|---|
| H1 | Holds for muting. MVDR reaches 4.3 dB; only the oracle reaches 9 dB. |
| H2 | Half: the front end helps most on speech-like interferers, but DFN3 does not help Whisper on stationary noise either. |
| H3 | Holds, but the optimum ("off"–12) does not move. |
| H4 | Holds. |
| Falsification | Not met, so the front end is worth adding: −26 WER points, 38 % relative, at ≥ 90°. |

- **home2 recommendations:** `doc/04_results.md`.
- **Project 15** got its labelled hardware DOA, calibration and distance
  clips (its `doc/03_experiments_log.md`).
