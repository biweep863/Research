# 03 · Experiments log

| Date | Step | Outcome |
|---|---|---|
| 2026-10-18 | hardware smoke test | MME/WASAPI gave ch0 = ch1, ch2–5 silent. Windows audio enhancements (VocaEffectPack) downmix to 2 ch at 48 kHz. Fixed in `mmsys.cpl` (Settings alone was not enough) + replug |
| 2026-10-18 | DFN install | `deepfilterlib` has no cp312 Windows wheel (needs Rust). Made `.venv-dfn` (py3.11, torch 2.4.1 CPU). `libtorchaudio.pyd` failed to load: `vcomp140.dll` missing system-wide, copied from sklearn |
| 2026-10-18 | DFN check | white noise: atten 12 → −12.0 dB, 24 → −23.8, None → −39.1, **0 → −39.1** (0 = no limit) |
| 2026-10-18 | calibration | pink −35 dBFS, then −22 dBFS: only +4 dB over the room at the mics. Windows volume was 8, then 24, then 60. At 60: pink −22 dBFS → −43.5 dBFS (>200 Hz) vs floor −59. Session level −16 dBFS with a peak cap |
| 2026-10-18 | noise takes | 10 takes, no clipping. R is 5 dB quieter than L. SRP-PHAT: L 8–16°, R 254–261° in the array frame, consistent across interferers (talker's estimate: 45° apart; measured: about 110°) |
| 2026-10-18 | speech takes | 8 angles at 0.5 m. First take relabelled 315° → 270° by the talker. Whisper-small already near-perfect on raw ch1. **ch0 hallucinated a repeated sentence** |
| 2026-10-18 | first DOA check | all takes pointed at 10–20°: the loudest frames were the chirps and beep from speaker L. Fixed by aligning and using only the reading window |
| 2026-10-18 | live + distance takes | 2 live takes, 7 distance clips (2 m was the room's limit) |
| 2026-10-18 | `run_spatial` v1 | MVDR speech-component SI-SDR ~4 dB: mains hum in the speech takes dominated. Added a 100 Hz high-pass to every take |
| 2026-10-18 | `run_spatial` | 320 scenes × 9 front ends, about 4 min. Gains: mute 0.4, RTF-MVDR 4.3, oracle 9.1 dB |
| 2026-10-18 | `run_live` | ch0 + DFN transcribes the interfering English voice (WER 200 %). Superposition residual −6.4 dB, mostly < 300 Hz |
| 2026-10-18 | `export_p15` + p15 `calibrate_distance` | offset 35°, mirror. DOA LOO 15.8°. Distance: |Spearman| ≤ 0.30, MAE 0.40 m vs 0.41 m for the mean |
| 2026-10-18/19 | `run_dfn_asr` | 80 scenes × 8 front ends × (off + 5 limits) + home2 mode = 4,016 transcriptions, 1 h 53 min (DFN CPU about 7 min per 16 scenes, the rest Whisper + ESTOI) |

## Observations
- **The chirp protocol works.** Offsets are stable within 2 ms across 29
  takes, and the clock-rate ratio is within 1e-4. The live take's played
  interferer matched its noise-only take at 0.999 correlation.
- **The user's talker marks were rough.** Labels are therefore measured from
  each clean take. Two positions (135° and 180°) came out at the same array
  angle (270°).
- **Superposition is approximate.** Before the talker speaks, a live
  recording matches its noise-only take with a −6.4 dB residual:
  - 100–300 Hz: −2.5 dB (room noise);
  - 300–1000 Hz: −9 dB;
  - above 1 kHz: −5 dB.

  The live MVDR gain (2.9 dB) is about 1.4 dB below the mixed-scene median.
- **Bug.** `noises.cafeteria` drew its excerpt offset from the requested
  length, so the cafeteria live take replayed a different excerpt. Fixed
  after the session (the offset is now independent of length). The recorded
  `_play.wav` files are the ground truth.
