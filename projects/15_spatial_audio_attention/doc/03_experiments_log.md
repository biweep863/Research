# 03 — Experiments log

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-10-04 | local_small | smoke test (6 clips) | pipeline runs, 0.3 s/clip; `level_db` +18 dBFS → STFT not normalised; fixed |
| 2026-10-04 | local_small | smoke test | pyroomacoustics propagates with 1/r not 1/4πr: source levels 22 dB too hot; fixed |
| 2026-10-04 | local_small | `run_beamform_sim` | SI-SDR vs mic 1 penalises DS/MVDR for linear filtering of the target; switched to output SINR gain |
| 2026-10-04 | local_small | `run_doa_sim` (v1) | F0 harmonic weighting identical to baseline → not applied in pooled path (bug); fixed |
| 2026-10-04 | local_small | `pytest` | static talker tracked 34° off: PHAT gives noise-only bins (between harmonics) full weight; added speech-presence bin weights |
| 2026-10-04 | local_small | `run_radar_demo` (v1) | 3 m talker: 6/119 speech frames — home2's 600 int16 gate is for AGC'd ch0; raw mics need the adaptive gate. Track blended old/new talker → reset spatial stats after 0.6 s silence |
| 2026-10-04 | local_small | debug CDR | CDR pinned at 23.2 dB for 0°/90° talkers: array at room centre → mirror-symmetric pair, Γx ≡ 1. Moved array off-centre, median over pairs |
| 2026-10-04 | local_small | `make run` | DOA, distance, beamforming, demo all written; 2.6 min wall clock |
| 2026-10-04 | local_small | `run_distance_sim` + RT60 covariate | cross-room MAE 0.55 → 0.33 m with RT60 |
| 2026-10-04 | local_small | `calibrate_distance` on 16 simulated "recordings" with a planted 30° rotation + mirror | recovered offset 31.5°, mirror=true; LOO DOA 1.5°; tooling verified (outputs deleted, not real HW) |

## Observations
- Most of the work in this pilot was not the estimators but the *silence
  model*: three of the bugs above were cases where "noise" was not what the
  code assumed (noise-only bins, speech below an absolute gate, stale
  statistics from a previous position).
- Delay-and-sum is useless on this array (≈0 dB): at 1 kHz the 6.4 cm
  aperture is 0.19 λ. Any spatial selectivity must be superdirective (MVDR),
  which is sensitive to steering error — hence the importance of DOA.
- The CDR feature is near-perfectly monotone within a room (Spearman −0.90
  to −0.95) but shifts by several dB between RT60s. Distance is a
  *room-calibrated* quantity on this array.

## Hardware session checklist (next)
1. Flash/confirm the 6-channel firmware; `make live ARGS=--list-devices`
   must show the ReSpeaker with 6 input channels.
2. Mark the robot front on the floor; tape marks at 0.5/1/2/3 m on 8 rays.
3. `make live` — check the blip follows you; compare against the dashed
   firmware DOA ray (on Windows the USB control interface needs a libusb
   driver; on the robot it works like `respeaker.py`).
4. `make record ANGLE=<a> DIST=<d>` over the grid in `01_dataset.md`; stay
   silent for the first 2 s; note who speaks with `ARGS="--speaker X"`.
5. A few clips with a fan/TV on and with a second person talking
   (`ARGS="--note 'second talker at 120'"`).
6. `make calibrate` → paste `azimuth_offset_deg` / `mirror` into the config;
   it saves `results/distance_model_hw.json` (radar prefers it).
7. `make live` again: is the range right now? Log a run with `ARGS="--log hw_run.csv"`.
8. `make beamform WAV=<two-talker clip> ARGS=--deepfilter` and listen:
   ch0 vs mic1 vs mvdr vs mvdr_df.
9. Append the rows here and the numbers to `04_results.md`.
