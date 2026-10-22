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
| 2026-10-05 | hardware | ReSpeaker on MacBook: `--list-devices`, 3 s capture, pyusb firmware read | device 2, 6 ch @ 16 kHz, 0 overflows, ch1–4 live, ch5 silent; firmware DOA/VAD/RT60 over USB work on macOS; RT60 0.9 s |
| 2026-10-05 | hardware | `make live --log hw_live.csv` (26 s, user talking) | **9/1645 speech frames, 0 ranged**: no distance ever shown. User: "direction good, distance not" |
| 2026-10-05 | hardware | 5 s silence spectrum | 85 % of raw-mic noise power < 100 Hz (peaks 20, 80 Hz); broadband RMS gate dominated by hum; speech barely moves it |
| 2026-10-05 | hardware | VAD high-pass 150 Hz; clip 0°/0.5 m | speech frames 12 → 148; sim-prior range 5 m at 0.5 m (CDR −3 dB vs +7…+10 in sim) |
| 2026-10-05 | hardware | user asks for no room calibration ("should be dynamic") | added `relrange.py`: near/mid/far vs the session's own utterance history, hardware prior −44 dB @ 1 m |
| 2026-10-05 | sim | relative bands on the moving-talker demo, band floor 4 / 3 / 2.5 dB | 1 → 3 m is only ~7.5 dB; 3 dB floor kept (2.5 flickers) |
| 2026-10-05 | hardware | `make live` → PortAudio −9986 "Unspecified Audio Hardware Error" | CoreAudio wedged after an interrupted recording; replug the ReSpeaker (or `sudo killall coreaudiod`) |
| 2026-10-05 | hardware | live relative run (2 min) | user: near/far distinguished, vocal effort moves the band; CDR tracked the near phase that speech level missed |
| 2026-10-05 | hardware | VAD ablation on the clip, 150 vs 200 Hz | 150 Hz: 13/68 silence frames "speech" (hum passes the pitch test); 200 Hz: 0/68, same speech coverage → default 200 Hz; cue made configurable |
| 2026-10-05 | hardware | 18-min live relative session (`hw_rel.csv`, 200 Hz) | 22 % speech frames; 238 utterances; speech-level vs CDR bands agree 59 %, Spearman 0.28 (no labels) |
| 2026-10-06 | local_small | `make run` with final config | DOA 0 dB RT60 0.6: MAE 3.0 → 15.0 (diffuse), 2.6 → 14.1 (fan), one clip each; medians 2.7 / 4.3°. Demo 0.7°, range 0.13 m, bands 4/7 |
| 2026-10-06 | local_small | `run_vad_highpass_sim` (new) | 100 Hz best in sim; 150/200 Hz cost one reverberant clip at RT60 0.6 → documented trade-off |
| 2026-10-06 | hardware | `make hw` (new `analyze_hw_session.py`) | `results/hw_session.json`, `paper/figures/hw_session.pdf` |

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

- On hardware the bottleneck was again the silence model's input: the VAD.
  The raw mics hear the room's hum (not present in simulation), the broadband
  RMS gate saw only the hum, and the hum's harmonics passed the pitch test.
  Everything downstream (SRP, CDR, MVDR) assumes the VAD is right.
- "Distance" on one small array with no room model is at best *relative*;
  the user's requirement (no calibration, works in a new room) turned the
  output into near/mid/far learned online.

## Hardware session checklist
Done on 2026-10-05: steps 1, 3 (live radar, firmware DOA visible) and a
first live distance test. Still open (needs a marked floor and ~30 min):
1. ~~6-ch firmware / `--list-devices`~~ — device 2, 6 ch.
2. Mark the robot front on the floor; tape at 0.5/1/2/3 m on 8 rays.
3. ~~`make live`~~ — done; the blip follows the talker.
4. `make record ANGLE=<a> DIST=<d>` over the grid in `01_dataset.md`
   (works on macOS; the talker cannot see the printed countdown, so give an
   audible cue at the talk mark, e.g. `afplay /System/Library/Sounds/Ping.aiff`).
5. A few clips with a fan/TV and a second talker.
6. `make calibrate` → `azimuth_offset_deg` / `mirror` into the config;
   LOO DOA error ours vs firmware.
7. Distance cues on labelled clips: 1 m ×2, 0.4 m, 3 m, 3 m loud, 0.4 m soft,
   1 m loud, concatenated as one session → band accuracy for `cue: speech_db`
   vs `cue: cdr_db`, and the effect of vocal effort.
8. `make beamform WAV=<two-talker clip> ARGS=--deepfilter` and listen.
9. `make hw` and append here / to `04_results.md`.

## Session 2 (2026-10-18, Windows 11, with project 19)
- Windows "audio enhancements" (VocaEffectPack) turn the 6-ch stream into
  2 identical channels at 48 kHz. Disable them in `mmsys.cpl` → Advanced and
  replug.
- Checklist steps done:
  - 2: angle marks only; 0° = USB port, ±20°.
  - 4: 8 angles at 0.5 m.
  - 5: covered by project 19's interferer takes.
  - 6: `make calibrate`: offset 35.2°, mirror, LOO 15.8°.
  - 7: 7 distance clips, 2 m maximum.
  - 8: superseded by project 19.
- Not done: the firmware DOA (no libusb driver), and the 3 m clips (no room).
