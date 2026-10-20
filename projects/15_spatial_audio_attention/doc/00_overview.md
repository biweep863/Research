# 00 — Overview

## Motivation
The robot's HRI pipeline (home2 `hri/packages/speech`) hears through a
ReSpeaker USB Mic Array v2.0: four omnidirectional mics on a 32 mm-radius
circle. The pipeline is spatially blind:

- `audio_capturer.py` opens the 6-channel firmware stream and keeps **ch0
  only** (the on-chip processed channel); the four raw mics are discarded.
- `respeaker.py` publishes the firmware `DOAANGLE` smoothed by a *linear*
  moving average (so 359° and 1° average to 180°) and nothing about distance.
- `noise_cancellation.py` runs DeepFilterNet3 on that mono channel. It removes
  stationary/diffuse noise well, but cannot tell the person the robot is
  talking to from another person talking behind it.
- `voice_detection.py` already knows *when* the voice is present (RMS gate +
  autocorrelation pitch in 85–255 Hz + MFCC speaker lock), i.e. it already
  splits the stream into **voice** and **silence**.

The idea of this project ("audio attention"): use the voice/silence split to
learn what the room sounds like when the talker is quiet, then use the 4 raw
mics to find the talker (angle and distance), show it as a radar, and steer a
spatial filter at them so that DeepFilterNet receives the attended voice with
competing directions already suppressed.

## Hypotheses
- **H1 (angle).** With a 6.4 cm aperture, the angle is recoverable to a few
  degrees at moderate SNR with SRP-PHAT; the silence noise model (subtracting
  the noise cross-spectra, weighting bins by speech presence) is what keeps it
  usable at low SNR and with directional noise.
- **H2 (distance).** Timing differences give no range (far field everywhere);
  distance must come from *how* the voice arrives. The coherent-to-diffuse
  ratio (direct vs. reverberant energy) should fall monotonically with
  distance, but its absolute value depends on the room (its critical
  distance, set by volume and RT60).
- **H3 (selectivity).** An MVDR beamformer steered at the estimated talker,
  with its noise covariance learned on target-silent frames, suppresses a
  competing talker by several dB, while delay-and-sum on such a small array
  does almost nothing.

## Falsification
- H1 is falsified if SRP-PHAT with the silence model is not clearly better
  than plain SRP-PHAT / GCC-PHAT at 0 dB SNR, or if it is worse than the
  firmware DOA on hardware.
- H2 is falsified if no feature has |Spearman| > 0.5 with log-distance within
  a room, or if the cross-room error is not below the predict-the-mean
  baseline even with the room covariate.
- H3 is falsified if MVDR SINR gain is ≤ 3 dB on average.

## Status
- **Simulation:** done; all three hypotheses hold (`04_results.md`).
- **Hardware, first session (2026-10-05):** the ReSpeaker works end to end on
  macOS (raw mics + firmware over USB). The live radar follows the talker's
  direction. The VAD failed on the real room's hum (9 speech frames in 26 s) and
  was fixed with a 200 Hz high-pass on its input. The simulation-trained
  distance model does not transfer (0.5 m read as 2.5–5 m), so the radar now
  shows near/mid/far learned online from the session itself (no room
  calibration). It separates near from far but is sensitive to vocal effort.
- **Hardware, second session (2026-10-18, Windows, shared with project 19):**
  - Labelled grid: orientation offset 35°, mirrored; DOA leave-one-out error
    16°.
  - Distance clips: no cue tracks distance (|Spearman| ≤ 0.30).
  - MVDR on real interferers: +2.9 dB (DOA) / +4.3 dB (RTF); see project 19.
- **Still open:** firmware DOA comparison (needs libusb on Windows or the
  robot), and a DOA estimator that survives a loudspeaker interferer at
  0 dB.
