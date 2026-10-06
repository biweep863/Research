# Project 15 — Spatial audio attention on the ReSpeaker (localization radar → selective cancellation)

## Question
With the 4 raw microphones of the ReSpeaker USB v2.0 on our robot (home2
`hri/` pipeline), can we tell **where** the person talking is — angle *and*
distance — and use that to cancel sound coming from **other** directions
before DeepFilterNet3, using the VAD's voice/silence decision as the
reference for what "noise" is?

## Quick start
```bash
# from repo root (Windows: .venv/Scripts/python.exe is picked automatically)
cd projects/15_spatial_audio_attention
make test     # 37 unit tests, ~2 s
make run      # DOA / distance / beamforming / VAD high-pass sims + moving-talker demo (~4 min)
make figures  # paper/figures/*.pdf + paper/macros.tex
make demo     # replay the simulated moving talker on the live radar
make paper    # compile paper/main.pdf
```
Extra deps (installed in the repo venv): `pyroomacoustics sounddevice soundfile pyusb`.
Optional: `deepfilternet` for `make beamform ARGS=--deepfilter`.

## On the robot / a laptop (hardware session)
Tested on macOS (MacBook, USB) on 2026-10-05; Linux on the robot works like home2.
```bash
make live ARGS="--list-devices"            # find the ReSpeaker (needs 6-ch firmware)
make live ARGS="--device 2"                # radar; title shows SPEECH + NEAR/MID/FAR; firmware DOA dashed
make live ARGS="--no-plot --log run.csv"   # headless (ssh / docker) + per-frame CSV
make record ANGLE=90 DIST=1.5              # 2 s silence, then talk; labels in a .json sidecar
make calibrate                             # orientation offset + distance model from recordings
make beamform WAV=data/recordings/<clip>.wav ARGS=--deepfilter   # listen to the spatial filter
make hw                                    # summarise data/hw_*.csv + newest clip -> results/hw_session.json
```
Range on the radar is **near / mid / far learned online** from the session's own
utterances (`distance.relative`, no room calibration); the metric model is still
logged. Choose the cue with `distance.relative.cue: speech_db | cdr_db`.
If PortAudio fails with `-9986` on macOS, replug the ReSpeaker.
Protocol: `doc/03_experiments_log.md` → *Hardware session checklist*.
Angles: degrees counter-clockwise from the robot front, front at the top of the radar.
After `make calibrate`, paste its `azimuth_offset_deg` / `mirror` into the config.

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
Bigger room, 24 angles, 5 SNRs, 4–5 RT60s, 9 distances up to 4 m, 3 seeds.
CPU only (numpy + image-source simulation), ~30–60 min.

## Files
- `src/analysis/vad.py` — home2 pitch VAD (energy + autocorrelation F0 in 85–255 Hz), returns F0, adaptive noise floor
- `src/analysis/doa.py` — SRP-PHAT (1° grid, frequency-domain steering), GCC-PHAT baseline, SNR / F0-harmonic bin weights
- `src/analysis/distance.py` — CDR (Schwarz & Kellermann), level, SNR, tilt features; ridge log-distance model
- `src/analysis/localizer.py` — streaming pipeline: VAD (200 Hz high-pass) → silence noise model → SRP → distance → tracker
- `src/analysis/relrange.py` — microphone-only near/mid/far that learns its reference during the session
- `src/analysis/tracker.py` — wrap-safe alpha-beta tracker (fixes the 359°/1° → 180° average in home2 `respeaker.py`)
- `src/analysis/beamform.py` — delay-and-sum and MVDR toward the tracked talker
- `src/data/simulate.py` — pyroomacoustics scenes with ReSpeaker geometry, voiced talkers, diffuse + fan noise
- `src/data/respeaker_io.py` — 6-ch capture, WAV replay, firmware DOA/VAD/RT60 over USB
- `src/experiments/live_radar.py` — the radar; `record_session.py`, `calibrate_distance.py`, `beamform_file.py` — hardware tools
- `src/experiments/run_*_sim.py`, `run_radar_demo.py` — simulation experiments → `results/*.json`
- `src/experiments/analyze_hw_session.py` — hardware session summary → `results/hw_session.json`
- `doc/06_home2_integration.md` — the plan to bring this into the home2 ROS 2 pipeline

## Headline
Simulation (`04_results.md`):
- **Angle:** the silence-learned noise model is what makes DOA robust. At 0 dB SNR with a
  directional fan, plain SRP-PHAT is off by 35–79° (median); subtracting the silence
  cross-spectra and weighting bins by speech presence brings it to 4–5°. Moving-talker
  demo: median 0.7° error.
- **Distance:** not available from timing differences (6.4 cm aperture). The CDR is strongly
  monotone *within* a room (Spearman −0.90 to −0.95) but room-relative: cross-room error
  0.53 m (CDR+level) → **0.33 m with the room RT60** (guessing the mean: 0.75 m).
- **Selective cancellation:** MVDR at the estimated talker: +2.6 to +10.8 dB SINR against a
  competing talker (mean 7.4); delay-and-sum ≈0 dB.

Hardware, first session (2026-10-05):
- The ReSpeaker works end to end on macOS; the radar follows the talker's direction.
- The real room's hum broke the VAD (**9 speech frames in 26 s** of talking, so no distance
  at all); a **200 Hz high-pass** on the VAD input fixed it (12 → 135 frames on a clip,
  0 false detections in silence). In simulation the same filter costs one reverberant
  0 dB clip — a documented trade-off.
- The simulated distance model does not transfer (0.5 m read as 2.5–5 m; room RT60 0.9 s).
  The radar now shows **near/mid/far learned online**: separates near from far, moves
  with vocal effort. Labelled DOA / distance grid and two-talker MVDR still to do.
