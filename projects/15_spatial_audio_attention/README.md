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
make test     # 33 unit tests, ~2 s
make run      # DOA / distance / beamforming sims + moving-talker demo (~2.5 min)
make figures  # paper/figures/*.pdf + paper/macros.tex
make demo     # replay the simulated moving talker on the live radar
make paper    # compile paper/main.pdf
```
Extra deps (installed in the repo venv): `pyroomacoustics sounddevice soundfile pyusb`.
Optional: `deepfilternet` for `make beamform ARGS=--deepfilter`.

## On the robot (hardware session)
```bash
make live ARGS="--list-devices"            # find the ReSpeaker (needs 6-ch firmware)
make live                                  # radar from the ReSpeaker; firmware DOA shown dashed
make live ARGS="--no-plot --log run.csv"   # headless (ssh / docker) + per-frame CSV
make record ANGLE=90 DIST=1.5              # 2 s silence, then talk; labels in a .json sidecar
make calibrate                             # orientation offset + distance model from recordings
make beamform WAV=data/recordings/<clip>.wav ARGS=--deepfilter   # listen to the spatial filter
```
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
- `src/analysis/localizer.py` — streaming pipeline: VAD → silence noise model → SRP → distance → tracker
- `src/analysis/tracker.py` — wrap-safe alpha-beta tracker (fixes the 359°/1° → 180° average in home2 `respeaker.py`)
- `src/analysis/beamform.py` — delay-and-sum and MVDR toward the tracked talker
- `src/data/simulate.py` — pyroomacoustics scenes with ReSpeaker geometry, voiced talkers, diffuse + fan noise
- `src/data/respeaker_io.py` — 6-ch capture, WAV replay, firmware DOA/VAD/RT60 over USB
- `src/experiments/live_radar.py` — the radar; `record_session.py`, `calibrate_distance.py`, `beamform_file.py` — hardware tools
- `src/experiments/run_*_sim.py`, `run_radar_demo.py` — simulation experiments → `results/*.json`
- `doc/06_home2_integration.md` — the plan to bring this into the home2 ROS 2 pipeline

## Headline (simulation; hardware pending)
- **Angle:** the silence-learned noise model is what makes DOA robust. At 0 dB SNR with a
  directional fan, plain SRP-PHAT is off by 69–76°; subtracting the silence cross-spectra and
  weighting bins by speech presence brings it to 3–4°. Moving-talker demo: median 1.0° error.
- **Distance:** not available from timing differences (6.4 cm aperture). The direct-to-reverberant
  cue (CDR) is strongly monotone *within* a room (Spearman −0.90 to −0.95), but its offset depends
  on the room: cross-room error 0.55 m (CDR+level) → **0.33 m with the room RT60 as a covariate**
  (always guessing the mean distance: 0.75 m).
- **Selective cancellation:** MVDR steered at the estimated talker, noise covariance from
  target-silent frames: +2.6 to +10.8 dB SINR against a competing talker. Delay-and-sum: ≈0 dB
  (the array is too small for it).
