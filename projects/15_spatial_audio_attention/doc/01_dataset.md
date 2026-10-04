# 01 — Data

There is no public dataset recorded with this exact array mounted on this
robot, so the project has two data sources: simulation (now) and labelled
hardware recordings (next session).

## Simulation (`src/data/simulate.py`)
- **Array:** ReSpeaker v2.0 raw mics (ch1–4), ODAS order, radius 32 mm,
  at 1.1 m height (robot head). Placed **off-centre** in the room: a
  mirror-symmetric placement made two mics receive identical reflections for
  talkers on the symmetry axis (perfect coherence → CDR saturates), an
  artefact real rooms do not have.
- **Room:** shoebox, image-source method (pyroomacoustics), absorption from
  Sabine inversion of the target RT60, air absorption on.
- **Talker:** synthetic voiced speech — harmonic source, F0 contour around a
  random F0 in 100–220 Hz, vowel formants per syllable, 4 Hz syllabic
  envelope, word pauses, aspiration noise. Mouth at 1.5 m height. Level
  −30 dBFS at 1 m, ±4 dB vocal-effort jitter in the distance sweep. Each clip
  starts with 0.6 s of silence (the noise model warm-up).
- **Noise:** spherically diffuse pink noise with the exact sinc coherence
  (Cholesky mixing in the STFT domain) + −10 dB uncorrelated mic self-noise;
  optionally a directional "fan" (pink point source at 200°, 2 m, 0.4 m high)
  with the same power as the diffuse part.
- **Interferer** (beamforming): a second voiced talker (F0 210 Hz vs 120 Hz),
  same level, starting 1 s before the target.

|                       | `local_small.yaml`              | `gpu_full.yaml`                    |
|-----------------------|---------------------------------|------------------------------------|
| room                  | 8 × 7 × 3 m, ISM order ≤ 25     | 10 × 9 × 3.2 m, ISM order ≤ 40     |
| DOA sweep             | 8 angles × SNR {0,10,20} × RT60 {0.3,0.6} × {diffuse, fan} | 24 angles × 5 SNR × 4 RT60 × 2 noise × 2 seeds |
| distance sweep        | 6 distances (0.5–3 m) × 3 angles × 3 RT60 × 2 seeds = 108 | 9 distances (0.5–4 m) × 6 angles × 5 RT60 × 3 seeds |
| beamforming           | 4 interferer angles × 2 RT60    | 8 angles × 4 RT60 × 3 seeds        |
| wall clock            | ~2.5 min                        | ~30–60 min (CPU)                   |

The synthetic talker is the main threat to validity: it is more periodic than
real speech (helps the VAD and SRP) and its spectrum is simpler. The F0
harmonic weighting showing no gain is partly *because* the source is already
perfectly harmonic.

## Hardware recordings (`record_session.py`)
6-ch 16 kHz int16 WAVs from the ReSpeaker (6-channel firmware), each with a
`.json` sidecar: angle (deg CCW from robot front), horizontal distance,
speaker id, note, firmware DOA/VAD samples and firmware RT60 when the USB
control interface is reachable. Each clip: 2 s silence, then talking.
Recordings live in `data/recordings/` and are git-ignored (WAV size); the
derived numbers go to `results/hw_calibration.json`.

Planned grid: angles {0, 45, 90, 135, 180, 225, 270, 315} × distances
{0.5, 1, 2, 3} m, two speakers, plus a few clips with a fan / TV on.
