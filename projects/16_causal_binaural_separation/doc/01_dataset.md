# 01 · Dataset

All mixtures are **simulated binaurally from real recordings**: real speech,
real noise and measured heads, placed in simulated rooms. Simulation is the
only way to get the clean binaural target that the cue metrics need.

| Component | Local (`local_small.yaml`) | Full (`gpu_full.yaml`) | License |
|---|---|---|---|
| Speech | LibriSpeech `dev-clean` (5.4 h, 40 speakers, split 30/5/5 speaker-disjoint) | `train-clean-100` / `dev-clean` / `test-clean` | CC BY 4.0 |
| Noise | DEMAND ch01: 6 envs (time split 70/10/20) | 12 envs | CC BY-SA 3.0 |
| Held-out noise | NPARK, OMEETING, SPSQUARE (never trained on) | + SCAFE, TCAR, PSTATION | CC BY-SA 3.0 |
| HRTF (train/test) | MIT KEMAR, normal pinna (710 dirs, bundled with pyroomacoustics) | same | MIT, free use |
| HRTF (held-out) | CIPIC subject 003 (1250 dirs; a different head) | same | public domain (CIPIC) |
| Rooms | 150 train / 15 val / 40 test / 40 held-out | 1000 / 50 / 200 / 100 | — |

## Room responses (`src/data/rooms.py`)
- Shoebox rooms, 3.5–9 × 3–7 × 2.5–3.5 m. RT60 is drawn uniformly from
  0.2–0.8 s. The listener stands 1 m from every wall, at an ear height of
  1.2–1.7 m, facing +x.
- The target is placed at a random azimuth (full circle), at elevation −20…40°
  and distance 0.5–2.5 m. Six noise sources sit anywhere in the room
  (0.5–6 m away).
- Image sources come from pyroomacoustics up to order 6. Each image is
  rendered through the HRIR nearest to its arrival direction. Past the ISM
  horizon, a binaural diffuse tail (32 random-direction HRIRs driven by
  independent noise) decays at the room's RT60 and is level-matched to the
  ISM. Measured T60 is within 0.02 s of the target on spot checks.
- **Target = direct path + reflections within 50 ms**, rendered from the same
  images. Late reverberation of the talker counts as interference.

## Mixtures (`src/data/mixtures.py`, on the GPU)
- Training mixtures are drawn fresh at every step, so the training set never
  repeats: 2 s crops (local) or 3 s (full), each with 0.5 s of reverb warm-up.
- Between one and six noise sources are active, each at ±6 dB relative gain.
  SNR (early target vs. noise at the ears) is drawn from −5…15 dB. The
  mixture level is drawn from −35…−15 dBFS.
- Evaluation sets are fixed and cached:
  - `val`: 64 × 3 s;
  - `test`: 300 × 3 s, with unseen speakers, unseen rooms, and later
    recording time of the training noise environments;
  - `heldout`: 150 × 3 s, with the CIPIC head, unseen rooms, and unseen noise
    environments.

## Fallback
With no network or files, speech comes from project 15's voiced-speech
generator, noise is pink, and the HRTF is the Brown–Duda spherical head.
Every result JSON carries `synthetic_data`, and the paper must not be built
from a synthetic run.

## Sizes on disk
Raw downloads are about 1.3 GB. The cache (`data/cache`: decoded pools, BRIR
banks, fixed sets) is about 1.4 GB locally. Both are git-ignored.
