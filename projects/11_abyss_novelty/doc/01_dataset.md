# 01 · Data — a physical soundscape simulator plus real marine-mammal calls

## Synthetic soundscape (`src/sim/`)
- **Species** (`species.py`): a call family + a species-specific parameter
  draw. Families: *tonal* (f0 150–3000 Hz, sweep ±1.5 octaves, 1–4 harmonics,
  vibrato), *click* (ICI 20–400 ms, fc 1.5–7 kHz, 3–30 clicks), *drum* (rate
  5–60 /s, fc 80–600 Hz), *burst* (band-limited noise pulses, 5–200 ms). Each
  clip re-draws the call with ±12 % parameter jitter and a random offset.
- **Propagation** (`propagate.py`): source level 150–185 dB re 1 µPa @ 1 m per
  species; transmission loss = hybrid spreading (spherical to 4 km, cylindrical
  beyond) + Thorp absorption at the call's centre frequency; range log-uniform
  1–100 km; ambient noise = Gaussian noise shaped by the Wenz spectrum (sea
  state 3, moderate shipping). SNR at the receiver is physical (mean ~22 dB in
  the base condition, ~34 dB with a 10 km ceiling, ~16 dB with 300 km).
- **Features** (`novelty/features.py`): 2 s at 16 kHz → STFT 512 → 64 HTK-mel
  bands 50–7500 Hz → log10 → 96 frames → per-clip standardisation.
- Local: 12 known + 5 unknown species, 40 clips each (680 clips); sweeps over
  range ceiling {10, 100, 300} km, known species {4, 8, 12}, clips {15, 40}.

## Real calls: Watkins / BEANS (`src/data/watkins.py`)
Watkins Marine Mammal Sound Database "best of" cuts (WHOI), packaged by the
BEANS benchmark (Hagiwara et al. 2022) and mirrored on the Hugging Face Hub as
`DBD-research-group/beans_watkins` (parquet with wav bytes + species label).
The WHOI site was under maintenance at build time. We downloaded two of the
four shards (train-00000 + valid, 437 MB): 805 clips of 31 species; 796 clips
of 29 species have ≥ 8 clips and are used. Clips are resampled to 16 kHz,
cropped/padded to 2 s, featurised identically, quantised to uint8 and cached
as `data/watkins_mel.npz` (3.1 MB, checked in). Raw shards are git-ignored;
with `WATKINS_DIR` or `data/raw/` the cache rebuilds. Licence: research use
per WHOI/BEANS.

## Survey community (`src/survey/community.py`)
300 species with Fisher log-series relative abundances (α = 40); a sensor
detects species i as Poisson with rate 0.02 · n_i · S · hours. Two habitats
(seamount 300 species, plain 60) for the allocation study, 20 sites.

## Physics inputs (`configs/local_small.yaml: budget`)
Jerlov Kd {I 0.022, IB 0.030, II 0.050, III 0.10} /m; detector floor 1e-9
W/m² vs 100 W/m² surface; σ_sea = 4 S/m; source levels baleen 180 @ 50 Hz,
odontocete click 210 @ 20 kHz, fish chorus 130 @ 300 Hz, snap 150 @ 5 kHz,
quiet 100 @ 1 kHz; DT 10 dB; one-third-octave noise band; eDNA shedding 1e8
copies/h, decay 0.02–0.1 /h, current 0.05 m/s, diffusivity 1 m²/s, floor 0.01
copies/L.
