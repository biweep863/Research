# 16 · Causal binaural speech enhancement at ultra-low latency

## Question
How much enhancement quality and binaural-cue fidelity does a strictly causal
network lose as its algorithmic latency drops from 32 ms to 1 ms, and can
capacity spent on the *past* (long asymmetric analysis windows, recurrent
state, deep filtering over past frames) recover what is lost by not hearing
the future?

## Quick start
```bash
make test                       # metrics, causality, streaming == offline, data
make sanity                     # tiny end-to-end run (< 5 min) -> results/sanity/
make all                        # data -> oracle -> train sweep -> eval -> bench -> figures -> paper
```
On Windows set `PY=../../.venv/Scripts/python.exe`. The first `make data`
downloads LibriSpeech dev-clean (~340 MB), nine DEMAND environments (~0.9 GB)
and one CIPIC HRTF into `data/raw/` (git-ignored). Without network every
loader falls back to synthetic speech/noise and a spherical-head HRTF, and
the results say `synthetic_data: true`.

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml   # train-clean-100, 1000 rooms, 1-2M-param models, 6 latencies
```

## Files
- `src/models/framing.py` — causal WOLA framing + asymmetric windows (latency = 2·hop)
- `src/models/stft_net.py` — `AsymDF`: asymmetric-window GRU + per-ear deep filter / common gain
- `src/models/tasnet.py` — binaural causal Conv-TasNet with bounded streaming state
- `src/models/classical.py` — oracle ratio masks, causal common-gain binaural Wiener
- `src/models/losses.py` — binaural SNR + interaural-cue (ILD/IPD) loss
- `src/data/` — LibriSpeech/DEMAND pools, SOFA HRTFs, ISM + HRIR room responses, GPU mixer
- `src/analysis/` — SI-SDR, ESTOI, PESQ (optional), ITD/ILD/IPD errors, MAC counts
- `src/experiments/` — `prepare_data`, `run_oracle_sweep`, `train_sweep`, `evaluate`, `bench_latency`
- `src/figures/plot_latency.py` — figures + `paper/macros.tex`
- `paper/main.tex` — the write-up; every number comes from `macros.tex`
- `doc/` — overview, dataset, methodology, experiment log, results, decisions
