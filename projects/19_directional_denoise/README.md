# Project 19 — Directional noise cancellation on the 4-mic ReSpeaker, before DeepFilterNet3

## Question
On a real 4-mic array with a noise source in one direction, how much do
spatial filters (mic muting, delay-and-sum, MVDR, adaptive GSC) improve speech
before and after DeepFilterNet3? How should DFN3's `atten_lim_db` be set
when the listener is Whisper rather than a person?

## Answer (one session, 320 mixed scenes + 2 live takes)

### Spatial filters
| Front end | SNR gain | Whisper WER, 0 dB, no DFN |
|---|---|---|
| raw mic ch1 | 0 | 74 % |
| **mute the noisiest mic** (intuitive idea) | +0.4 dB | 73 % (n.s.) |
| delay-and-sum | +0.4 dB | 70 % |
| MVDR, far-field DOA | +2.9 dB | 54 % |
| GSC (NLMS) | +3.7 dB | 54 % |
| **MVDR, RTF from the silence model** | **+4.3 dB** | **50 %** (p = 5e-9) |
| oracle MVDR | +9.1 dB | 20 % (clean: 15 %) |

### DeepFilterNet3
- **Never lowers the WER.**
  - At atten 6–12 dB the change is not significant.
  - Without a limit it adds +16 to +19 points.
  - ESTOI rises monotonically with the limit: it sounds cleaner and is
    recognized worse.
- **home2's mode costs accuracy.** Independent 64 ms chunks add +6 to +11
  points compared with one continuous pass.
- **`atten_lim_db=0` means *no limit*, not off.**

### Live takes
With a second (English) talker playing, the chip's processed **ch0**
(home2's input) gives a WER of **200 %** with DFN: Whisper transcribes the
*other* voice. MVDR + DFN gives **11 %**.

### Project 15 (same session)
- The array is mirrored and offset by 35° from the talker's marks.
- DOA leave-one-out error is 16°.
- Distance does not work on hardware (|Spearman| ≤ 0.30).

## Quick start
```bash
cd projects/19_directional_denoise
make test        # 6 unit tests (~2 s)
make all         # spatial (~4 min) + DFN/Whisper sweep (~2 h, GPU) + live + figures + paper
```
The recordings are not in git (`data/session/`, about 240 MB). The protocol to
record a new session, about 40 minutes, is in `doc/01_dataset.md`:
```bash
make calibrate LEVEL=-22
make record KIND=noise ARGS="--noise pink --channel L"
make record KIND=speech ARGS="--angle 90 --dist 0.5 --text t6"
```

### Environments
- **Main venv:** `faster-whisper`, `pystoi`, `jiwer`, plus project 15's
  `sounddevice`, `soundfile` and `pyusb`.
- **DeepFilterNet3:** runs in `../../.venv-dfn`, a Python 3.11 venv made with
  uv, with torch 2.4.1 CPU and `deepfilternet==0.5.6`. There is no Windows
  wheel for 3.12; see `doc/05_decisions.md`.

### Windows
The ReSpeaker gives 6 channels only with **audio enhancements disabled**.
Uncheck `mmsys.cpl` → Recording → ReSpeaker → Advanced → "Enable audio
enhancements", then replug.

## Scaling up
`make all CONFIG=configs/gpu_full.yaml` uses:
- every SNR, through DFN and ASR;
- 9 attenuation limits;
- home2 mode for 5 front ends;
- Whisper large-v3.

## Files
- `src/data/session.py`: one take; plays chirp, beep and interferer while
  recording array and reference.
- `src/data/align.py`: chirp matched filter; offset and clock-rate ratio.
- `src/data/scenes.py`: takes → aligned segments → mixed scenes.
- `src/analysis/spatial.py`: mute, delay-and-sum, MVDR (DOA or RTF), GSC,
  oracle.
- `src/dfn_runner.py`: DFN3 batch runner (continuous or home2 chunk mode),
  run in `.venv-dfn`.
- `src/analysis/asr.py`: faster-whisper.
- `src/analysis/metrics.py`: SI-SDR, ESTOI, WER.
- `src/experiments/run_spatial.py`, `run_dfn_asr.py`, `run_live.py`: stages
  1–3.
- `src/experiments/export_p15.py`: export labelled clips to project 15.
- `src/analysis/{geometry,doa,stft,beamform}.py`: vendored from project 15.
