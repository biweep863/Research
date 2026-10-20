# 05 · Decisions & alternatives

## One project, not two
Localization on this array is project 15. Project 19 reuses its estimator
and contributes the hardware DOA labels and calibration back to it through
`export_p15.py`. A second localization paper would have repeated project 15.

## Superposition instead of only live takes
Live takes cannot be scored exactly: there is no clean component, and the
close-talk mic also hears the speaker. Recording speech and noise separately
on a fixed array gives:
- exact output SNR for every linear front end;
- 320 scenes from 40 minutes of recording;
- every interferer at every SNR against every utterance.

The price is that the room noise is counted twice and the talker's body is
missing in the noise takes. The talker live take measures that gap.

## Talker moves, array and speakers stay
Rotating the array instead would keep the talker-to-interferer separation
constant, and the noise takes would no longer match the geometry. With two
speakers driven separately, each utterance gets two different separations
for free.

## RTF steering (GEVD) next to DOA steering
Project 15 steers MVDR with a far-field DOA. On real hardware at 0.5 m the
far-field model and the DOA itself (median error 47° at 0 dB input SNR) are
both off. The covariance-whitening RTF needs neither: it learns the talker's
real acoustic transfer function, reverberation included, from the mixture
and the silence model. Keeping both versions measures what geometry costs.

## The user's mute idea, implemented literally and softly
"Shutting the mic that faces the noise" became two methods:
- `mute`: drop the noisiest mic.
- `mute_soft`: per-band inverse-noise weights.

Both are named baselines. A unit test also checks the analytic prediction
that, on a 6.4 cm array of omni mics, a far-field interferer reaches every
mic at the same level.

## DeepFilterNet in its own venv
- **Problem:** DeepFilterLib 0.5.6 has Windows wheels only for Python ≤ 3.11.
  The repo venv is 3.12, and building from source needs Rust.
- **Fix:** `.venv-dfn` is a uv-managed Python 3.11 venv with torch 2.4.1 CPU
  and DFN 0.5.6, called through `src/dfn_runner.py`.
- **DLL fix:** torchaudio's `libtorchaudio.pyd` needed `vcomp140.dll`, which
  is missing system-wide. It was copied into the venv from scikit-learn's
  bundled runtime.
- **Speed:** DFN3 on CPU runs at about 0.03× real time.

## Whisper small, not large-v3, locally
`small` (fp16, GPU) keeps the full sweep (about 4,000 transcriptions) near
an hour. `gpu_full.yaml` switches to large-v3.

On Windows, CTranslate2 needs cuBLAS 12 and cuDNN 9. The CUDA torch wheel
ships them, and `asr.py` adds `torch/lib` to the DLL search path.

## Not used
- **PESQ:** no Windows wheel, and it needs a C++ toolchain.
- **DNSMOS:** an extra model; WER is the downstream that matters for home2.
- **Firmware DOA:** on Windows, pyusb needs a libusb driver (Zadig) on the
  control interface. That was not installed, to avoid touching the audio
  driver.
