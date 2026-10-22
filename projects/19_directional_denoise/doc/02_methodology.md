# 02 · Methodology

## Scenes by superposition (`src/data/scenes.py`)
`y = s + g·n` on the 4 raw mics:
- `s` is a speech-only take: 1.5 s of the talker's silence followed by the
  utterance, cut so that neither the beep nor the chirps are inside.
- `n` is a random excerpt of a noise-only take.
- `g` sets the input SNR (−5/0/5/10 dB). The SNR is measured on the
  mic-average power, speech over its active samples and noise over the same
  samples.

The result is 8 utterances × 5 interferers × 2 speakers × 4 SNRs = **320
scenes**. Because both takes come from the same fixed array, the mixture is
what the array would hear with both sources on, except that:
1. the room's own noise is counted twice (15–20 dB below the interferers);
2. the talker's body was not present during the noise takes.

The talker live take measures how much these two matter (stage 3).

**What a method may use.** It may use the mixture and its first 1.5 s,
where the talker is silent. That is project 15's *silence model*, the only
noise statistics a deployed system has. Only `oracle` uses the true
components.

## Front ends (`src/analysis/spatial.py`), STFT 512/128 at 16 kHz
| ID | Weights |
|---|---|
| ref | raw mic ch1 (the output reference of every method) |
| mean | average of the 4 mics |
| mute | **the user's idea**: drop the mic with the largest silence-model power, average the other 3 |
| mute_soft | per-bin inverse-noise-power weights, real (no phase alignment) |
| ds | delay-and-sum toward the DOA (project 15's SRP-PHAT on mixture − silence covariance) |
| mvdr_doa | MVDR, far-field steering at that DOA, silence-model `Rn`, loading 1e-2 |
| mvdr_rtf | MVDR, steering = RTF from covariance whitening (GEVD of `Ry`, `Rn`), loading 1e-3. No DOA, no geometry |
| gsc | Griffiths–Jim GSC: fixed beam `h/|h|²` + blocking matrix ⟂ RTF + per-bin NLMS (μ 0.05), adapting only when the fixed beam is within 3 dB of its noise-only level |
| oracle | MVDR with the true `Rn` and the RTF from the true speech covariance |

Every method is linear in the input. Passing `S` and `N` through the *same*
weights gives the **exact output SNR** (`snr_out − snr_in` on ch1). The GSC
weights are time-varying but still per-frame linear.

## DeepFilterNet3 (`src/dfn_runner.py`, own venv)
- **Model:** DFN3 0.5.6 at 48 kHz. Input is upsampled ×3 and peak-normalised,
  as in home2.
- **`atten_lim_db`:** {no DFN, 6, 12 (home2), 18, 24, no limit}.
  `atten_lim_db = 0` means *no limit* in DF, not off (verified: white noise
  −39 dB for both 0 and None, −12.0 dB for 12).
- **`cont` mode:** one `enhance()` per utterance, so the recurrent state
  persists.
- **`home2` mode:** exactly `noise_cancellation.py`, with 1024-sample (64 ms)
  chunks, each peak-normalised and enhanced independently.

## Scores (`src/analysis/metrics.py`, `asr.py`)
- **WER:**
  - Whisper `small` (faster-whisper, fp16 on the RTX 3060), Spanish, greedy,
    no VAD filter, no context.
  - Reference is the read prompt, normalised: lower case, no accents or
    punctuation, numerals → words. Values above 100 % mean insertions, here
    usually the English interferer transcribed.
  - Floor: the same Whisper on the clean speech-only mic and on the Akane.
- **SI-SDR** against the clean ch1 speech component (reverberant). It
  penalises de-reverberation, so it is secondary.
- **ESTOI** against the Akane close-talk reference.

## Live takes (`run_live.py`)
- **Superposition check.** The talker live take replayed the same
  interferer samples as `noise_talker_L`. In the talker-silent 1.5 s before
  the beep, `live ≈ g·noise_take`; `g` is fitted by least squares, and the
  residual energy relative to the live take measures repeatability. The
  residual over the utterance is the live speech component, so the live
  take also gets exact SNR gains.
- **Front ends** are run on the live audio, together with the chip's ch0,
  each with and without DFN (atten 12 and no limit, `cont` and `home2`
  modes), and scored by WER against the prompt.

## DOA (`run_spatial.py`)
- **Labels** come from SRP-PHAT on each source's clean take.
- **Estimate** is from the mixture, using project 15's method.
- **Error** is the circular difference. On the labelled grid,
  `calibrate_distance.py` (project 15) fits the array → talker-mark
  orientation and reports leave-one-out error.
