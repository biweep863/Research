# 02 — Methodology

## Pipeline
```
 ReSpeaker 6ch ──► ch1..4 (raw mics, float [-1,1])
                     │  STFT: 512-pt Hann, hop 256 (16 ms), 16 kHz
                     ▼
         ┌─── VAD (mic average, last 1024 samples, ───┐
         │      4th-order Butterworth high-pass 200 Hz)│  (hardware: room hum)
         │ RMS > max(abs gate, floor·10^(6/20))        │  home2 voice_detection logic,
         │ autocorr peak in 85–255 Hz lag band > 0.6   │  + F0 output, + adaptive floor
         └───────────────┬─────────────────────────────┘
              speech     │      silence (after 0.3 s hangover)
                ▼        │          ▼
     Φx_fast (τ 0.25 s)  │   Φn (τ 2 s)  noise cross-spectra  ◄─ "sound of silence"
     Φx_slow (τ 1 s)     │   floor RMS
                │        └───────────┐
                ▼                    ▼
  SRP-PHAT on Φx_fast − Φn, bin weights 1 − tr Φn / tr Φx  (+ optional F0 comb)
                │ azimuth, confidence
                ▼
  features on (Φx_slow, Φn): CDR, level, speech, SNR, tilt
                ├─► ridge model ─► log-distance          (metric; needs the room)
                └─► relative range: near / mid / far     (default on the radar;
                    vs this session's utterance history    learns online)
                │
                ▼
  alpha-beta tracker on the circle (+ log-range EMA) ─► radar / ROS topic
                │
                ▼
  MVDR(Φn, steering to tracked azimuth) ─► mono ─► DeepFilterNet3 (home2)
```
Spatial statistics are reset when speech resumes after > 0.6 s of silence
(the talker may have moved).

## Formulas
- **Model.** `X_m(f) = S(f) e^{-j2πfτ_m}`, `τ_m = −p_m·u(θ)/c`, c = 343 m/s.
- **SRP-PHAT.** `P(θ) = Σ_{i<j} Σ_f w_f Re{ Ĝ_ij(f) e^{j2πf(τ_i(θ)−τ_j(θ))} } / (6 Σ w_f)`,
  `Ĝ_ij = R_ij / |R_ij|`, `R = Φx − Φn`, band 200–3800 Hz (adjacent-pair
  spatial aliasing starts ≈ 3.8 kHz), 1° grid, parabolic peak refinement.
  Confidence `(P_max − mean P)/(1 − mean P)`.
- **GCC-PHAT baseline.** Per pair, PHAT cross-correlation upsampled 16×, peak
  within ±d/c, then least squares `(p_i − p_j)·u = −c τ_ij`.
- **CDR** (Schwarz & Kellermann 2015, DOA-independent estimator) per pair:
  `Γx = Φs_ij/√(Φs_ii Φs_jj)` with `Φs = Φx − Φn`, `Γn = sinc(2πfd/c)`,
  `CDR = [Γn Re Γx − |Γx|² − √(Γn² Re²Γx − Γn²|Γx|² + Γn² − 2Γn Re Γx + |Γx|²)] / (|Γx|² − 1)`,
  only bins with Γn < 0.7 (ill-conditioned below), 400–7000 Hz, median over
  bins, then median over the 6 pairs, in dB.
- **Level** = mean speech power 200–3800 Hz (dBFS); **SNR** = speech over Φn
  power; **tilt** = 2–3.8 kHz over 0.3–1 kHz speech power.
- **Distance model.** `log r = b + Σ_k w_k (x_k − μ_k)/σ_k`, ridge λ = 1.
  Untrained prior: inverse square on level, `r = 10^{(L_1m − L)/20}`.
  With room RT60 known, physics suggests `r ≈ r_c(RT60) · 10^{−CDR/20}`,
  hence RT60 as a covariate.
- **Relative range** (`relrange.py`). Cue `x` = `speech_db` (10 log of the
  noise-subtracted speech power, 200–3800 Hz) or `cdr_db`, per speech frame
  with SNR > 3 dB. Each utterance (ends after 0.6 s of silence) commits
  `median(x)` to a history of the last 20. Reference `c = median(history)`,
  half-width `w = max(3 dB, 1.4826·MAD)`; until 3 utterances exist the history is
  padded with a 2-utterance hardware prior (speech at 1 m: −44 dB). The current
  utterance's running median `x̄` is near if `x̄ > c + w`, far if `x̄ < c − w`,
  else mid; leaving the current band costs 0.5 dB extra (hysteresis). Bands
  are drawn at 0.75 / 1.75 / 3 m.
- **MVDR.** `w = (Φn + δI)^{-1} a / (aᴴ(Φn + δI)^{-1} a)`, δ = 0.01·tr Φn / 4.
- **Metrics.** Angular error `|wrap(θ̂ − θ)|`; distance MAE (m); beamformer
  output SINR gain = (target/residual power after w) − (same at mic 1), with
  w applied separately to the target image and to interferer+noise.

## Evaluation protocol
- DOA: pooled estimate per clip (all speech frames) and per-frame streaming
  estimates. Variants isolate each use of the silence model.
- Distance: leave-one-angle-out CV (generalise across directions) and
  leave-one-RT60-out (generalise across rooms), against predict-the-mean and
  the inverse-square prior.
- Beamforming: target DOA *estimated* (onset subtraction), not oracle.
- Hardware: leave-one-recording-out for both orientation offset and distance
  (`calibrate_distance.py`, needs the labelled grid). Session summary
  (`analyze_hw_session.py`): noise spectrum of a clip's silence, VAD high-pass
  ablation on that clip, live-log counts, per-utterance cues and an offline
  replay of the relative range with each cue.
- Relative range in simulation: band of the true distance (edges at the
  geometric means of the drawn radii, 1.15 / 2.29 m) vs the modal predicted
  band per demo position.

## Statistical caveat
Local sweep: 8 clips per DOA cell, 108 distance clips, 8 beamforming scenes.
Means are reported without CIs in the pilot; `gpu_full.yaml` triples seeds.
