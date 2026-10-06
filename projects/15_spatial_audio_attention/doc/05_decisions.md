# 05 — Decisions

**Raw mics ch1–4, not ch0.** ch0 is the firmware's processed (beamformed,
AGC'd) channel: one channel carries no spatial information. *Rejected:*
using the firmware `DOAANGLE` only — opaque, no distance, no confidence;
kept as a comparison marker on the radar.

**SRP-PHAT on a 1° grid, not GCC-PHAT peaks.** The largest inter-mic delay
is 0.19 ms ≈ 3 samples at 16 kHz, so integer-lag GCC peaks quantise the
angle into a handful of values. Frequency-domain steering has no lag
quantisation. GCC-PHAT (16× upsampled + least squares) is kept as baseline;
it is fine at high SNR and fails at 0 dB.

**Band 200–3800 Hz for DOA.** Above ~3.8 kHz adjacent mics (4.5 cm) alias
spatially; below 200 Hz the phase differences are too small to matter.

**Silence model = noise cross-spectra, used twice.** Subtracting Φn removes
directional noise; weighting bins by 1 − trΦn/trΦx removes noise-only bins.
*Rejected:* subtraction alone (fails with diffuse noise: noise-only bins
become structured residue that PHAT amplifies — found via a failing unit
test), and weighting alone (fails with a fan).

**Adaptive VAD gate.** home2's `ENERGY_THRESHOLD: 600` (int16) is tuned for
the AGC'd ch0; on raw mics a 3 m talker never passed it. The gate is
`max(100 int16, floor·10^{6/20})` where `floor` follows silence frames
(fast down, slow up). Pitch test unchanged (0.6, 85–255 Hz). The MFCC
speaker lock was not ported: the spatial track does that job here, and
combining both is an integration item.

**Reset spatial statistics after 0.6 s of silence.** Otherwise the first
estimates of a new utterance blend the previous position (seen as the
tracker sticking at the old angle in the demo).

**Distance from acoustic cues, not geometry.** The aperture is 6.4 cm; even
at 0.5 m the talker is in the far field and the curvature of the wavefront
is unmeasurable. *Rejected:* near-field SRP over (θ, r), which is flat in r.
Elevation (mouth above array) varies with distance too, but a planar array
cannot separate elevation sign and it changes with the talker's height.

**DOA-independent CDR estimator, bins with Γn < 0.7, median over pairs.**
The DOA-dependent estimator assumes a horizontal plane wave, but the mouth
is ~0.4 m above the array, and the error is largest at short range. Low
bins: diffuse and direct coherence are both ≈1 → ill-conditioned, biased
high. Median over pairs: robust to one degenerate pair.

**RT60 as a room covariate.** CDR is room-relative (critical distance
r_c ∝ √(V/RT60)). The ReSpeaker firmware exposes an RT60 estimate; the live
radar switches to the RT60 model when it is available or when
`distance.room_rt60_s` is set. *Rejected:* per-room calibration only —
works, but the robot moves between rooms in competitions.

**Saved default model = CDR + level.** Best cross-angle CV without the
room covariate; all-feature model overfits (SNR and tilt add variance).

**Off-centre array in simulation.** A centred array in a shoebox made two
mics receive identical reflections for talkers on a symmetry axis
(coherence exactly 1). Real rooms are not symmetric.

**MVDR, not delay-and-sum.** At 1 kHz the array is 0.19 λ wide; DS has
almost no directivity (measured ≈0 dB). MVDR is superdirective but needs a
good noise covariance (from target-silent frames) and diagonal loading
(0.01 of per-bin power) against steering errors.

**Output SINR instead of SI-SDR vs mic 1** for beamformers: SI-SDR counts the
beamformer's own (harmless) linear filtering of the target as distortion.

**Synthetic voiced talker instead of a speech corpus.** No network/data
dependency, controllable F0, and it exercises the pitch VAD exactly. Threat
to validity acknowledged in `01_dataset.md`; hardware recordings replace it.

## Hardware session (2026-10-05)

**VAD input high-passed at 200 Hz.** Room hum on the raw mics (65 % of the
power < 100 Hz, peaks 20 / 80 Hz) dominated the broadband RMS gate: 9 speech
frames in 26 s of talking. 150 Hz fixed coverage but let hum harmonics pass the
85–255 Hz pitch test (13/68 silence frames called speech, which also poisons
Φn); 200 Hz gives the same coverage with 0/68. Speech pitch survives through
its harmonics. Only the VAD input is filtered: SRP and the distance features
already work in their own bands. *Cost:* in simulation (no hum) 200 Hz lets
one reverberant 0 dB clip go wrong (`run_vad_highpass_sim`); 100 Hz is best
there. Hardware evidence wins; 100 Hz is the setting to try in a quiet room.

**Microphone-only range = near/mid/far learned online, not a room model.**
The simulation-trained metric model read 2.5–5 m for a talker at 0.5 m in a
real room (RT60 0.9 s, quieter talker). A per-room calibration would fix it
but the robot changes rooms in competition, and the user asked for something
that adapts by itself. `relrange.py` compares each utterance with the
session's own utterance history (median ± max(3 dB, MAD)), with hysteresis.
Noise reference = the silence model (re-learned continuously); voice reference
= past utterances. The only constant is a hardware prior (normal speech at 1 m
on the raw mics, −44 dB, a property of the fixed-gain mics, not the room),
which real utterances wash out after three. *Rejected:* fixed near/far
cut-offs (that is a calibration); terciles of the history (forces three bands
on a talker who never moves). The metric model is still computed and logged
(`model_range_m`) for comparison.

**Band floor 3 dB, hysteresis 0.5 dB.** 1 m → 3 m is ~7.5 dB of speech level
in the simulated room (reverberation flattens the inverse-square law);
4 dB never left "mid" for 1 m, 2.5 dB flickered at 2 m.

**Cue configurable, speech level by default for now.** `distance.relative.cue`:
`speech_db` (noise-subtracted speech power; moves with vocal effort) or
`cdr_db` (direct-to-reverberant; effort-invariant but room- and
SNR-sensitive). Live, the two agree on 59 % of utterances and the CDR switches
less (18 % vs 28 %), but without distance labels the choice is not settled;
the labelled protocol is in `03_experiments_log.md`.

**Radar demo scored in two passes.** Metric model (relative off) for the
azimuth / metres numbers, relative on for the band accuracy, so the paper's
metric numbers stay comparable to the pilot.

**macOS is a supported host for the hardware tools.** sounddevice and pyusb
both work; firmware control transfers need no driver change. A recording
interrupted mid-stream can wedge CoreAudio (PortAudio −9986): replug.
