# 04 — Results (`local_small.yaml`)

Simulation sweeps + first hardware session (2026-10-05, ReSpeaker on a MacBook).
Formal write-up: `paper/main.pdf`. All simulation numbers below were regenerated
on 2026-10-06 with the final config (VAD high-pass 200 Hz, relative range on);
the pilot numbers of 2026-10-04 are in git history.

## Angle of arrival (8 angles × 2 RT60, talker at 1.5 m; pooled per clip)
At **0 dB SNR**, RT60 0.3 / 0.6 s. MAE of the pooled per-clip estimate, and in
brackets the median of the streaming per-frame estimates:

| estimator                         | diffuse noise              | fan + diffuse              |
|-----------------------------------|----------------------------|----------------------------|
| SRP-PHAT + silence model (full)   | 2.1 / 15.0 (2.6 / 2.7)     | 5.8 / 14.1 (4.5 / 4.3)     |
| − noise subtraction               | 1.4 / 14.5 (1.9 / 1.9)     | 9.7 / 16.5 (10.1 / 6.9)    |
| − speech-presence bin weights     | 21.9 / 35.3 (6.8 / 7.6)    | 6.9 / 28.2 (7.7 / 7.1)     |
| plain SRP-PHAT                    | 8.5 / 22.3 (7.6 / 19.6)    | 42.9 / 56.4 (35.0 / 78.9)  |
| GCC-PHAT + least squares          | 19.0 / 26.6                | 7.3 / 51.9                 |
| + F0 harmonic weights             | 2.1 / 15.3 (3.3 / 2.7)     | 5.8 / 14.5 (5.0 / 4.1)     |

7 of 8 clips are detected in each 0 dB cell (the 315° clip has no frame that
passes the VAD). At 10–20 dB SNR the full method is within 0.7–1.7°.

- The silence model still separates the methods: plain SRP-PHAT locks on the fan
  (median 35–79°), the full method stays at a 3–5° median.
- The **RT60 0.6 s MAE of 14–15° is one clip** (135°: 73° / 60° off). The VAD
  high-pass needed on hardware (below) costs this in simulation, see the next table;
  with the pilot's VAD (no high-pass) the same cells were 3.0 / 2.6°.
- F0 harmonic weighting still adds nothing on the synthetic talker.

### VAD high-pass in simulation (`run_vad_highpass_sim.py`, 0 dB, full SRP)
MAE (max) in degrees:

| cut-off | diffuse 0.3 | diffuse 0.6 | fan 0.3 | fan 0.6 |
|---------|-------------|-------------|---------|---------|
| none    | 3.2 (13)    | 3.0 (7)     | 4.2 (14)| 2.6 (5) |
| 100 Hz  | 1.8 (4)     | 3.2 (8)     | 3.0 (6) | 3.2 (6) |
| 150 Hz  | 1.9 (5)     | 2.8 (9)     | 5.1 (11)| 11.8 (60)|
| 200 Hz  | 2.1 (5)     | 15.0 (73)   | 5.8 (11)| 14.1 (60)|

The simulated noise has no low-frequency hum, so the filter only removes F0
energy here; at RT60 0.6 s it lets one reverberant clip go wrong. On hardware
the same filter is what makes the VAD work at all. 200 Hz is kept as the default
(hardware wins); 100 Hz is the best simulation setting and the fallback to test
in a quieter room.

### Moving-talker demo (7 positions, 0.5–3 m, RT60 0.45 s, streaming)
- Azimuth: median **0.7°**, 90th percentile 7.1°.
- Metric range (sim-trained model, same room): median error **0.13 m**, 3 m → ~2.2 m.
- Online **near/mid/far** (relative mode, speech-level cue, no room model):
  modal band right at 4 of 7 positions, 51 % of frames. Right at the extremes
  (1 m first visit, 3 m), wrong between 1 and 2.5 m. 1 m → 3 m is only ~7.5 dB
  of speech level in this room.

## Distance (108 clips, 0.5–3 m, 3 RT60s, ±4 dB vocal effort)
Spearman with log-distance (pooled over rooms / within each room):
CDR −0.77 / −0.90…−0.95; level −0.62; SNR −0.44; tilt +0.08.

| model (MAE, m)          | leave-angle-out | leave-room-out |
|-------------------------|-----------------|----------------|
| predict mean            | 0.75            | —              |
| inverse-square prior    | 0.92            | —              |
| CDR                     | 0.44            | —              |
| CDR + level             | 0.42            | 0.53           |
| all 4 features          | 0.60            | 0.50           |
| **CDR + RT60**          | **0.24**        | **0.33**       |

Loudness alone is a poor range cue once people speak at different levels.
The CDR is the cue, but it is room-relative; giving the model the room RT60
halves the cross-room error.

## Selective cancellation (target 0°, competing talker, SIR 0 dB, 1.2 m)
Output SINR gain vs mic 1, DOA estimated by onset subtraction (≤ 9.6° error):

| interferer | 45° | 90° | 135° | 180° |
|------------|-----|-----|------|------|
| MVDR, RT60 0.3 | 5.6 | 9.8 | 10.8 | 10.7 |
| MVDR, RT60 0.6 | 2.6 | 6.7 | 6.8 | 6.0 |
| delay-and-sum  | 0.0–0.5 for all cases |||| 

Mean MVDR gain **7.4 dB**; delay-and-sum 0.2 dB (unchanged by the VAD change:
this experiment uses the onset split, not the VAD).

## Hardware session (2026-10-05) — `make hw` → `results/hw_session.json`
ReSpeaker USB v2.0, 6-ch firmware, 16 kHz, on macOS (sounddevice/PortAudio).
Firmware DOA/VAD/RT60 readable over USB (pyusb) on macOS as well. Firmware
RT60 estimate: 0.9 s (outside the simulated 0.3–0.6 s). Raw data (WAV, CSV)
is local and git-ignored; the JSON keeps every derived number.

**Room noise on the raw mics** (2 s silence of a labelled clip): −41.8 dBFS
broadband, **65 % of the power below 100 Hz** (peaks ≈ 20 and 80 Hz), 81 %
below 200 Hz; −49.5 dBFS after a 200 Hz high-pass. The four mics are 0.98–0.99
correlated (low-frequency field). Speech at 0.5 m raised the *broadband* level
by < 1 dB.

**The VAD was the failure, not the localizer.** First live run, VAD as in
simulation: **9 speech frames out of 1645** (26 s of talking), 0 distance
estimates; the radar showed the hollow "no range" marker the whole time.
Direction from those few frames was stable (SRP works in 200–3800 Hz, above the
hum). VAD high-pass ablation on the 0.5 m clip:

| VAD input       | speech frames while talking | "speech" in silence |
|-----------------|-----------------------------|---------------------|
| broadband       | 12 / 477                    | 0 / 68              |
| 150 Hz high-pass| 134 / 477                   | **13 / 68** (hum at 80–150 Hz passes the pitch test) |
| 200 Hz high-pass| 135 / 477                   | 0 / 68              |

With 200 Hz, an 18-minute live session gives 22 % speech frames (voiced
frames only; unvoiced consonants and pauses are not speech for this VAD).

**Metric range does not transfer.** The 0.5 m clip: CDR −3.2 dB (simulation at
0.5 m: +7…+10 dB), SNR 10.6 dB, sim-trained model → 2.5 m. The room is more
reverberant than any simulated room and the talker is quieter than the
simulated −30 dBFS @ 1 m.

**Online relative range, live** (18 min, 238 utterances with ≥ 5 frames,
no distance labels — the protocol was ~1 m, near, ~3 m, back):
- Speech-level cue (the radar's default): bands near/mid/far 14/74/12 %,
  band changes between consecutive utterances 28 %.
- Replaying the same utterances with the CDR cue: 20/69/11 %, 18 % changes.
- The two cues agree on the band of 59 % of the utterances; Spearman between
  them across utterances is only **+0.28**. Without labels we cannot say which
  one is right more often; the CDR switches less and does not depend on vocal
  effort, the speech level does.
- Observed by the user: near vs far is distinguished, but talking louder or
  softer moves the band.

## Hardware session 2 (2026-10-18, Windows, recorded with project 19)
15 labelled clips (`data/recordings/20261018_*`, exported by project 19's
`export_p15.py`): 8 angles at 0.5 m and 7 distance clips at 270°
(0.4/1/2 m, soft/normal/loud). The talker's angle marks are approximate
(about ±20°). `make calibrate` → `results/hw_calibration.json`:
- **Orientation:** `azimuth_offset_deg: 35.2`, `mirror: true` (in-sample
  median 14.5°).
- **DOA leave-one-out median error: 15.8°.** This is bounded by the marks;
  the firmware DOA was not read (pyusb needs a libusb driver on Windows).
- **Distance does not work** at 0.4–2 m in this room. The largest
  |Spearman| against log-distance is 0.30 (tilt; CDR 0.26, level 0.04).
  The leave-one-out MAE of every model (0.40–0.43 m) equals predicting the
  mean (0.41 m), and the simulation prior is off by 4.2 m. Vocal effort
  moves the level more than distance does.
- **The DOA estimator under a directional interferer** (project 19, 320
  mixed scenes, label = SRP-PHAT on the clean take): median error 65/58/11/5°
  at −5/0/5/10 dB SNR. Simulation (diffuse noise + fan) gave 4–5° at 0 dB.
  Against a loudspeaker playing speech or babble at 0 dB, the silence model
  is not enough.
- **H3 on hardware** (project 19): MVDR toward the talker gains +2.9 dB with
  this project's DOA steering and +4.3 dB with RTF steering (median over 320
  scenes). In a live take with a second talker it cuts Whisper's WER from
  200 % (chip ch0 + DFN3) to 7–11 %.

## Verdict
- **H1 (angle)** holds in simulation. On hardware the labelled
  leave-one-out error is 16° in quiet (bounded by the marks), but 58° at
  0 dB against a loudspeaker interferer.
- **H2 (distance)** holds in simulation only with the room covariate, and the
  simulated model does not transfer to the real room. Microphone-only output is
  near/mid/far, learned online: right at the extremes, unreliable in between.
- **H3 (selectivity)** holds on hardware: MVDR +2.9 dB (DOA) and +4.3 dB
  (RTF) SNR, with large WER gains (project 19).
