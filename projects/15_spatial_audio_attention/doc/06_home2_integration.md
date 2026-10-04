# 06 — Plan: bringing spatial audio attention into home2

Target: `RoBorregos/home2` → `hri/packages/speech` (ROS 2). Read against
home2 `a0aee4ca`. Nothing in home2 is changed by this project; this is the
plan for the second phase, gated on the hardware results.

## Current audio graph
```
audio_capturer.py ──/hri/rawAudioChunk (ch0 only, int16, 1024)──► noise_cancellation.py (DeepFilterNet3)
                                                                         │ /hri/processedAudioChunk
respeaker.py ──/hri/speech/respeaker/doa (Int16, firmware DOA,           ▼
               linear moving average)                            voice_detection.py ──► /hri/vadAudioChunk
                                                                         │                /hri/voice_activity (Bool)
                                                                         ▼
                                                                  hear_streaming.py (STT)
```

## Proposed graph
```
audio_capturer.py ─┬─ /hri/rawAudioChunk            (ch0, unchanged: nothing downstream breaks)
                   └─ /hri/rawAudioMultichannel     (NEW: ch1..4 interleaved int16, AudioDataStamped)
                              │
                              ├──► sound_localization.py (NEW) ──► /hri/speech/speaker_position
                              │      uses /hri/voice_activity        (PointStamped in frame "respeaker_link":
                              │                                       x,y from az+range; z = confidence? → custom msg)
                              │
                              └──► spatial_filter.py (NEW, MVDR) ──► /hri/spatialAudioChunk (mono)
                                         steering from speaker_position
                                                    │
                       noise_cancellation.py: RAW_AUDIO_TOPIC := /hri/spatialAudioChunk  (param switch only)
```

## Steps
1. **Multichannel topic (small, safe).** In `audio_capturer.py`, besides
   `data[0::6]`, publish `np.frombuffer(data, int16).reshape(-1, 6)[:, 1:5]`
   on `RAW_MULTICHANNEL_AUDIO_TOPIC` (new constant in
   `frida_constants/hri_constants.py`). Use `AudioDataStamped` so the
   localizer can align with `/hri/voice_activity`. Bandwidth: 4 × 16 kHz ×
   2 B = 128 kB/s — fine for a local topic.
2. **Fix `respeaker.py` now (independent of the rest).** Replace the linear
   `MovingAverage` with a circular mean (`tracker.circular_mean_deg`), so a
   person at 0° no longer reads as 180° while the angle crosses 359→0.
3. **`sound_localization.py` node.** Wrap `src/analysis/localizer.py`
   (pure numpy; copy into `speech/spatial/` with `geometry, doa, vad,
   distance, tracker, stft`). Parameters mirror `configs/local_small.yaml`
   (`audio`, `vad`, `tracker`, `distance`) + `azimuth_offset_deg`/`mirror`
   from the hardware calibration + the distance model JSON path in
   `assets/`. Two VAD options: (a) internal pitch VAD on the raw mics
   (default; same logic as `voice_detection.py`), (b) subscribe to
   `/hri/voice_activity`. Read firmware `RT60` once a minute via the existing
   `Tuning` and feed it as the room covariate. Publish:
   - `speaker_position` (custom `SpeakerPosition.msg`: header, azimuth_deg,
     range_m, confidence, is_speech, f0_hz), and
   - a `geometry_msgs/PointStamped` for RViz / tf consumers.
   Timer-free: one estimate per 16 ms frame, publish every 4 frames.
4. **`spatial_filter.py` node.** MVDR per STFT frame on the multichannel
   topic, steered at the latest `speaker_position`, noise covariance updated
   when `is_speech` is false. Output mono int16 chunks of the same size as
   today's `/hri/rawAudioChunk` so `noise_cancellation.py` needs only a
   parameter change. Param `ENABLE_SPATIAL_FILTER` (default false) — off =
   exact current behaviour. Latency budget: one STFT frame (32 ms) + overlap.
5. **Launch.** Add both nodes to `devices_launch.py` inside the existing
   `if SpeechApiUtils.respeaker_available():` block.
6. **Consumers.**
   - Head/base turns toward the talker on `/hri/voice_activity` rising edge
     (HRI ↔ navigation/manipulation interface).
   - `voice_detection.py`: optional "same direction" check next to the MFCC
     speaker lock — reject voices more than X° from the locked talker.
   - Receptionist / GPSR "find person who is talking": angle + coarse range
     as a prior for vision.

## Acceptance criteria (from the hardware session)
- DOA: LOO median error of ours ≤ firmware's, and ≤ 10° at 1–3 m.
- Range: LOO MAE < predict-mean baseline in the competition-like room;
  otherwise publish range with low confidence / only near/mid/far buckets.
- Spatial filter: audible suppression of a second talker in
  `make beamform` outputs; STT word error rate on two-talker clips (via
  `hear_streaming`) not worse than ch0 + DF. Real-time on the robot CPU.

## Risks
- **Firmware/channel mapping.** Mic order and orientation of ch1–4 vs the
  ODAS layout must be confirmed (`make calibrate` fits offset and mirror).
- **ch0 vs raw levels.** Raw mics are not AGC'd; any threshold copied from
  home2 must be re-tuned (see `05_decisions.md`).
- **Robot self-noise.** Motors/fans on the robot are near-field and
  directional; the silence model handles stationary ones, not bursts.
- **Own voice.** The robot's TTS is loud at the array; gate localization off
  while `say.py` is speaking (ch5 playback reference can detect it).
