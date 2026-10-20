# 01 · Data: one recorded session (2026-10-18)

There is no public corpus recorded on this array, with this interferer
geometry, before home2's DeepFilterNet3. All data come from one guided
session (`record_session.py`, about 40 minutes), stored in `data/session/`.
That directory is git-ignored; this file and the JSON sidecars describe it.

## Hardware
| Item | What | Windows device (resolved by name) |
|---|---|---|
| Array | ReSpeaker USB Mic Array v2.0, 6-ch firmware: ch0 = chip-processed beam, ch1–4 = raw mics on a 32 mm circle, ch5 = playback | WASAPI, 6 ch, 16 kHz |
| Reference | Akane Omnia 300 USB mic, 10–20 cm from the mouth | WASAPI, 48 kHz → 16 kHz |
| Interferer | the PC's stereo desktop speakers, L or R driven alone, ~0.5 m from the array | WASAPI, 48 kHz, Windows volume 60 |

**Windows pitfall.** With Windows' "audio enhancements" on (an effect pack,
*VocaEffectPack* / Voice Clarity), every host API returns **2 identical
channels at 48 kHz** instead of the 6 raw channels. Turning off the
enhancements in Settings was not enough. The classic `mmsys.cpl` →
Advanced → "Enable audio enhancements" had to be unchecked and the device
replugged. `session.py` warns if ch1–4 come back identical.

## Geometry (array frame = the frame of `respeaker_v2_positions`)
| Source | Talker's label | Array DOA from its clean take |
|---|---|---|
| speaker L | 22.5° (talker's estimate) | 11–16° |
| speaker R | 67.5° (talker's estimate) | 254–260° |
| talker | 0, 45, …, 315° at 0.5 m (own marks, 0° = USB port) | see `results/spatial.json → src_doa` |

The two speakers are ~110° apart as seen by the array, not the 45° the
talker estimated. The array is mirrored and rotated by about 35° relative to
the talker's marks (project 15 calibration). Because the talker's marks are
approximate (about ±20°), **every separation in the results uses the DOA
measured from each source's own clean take** (high SNR, nothing else
playing), not the labels.

## Takes
| Kind | Count | Content |
|---|---|---|
| silence | 1 | 15 s room noise |
| noise | 10 | 5 interferers × {L, R}, 20 s each, talker silent |
| speech | 8 | one Spanish prompt each (t1–t8, `src/data/texts.py`), one per angle, 0.5 m |
| live | 2 | talker reads while an interferer plays (talker at 270°/L; cafeteria at 0°/L) |
| distance (project 15) | 7 | free speech at 270°: 0.4 m normal/soft, 1 m ×2/loud, 2 m normal/loud |

Every take is `chirp | gap | [2 s silence | beep] | content | gap | chirp`.
The chirps align each recording to the played signal, and the array and
reference clocks to each other (`align.py`). Measured offset jitter is below
2 ms. Clock-rate ratio is 1.00001 for the array and 0.99997–1.00009 for the
Akane.

## Interferers (`src/data/noises.py`)
| Name | Source |
|---|---|
| pink | synthetic, < 60 Hz removed |
| fan | synthetic airflow (80–1500 Hz) + 120 Hz motor harmonics + 18 Hz blade AM |
| cafeteria | DEMAND PCAFETER ch01 (real babble and dishes), from project 16's download |
| talker | one LibriSpeech dev-clean speaker (English): a TV or a second person |
| music | synthetic chord progression + kick/hat |

Playback level is −16 dBFS RMS, capped so the peak stays ≤ −1.5 dBFS. That
gives cafeteria −19.5, talker −18.5, music −17.2 and fan −16.4 dBFS.
- **At the array:** pink from L reaches about −37 dBFS RMS above 200 Hz,
  against a room floor of −59 dBFS. Speech at 0.5 m reaches −44 to −49 dBFS.
- **Room noise:** −50 dBFS RMS full band, dominated by mains hum below
  100 Hz, as project 15 found. A 100 Hz 4th-order high-pass is applied to
  every take before analysis.

**Bug found (documented, not fixed in the data).** `noises.make` draws the
cafeteria offset from the requested length, so the cafeteria *live* take
replayed a different excerpt than its noise-only take. The superposition
check therefore uses only the talker live take, where the played samples are
identical (correlation 0.999).

## License
DEMAND: CC BY-SA 3.0. LibriSpeech: CC BY 4.0. The recordings are of the
author and are not redistributed.
