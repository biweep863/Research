"""One labelled take: play (chirp, cue, noise) on the PC speaker while recording
the ReSpeaker (6 ch) and the Akane reference mic.

Playback timeline (all marks stored in the JSON, in playback samples)::

    lead | chirp | gap | [pre-silence | beep] | content (D s) | gap | chirp | tail

- ``noise``   content = the interferer, nobody talks.
- ``speech``  content = silence on the speaker; the talker reads after the beep.
- ``live``    content = the interferer *and* the talker reads after the beep.
- ``silence`` content = nothing (room noise model).

The two chirps are heard by every microphone, so each recording is aligned to
the played signal (and the two USB clocks to each other) to a sample, see
``align.py``.
"""

from __future__ import annotations

import json
import queue
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd

from . import noises
from .devices import describe, find


@dataclass
class TakeSpec:
    kind: str  # noise | speech | live | silence
    seconds: float
    talker_angle: float | None = None
    talker_dist: float | None = None
    noise: str | None = None
    noise_angle: float | None = None
    noise_dist: float | None = None
    level_db: float = -30.0  # playback RMS of the interferer, dBFS
    text_id: str | None = None
    effort: str = "normal"  # soft | normal | loud
    note: str = ""
    seed: int = 0
    channel: str = "L"  # speaker that plays chirps + interferer: L | R | LR


@dataclass
class Timeline:
    fs: int
    chirp1: int = 0
    beep: int | None = None
    content: int = 0
    content_end: int = 0
    chirp2: int = 0
    end: int = 0
    extra: dict = field(default_factory=dict)


def build_playback(spec: TakeSpec, cfg: dict) -> tuple[np.ndarray, Timeline]:
    p = cfg["playback"]
    fs = p["fs"]
    ch = noises.chirp(fs) * 10 ** (p["chirp_db"] / 20)
    bp = noises.beep(fs) * 10 ** (p["beep_db"] / 20)
    z = lambda s: np.zeros(int(s * fs))  # noqa: E731
    tl = Timeline(fs=fs)
    parts = [z(p["lead_s"])]
    tl.chirp1 = len(parts[0])
    parts += [ch, z(p["gap_s"])]
    n_content = int(spec.seconds * fs)
    talk = spec.kind in ("speech", "live")
    pre = int(p["pre_silence_s"] * fs) if talk else 0
    content = np.zeros(pre + len(bp) + n_content if talk else n_content)
    if spec.kind in ("noise", "live"):
        nz = noises.make(spec.noise, len(content) / fs, fs, spec.seed)[: len(content)]
        # never clip the speaker: cap the level so the peak stays at max_peak_db
        cap = p["max_peak_db"] - 20 * np.log10(np.abs(nz).max())
        tl.extra["level_db_used"] = float(min(spec.level_db, cap))
        content += nz * 10 ** (tl.extra["level_db_used"] / 20)
    start = sum(map(len, parts))
    if talk:
        content[pre : pre + len(bp)] += bp
        tl.beep = start + pre
    tl.content = start + (pre + len(bp) if talk else 0)
    parts.append(content)
    tl.content_end = start + len(content)
    parts.append(z(p["gap_s"]))
    tl.chirp2 = sum(map(len, parts))
    parts += [ch, z(p["tail_s"])]
    x = np.concatenate(parts)
    tl.end = len(x)
    return x, tl


class _Rec:
    def __init__(self, device, channels, fs):
        self.q: queue.Queue = queue.Queue()
        self.status = 0

        def cb(indata, frames, t, status):
            if status:
                self.status += 1
            self.q.put(indata.copy())

        self.stream = sd.InputStream(device=device, channels=channels, samplerate=fs, dtype="int16", callback=cb)
        self.fs = fs

    def data(self) -> np.ndarray:
        out = []
        while not self.q.empty():
            out.append(self.q.get())
        return (np.concatenate(out).T.astype(np.float32) / 32768.0) if out else np.zeros((1, 0), np.float32)


def run_take(spec: TakeSpec, cfg: dict, out_dir: Path, name: str | None = None, verbose: bool = True) -> Path:
    d = cfg["devices"]
    arr_dev = find(d["array"], "input", d["array_channels"])
    ref_dev = find(d["reference"], "input", 1) if d.get("reference") else None
    out_dev = find(d["speaker"], "output", 2)
    play, tl = build_playback(spec, cfg)
    if np.abs(play).max() > 0.99:
        raise SystemExit("playback would clip; lower --level-db")
    if verbose:
        print("array    ", describe(arr_dev))
        if ref_dev is not None:
            print("reference", describe(ref_dev))
        print("speaker  ", describe(out_dev))

    fs_ref = int(sd.query_devices(ref_dev)["default_samplerate"]) if ref_dev is not None else None
    arr = _Rec(arr_dev, d["array_channels"], d["array_fs"])
    ref = _Rec(ref_dev, 1, fs_ref) if ref_dev is not None else None
    gains = {"L": (1.0, 0.0), "R": (0.0, 1.0), "LR": (1.0, 1.0)}[spec.channel]
    stereo = (play[:, None] * np.array(gains)[None, :]).astype(np.float32)
    pos = [0]

    def ocb(outdata, frames, t, status):
        n = min(frames, len(stereo) - pos[0])
        outdata[:] = 0
        if n > 0:
            outdata[:n] = stereo[pos[0] : pos[0] + n]
            pos[0] += n

    ostream = sd.OutputStream(device=out_dev, channels=2, samplerate=tl.fs, dtype="float32", callback=ocb)
    arr.stream.start()
    if ref:
        ref.stream.start()
    time.sleep(0.3)
    t0 = time.monotonic()
    ostream.start()
    dur = len(play) / tl.fs
    if verbose and tl.beep is not None:
        print(f"... beep in {(tl.beep) / tl.fs:.1f} s -> read; the take ends with the second chirp ({dur:.0f} s)")
    while pos[0] < len(stereo):
        time.sleep(0.05)
    time.sleep(0.4)
    ostream.stop(), ostream.close()
    arr.stream.stop(), arr.stream.close()
    if ref:
        ref.stream.stop(), ref.stream.close()
    wall = time.monotonic() - t0

    from .wavio import save_wav

    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = name or f"{stamp}_{spec.kind}"
    out_dir.mkdir(parents=True, exist_ok=True)
    x = arr.data()
    save_wav(out_dir / f"{name}.wav", x, d["array_fs"])
    save_wav(out_dir / f"{name}_play.wav", play, tl.fs)
    meta = {"name": name, "stamp": stamp, "spec": asdict(spec), "timeline": asdict(tl),
            "array": {"device": describe(arr_dev), "fs": d["array_fs"], "channels": int(x.shape[0]),
                      "overflows": arr.status},
            "speaker": describe(out_dev), "wall_s": wall}
    if ref:
        r = ref.data()
        save_wav(out_dir / f"{name}_ref.wav", r, fs_ref)
        meta["reference"] = {"device": describe(ref_dev), "fs": fs_ref, "overflows": ref.status}
    raw = x[1:5] if x.shape[0] >= 5 else x
    pk = 20 * np.log10(np.abs(raw).max() + 1e-12)
    rms = 20 * np.log10(np.sqrt(np.mean(raw**2, axis=1)) + 1e-12)
    meta["raw_peak_dbfs"], meta["raw_rms_dbfs"] = float(pk), rms.round(1).tolist()
    if x.shape[0] >= 5:
        meta["raw_mic_corr_min"] = float(np.corrcoef(raw).min())
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1))
    if verbose:
        print(f"wrote {out_dir / name}.wav  {x.shape[1] / d['array_fs']:.1f} s  raw peak {pk:.1f} dBFS  "
              f"rms {rms.round(1).tolist()}  overflows {arr.status}")
        if ref:
            print(f"      reference peak {20 * np.log10(np.abs(r).max() + 1e-12):.1f} dBFS")
        if pk > -1:
            print("WARNING: raw mics clipped")
        if x.shape[0] >= 5 and np.corrcoef(raw).min() > 0.9999:
            print("WARNING: raw channels identical -> Windows audio enhancements are on again")
    return out_dir / f"{name}.wav"
