"""Session takes -> aligned segments -> mixed scenes (superposition).

A scene is ``y = s + g * n`` on the 4 raw mics, where ``s`` is a speech-only
take (talker at a known position, room noise included) and ``n`` a noise-only
take (one speaker, one interferer). Because both are recorded on the same
fixed array, the mixture is what the array would have heard with both sources
on, up to the room's own noise being counted twice (it is 15-20 dB below the
interferers, doc/02). ``g`` sets the input SNR, measured on the mic average
over the speech-active frames.

Every scene starts with ``lead_s`` of noise only (the talker has not started
yet): that is the "silence model" of project 15 - the only noise statistics a
deployed system has - and the evaluation window starts after it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal

from .align import align, segment
from .noises import beep, chirp
from .wavio import load_take


@dataclass
class SpeechSeg:
    name: str
    angle: float
    dist: float
    text_id: str | None
    x: np.ndarray  # (6, T) all firmware channels, lead + utterance
    ref: np.ndarray | None  # Akane, aligned and resampled to the array rate
    lead: int  # samples of talker silence at the start
    active: np.ndarray  # (T,) bool, speech-active samples (from the clean take)


@dataclass
class NoiseSeg:
    name: str
    kind: str
    channel: str
    x: np.ndarray  # (6, T)


@dataclass
class Scene:
    sid: str
    speech: SpeechSeg
    noise: NoiseSeg
    snr_db: float
    s: np.ndarray  # (4, T) raw-mic speech component
    n: np.ndarray  # (4, T) raw-mic noise component (scaled)
    meta: dict = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.s + self.n


def _resample(x: np.ndarray, fs_in: int, fs: int) -> np.ndarray:
    g = np.gcd(int(fs_in), int(fs))
    return scipy.signal.resample_poly(x, int(fs) // g, int(fs_in) // g, axis=-1)


def _activity(x4: np.ndarray, fs: int, frame: int = 512, margin_db: float = 8.0) -> np.ndarray:
    """Energy activity on the 200 Hz high-passed mic average (hum removed, as p15)."""
    b, a = scipy.signal.butter(4, 200, "high", fs=fs)
    y = scipy.signal.lfilter(b, a, x4.mean(0))
    n = len(y) // frame
    e = 10 * np.log10(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-15)
    act = e > np.percentile(e, 15) + margin_db
    # close 150 ms gaps (between words), drop isolated frames
    k = max(1, int(0.15 * fs / frame))
    act = scipy.ndimage.binary_closing(act, np.ones(k)) if act.any() else act
    act = scipy.ndimage.binary_opening(act, np.ones(2))
    return np.repeat(act, frame)[: len(y)] if n else np.zeros(len(y), bool)



def highpass(x: np.ndarray, fs: int, hz: float) -> np.ndarray:
    """Removes the room's mains hum (raw mics: most noise power is < 100 Hz,
    project 15) - a fixed pre-filter any deployment applies; 0 disables."""
    if not hz:
        return x
    sos = scipy.signal.butter(4, hz, "high", fs=fs, output="sos")
    return scipy.signal.sosfilt(sos, x, axis=-1)


def _load(wav: Path, hp_hz: float) -> dict:
    t = load_take(wav)
    t["x"] = highpass(t["x"], t["fs"], hp_hz)
    if "ref" in t:
        t["ref"] = highpass(t["ref"], t["fs_ref"], hp_hz)
    return t


def load_speech(wav: Path, lead_s: float, max_s: float, hp_hz: float = 0.0) -> SpeechSeg:
    t = _load(wav, hp_hz)
    m, tl, fs = t["meta"], t["meta"]["timeline"], t["fs"]
    al = align(t["x"][1:5], fs, tl)
    fp = tl["fs"]
    # talker silence: between the end of the first chirp and the beep (minus chirp reverb)
    pre = segment(al, {**tl, "a": tl["chirp1"] + len(chirp(fp)), "b": tl["beep"]}, "a", "b", (0.3, -0.05))
    utt = segment(al, {**tl, "a": tl["beep"] + len(beep(fp))}, "a", "content_end", (0.1, 0.0))
    lead = min(int(lead_s * fs), pre.stop - pre.start)
    x_pre = t["x"][:, pre.stop - lead : pre.stop]
    x_utt = t["x"][:, utt]
    act = _activity(x_utt[1:5], fs)
    last = np.flatnonzero(act)
    end = min(len(act), (last[-1] if len(last) else len(act)) + int(0.5 * fs), int(max_s * fs))
    x = np.concatenate([x_pre, x_utt[:, :end]], axis=1)
    active = np.concatenate([np.zeros(lead, bool), act[:end]])
    ref = None
    if "ref" in t:
        alr = align(t["ref"], t["fs_ref"], tl)
        # map array samples -> reference samples through the shared playback clock
        n_play = (np.array([pre.stop - lead, utt.start]) - al.offset) / (al.ratio * fs / fp)
        r0, r1 = alr.to_rec(n_play)
        seg_pre = t["ref"][int(round(r0)) : int(round(r0)) + int(lead * t["fs_ref"] / fs)]
        seg_utt = t["ref"][int(round(r1)) : int(round(r1)) + int(end * t["fs_ref"] / fs)]
        ref = _resample(np.concatenate([seg_pre, seg_utt]), t["fs_ref"], fs)[: x.shape[1]]
        ref = np.pad(ref, (0, x.shape[1] - len(ref)))
    sp = m["spec"]
    return SpeechSeg(m["name"], sp["talker_angle"], sp["talker_dist"], sp.get("text_id"), x, ref, lead, active)


def load_noise(wav: Path, trim_s: float = 0.3, hp_hz: float = 0.0) -> NoiseSeg:
    t = _load(wav, hp_hz)
    m, tl, fs = t["meta"], t["meta"]["timeline"], t["fs"]
    al = align(t["x"][1:5], fs, tl)
    sl = segment(al, tl, "content", "content_end", (trim_s, -trim_s))
    sp = m["spec"]
    return NoiseSeg(m["name"], sp["noise"], sp["channel"], t["x"][:, sl])


def mic_power(x4: np.ndarray, mask: np.ndarray | None = None) -> float:
    x = x4 if mask is None else x4[:, mask]
    return float(np.mean(x**2))


def make_scene(sp: SpeechSeg, nz: NoiseSeg, snr_db: float, rng, channels=(1, 2, 3, 4)) -> Scene:
    T = sp.x.shape[1]
    if nz.x.shape[1] < T:
        raise ValueError(f"noise take {nz.name} shorter than speech {sp.name}")
    o = int(rng.integers(0, nz.x.shape[1] - T + 1))
    ch = list(channels)
    s = sp.x[ch]
    n = nz.x[ch, o : o + T]
    g = np.sqrt(mic_power(s, sp.active) / (mic_power(n, sp.active) + 1e-20) / 10 ** (snr_db / 10))
    sid = f"{sp.name}__{nz.kind}_{nz.channel}__snr{snr_db:+g}"
    return Scene(sid, sp, nz, snr_db, s, g * n, {"noise_offset": o, "gain": float(g)})


def load_session(session_dir: Path, lead_s: float = 1.5, max_s: float = 16.0, hp_hz: float = 0.0):
    session_dir = Path(session_dir)
    speech, noise = [], []
    for p in sorted(session_dir.glob("*.json")):
        m = json.loads(p.read_text())
        wav = p.with_suffix(".wav")
        k = m["spec"]["kind"]
        if k == "speech" and m["name"].startswith("speech_"):
            speech.append(load_speech(wav, lead_s, max_s, hp_hz))
        elif k == "noise" and m["name"].startswith("noise_"):
            noise.append(load_noise(wav, hp_hz=hp_hz))
    return speech, noise
