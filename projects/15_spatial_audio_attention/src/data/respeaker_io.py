"""ReSpeaker USB v2.0 I/O: 6-ch capture, WAV replay, firmware DOA/VAD over USB.

Channel layout of the 6-ch firmware (same as home2 ``audio_capturer.py``,
which only keeps ch0): ch0 = on-chip processed/beamformed, ch1..4 = raw mics,
ch5 = playback reference.
"""

from __future__ import annotations

import json
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np

INT16_SCALE = 32768.0
RESPEAKER_VID, RESPEAKER_PID = 0x2886, 0x0018


# --------------------------------------------------------------------- devices
def list_input_devices() -> list[tuple[int, str, int]]:
    import sounddevice as sd

    return [(i, d["name"], d["max_input_channels"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0]


def find_device(name: str = "ReSpeaker", channels: int = 6) -> int | None:
    """Like home2 ``SpeechApiUtils.getIndexByNameAndChannels`` (substring + channels)."""
    for i, dev_name, ch in list_input_devices():
        if name.lower() in dev_name.lower() and ch >= channels:
            return i
    return None


class LiveSource:
    """Non-blocking 6-ch capture. ``read()`` returns (C, n) float32 or None."""

    def __init__(self, device: int | str | None, channels: int = 6, fs: int = 16000, block: int = 512):
        import sounddevice as sd

        self.q: queue.Queue = queue.Queue(maxsize=200)
        self.overflows = 0

        def cb(indata, frames, t, status):
            if status:
                self.overflows += 1
            try:
                self.q.put_nowait(indata.copy())
            except queue.Full:
                pass

        self.stream = sd.InputStream(device=device, channels=channels, samplerate=fs,
                                     dtype="int16", blocksize=block, callback=cb)
        self.fs, self.channels = fs, channels

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop()
        self.stream.close()

    def read(self):
        chunks = []
        while True:
            try:
                chunks.append(self.q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return None
        return (np.concatenate(chunks, axis=0).T.astype(np.float32)) / INT16_SCALE


class WavSource:
    """Replays a multichannel WAV in (scaled) real time, or as fast as possible."""

    def __init__(self, path: str | Path, speed: float = 1.0):
        self.data, self.fs = load_wav(path)
        self.channels = self.data.shape[0]
        self.speed = speed
        self.pos = 0
        self.t0 = None

    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *a):
        pass

    @property
    def done(self):
        return self.pos >= self.data.shape[1]

    def read(self):
        if self.done:
            return None
        if self.speed <= 0:
            end = self.data.shape[1]
        else:
            end = min(self.data.shape[1], int((time.monotonic() - self.t0) * self.fs * self.speed))
        if end <= self.pos:
            return None
        out = self.data[:, self.pos : end]
        self.pos = end
        return out


def select_mics(block: np.ndarray, mic_channels) -> np.ndarray:
    """6-ch firmware block -> the 4 raw mics. A 4-ch block is passed through."""
    if block.shape[0] == len(mic_channels):
        return block
    return block[list(mic_channels)]


# ------------------------------------------------------------------------- wav
def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """-> (C, T) float in [-1, 1], fs."""
    try:
        import soundfile as sf

        x, fs = sf.read(str(path), always_2d=True, dtype="float32")
        return x.T, fs
    except ImportError:  # pragma: no cover
        from scipy.io import wavfile

        fs, x = wavfile.read(str(path))
        x = x.astype(np.float32) / (INT16_SCALE if x.dtype == np.int16 else 1.0)
        return (x[:, None] if x.ndim == 1 else x).T, fs


def save_wav(path: str | Path, x: np.ndarray, fs: int) -> None:
    """(C, T) or (T,) float -> int16 WAV."""
    from scipy.io import wavfile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.atleast_2d(x)
    pcm = np.clip(x.T * INT16_SCALE, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), fs, pcm if pcm.shape[1] > 1 else pcm[:, 0])


def load_labels(wav_path: str | Path) -> dict:
    p = Path(wav_path).with_suffix(".json")
    return json.loads(p.read_text()) if p.exists() else {}


# --------------------------------------------------------- firmware over USB
# Minimal read path vendored from home2 hri/packages/speech/speech/tuning.py
_PARAMS = {
    "DOAANGLE": (21, 0, "int"),
    "VOICEACTIVITY": (19, 32, "int"),
    "SPEECHDETECTED": (19, 22, "int"),
    "RT60": (18, 26, "float"),
}


class FirmwareTuning:
    def __init__(self):
        import usb.core
        import usb.util

        self.usb = usb
        self.dev = usb.core.find(idVendor=RESPEAKER_VID, idProduct=RESPEAKER_PID)
        if self.dev is None:
            raise RuntimeError("ReSpeaker USB control interface not found")

    def read(self, name: str):
        pid, offset, typ = _PARAMS[name]
        cmd = 0x80 | offset | (0x40 if typ == "int" else 0)
        u = self.usb.util
        resp = self.dev.ctrl_transfer(
            u.CTRL_IN | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE, 0, cmd, pid, 8, 100000
        )
        a, b = struct.unpack(b"ii", resp.tobytes())
        return a if typ == "int" else a * (2.0**b)


class FirmwarePoller:
    """Background thread polling firmware DOA / VAD / RT60 (None if unavailable).

    On Windows pyusb needs a libusb driver for the control interface (Zadig);
    on the robot (Linux) it works like home2 ``respeaker.py``.
    """

    def __init__(self, period_s: float = 0.1):
        self.doa = self.vad = self.rt60 = None
        self.error = None
        self.log: list[tuple[float, int, int]] = []
        try:
            self.tuning = FirmwareTuning()
        except Exception as e:  # noqa: BLE001 - optional hardware path
            self.tuning, self.error = None, str(e)
            return
        self._stop = threading.Event()
        self.period = period_s
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        t0 = time.monotonic()
        k = 0
        while not self._stop.is_set():
            try:
                self.doa = self.tuning.read("DOAANGLE")
                self.vad = self.tuning.read("VOICEACTIVITY")
                if k % 20 == 0:
                    self.rt60 = self.tuning.read("RT60")
                self.log.append((time.monotonic() - t0, self.doa, self.vad))
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
            k += 1
            time.sleep(self.period)

    def stop(self):
        if self.tuning:
            self._stop.set()
