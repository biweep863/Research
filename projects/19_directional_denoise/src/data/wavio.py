"""WAV I/O (int16 on disk, float (C, T) in memory), same convention as project 15."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INT16_SCALE = 32768.0


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """-> (C, T) float32 in [-1, 1], fs."""
    import soundfile as sf

    x, fs = sf.read(str(path), always_2d=True, dtype="float32")
    return x.T, fs


def save_wav(path: str | Path, x: np.ndarray, fs: int) -> None:
    """(C, T) or (T,) float -> int16 WAV."""
    from scipy.io import wavfile

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.atleast_2d(x)
    pcm = np.clip(x.T * INT16_SCALE, -32768, 32767).astype(np.int16)
    wavfile.write(str(path), fs, pcm if pcm.shape[1] > 1 else pcm[:, 0])


def load_take(wav: str | Path) -> dict:
    """Array audio + labels + (if present) reference mic and played signal."""
    wav = Path(wav)
    meta = json.loads(wav.with_suffix(".json").read_text())
    x, fs = load_wav(wav)
    out = {"meta": meta, "x": x, "fs": fs}
    for key in ("ref", "play"):
        p = wav.with_name(f"{wav.stem}_{key}.wav")
        if p.exists():
            y, fy = load_wav(p)
            out[key], out[f"fs_{key}"] = y[0], fy
    return out
