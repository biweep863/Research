"""Whisper (faster-whisper, the STT home2 runs) as the downstream judge."""

from __future__ import annotations

import os

import numpy as np

_MODEL = None


def _dll_path():
    # CTranslate2 needs cuBLAS 12 / cuDNN 9: the CUDA torch wheel ships them
    if os.name == "nt":
        import torch

        os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))


def model(name: str = "small", device: str = "cuda"):
    global _MODEL
    if _MODEL is None or _MODEL[0] != name:
        _dll_path()
        from faster_whisper import WhisperModel

        ct = "float16" if device == "cuda" else "int8"
        _MODEL = (name, WhisperModel(name, device=device, compute_type=ct))
    return _MODEL[1]


def transcribe(x: np.ndarray, name: str = "small", language: str = "es", device: str = "cuda") -> str:
    """16 kHz mono float -> text. Greedy (beam 1), no VAD filter, no context:
    the same settings for every condition, so differences come from the audio."""
    x = np.asarray(x, np.float32)
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak * 0.5
    segs, _ = model(name, device).transcribe(
        x, language=language, beam_size=1, condition_on_previous_text=False, vad_filter=False
    )
    return " ".join(s.text.strip() for s in segs).strip()
