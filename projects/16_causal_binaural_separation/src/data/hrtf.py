"""HRTF sets: measured (SOFA SimpleFreeFieldHRIR) or the spherical-head fallback.

Head frame everywhere: x front, y left, z up. Row 0 of every HRIR pair is the
left ear (checked by ``tests/test_data.py``: a source at +90 deg is louder left).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.signal

from .synth import spherical_head_hrirs

PRA_SOFA_DIR = None  # resolved lazily from the pyroomacoustics package


@dataclass
class HrtfSet:
    name: str
    dirs: np.ndarray  # (D, 3) unit vectors, head frame
    hrir: np.ndarray  # (D, 2, L) at ``fs``
    fs: int

    def nearest(self, vecs: np.ndarray) -> np.ndarray:
        v = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
        return np.argmax(v @ self.dirs.T, axis=-1)


def _unit(az_deg, el_deg):
    a, e = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)


def sofa_path(name: str, raw_dir: Path) -> Path:
    if name.startswith("mit_kemar"):
        import pyroomacoustics

        return Path(pyroomacoustics.__file__).parent / "data" / "sofa" / f"{name}.sofa"
    return Path(raw_dir) / f"{name}.sofa"


def load_sofa(path: Path, fs: int, name: str | None = None) -> HrtfSet:
    import h5py

    with h5py.File(path, "r") as f:
        ir = np.asarray(f["Data.IR"][()], dtype=np.float64)  # (M, R, N)
        fs_in = int(np.asarray(f["Data.SamplingRate"][()]).ravel()[0])
        pos = np.asarray(f["SourcePosition"][()], dtype=np.float64)
        rec = np.asarray(f["ReceiverPosition"][()], dtype=np.float64).reshape(ir.shape[1], 3)
    if rec[0, 1] < rec[1, 1]:  # make row 0 the left (+y) ear
        ir = ir[:, ::-1]
    if fs_in != fs:
        fr = Fraction(fs, fs_in).limit_denominator(1000)
        ir = scipy.signal.resample_poly(ir, fr.numerator, fr.denominator, axis=-1)
    ir = ir / np.sqrt(np.mean(np.sum(ir**2, axis=-1)))  # unit mean energy per ear
    return HrtfSet(name or Path(path).stem, _unit(pos[:, 0], pos[:, 1]), ir.astype(np.float32), fs)


def spherical_head(fs: int) -> HrtfSet:
    az, el, h = spherical_head_hrirs(fs)
    h = h / np.sqrt(np.mean(np.sum(h**2, axis=-1)))
    return HrtfSet("spherical_head", _unit(az, el), h.astype(np.float32), fs)


def load_hrtf(name: str, fs: int, raw_dir: Path, allow_fallback: bool = True) -> HrtfSet:
    if name == "spherical_head":
        return spherical_head(fs)
    p = sofa_path(name, raw_dir)
    if p.exists():
        return load_sofa(p, fs, name)
    if not allow_fallback:
        raise FileNotFoundError(p)
    print(f"[hrtf] {p} missing -> spherical-head fallback")
    return spherical_head(fs)
