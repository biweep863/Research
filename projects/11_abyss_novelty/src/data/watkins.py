"""Real marine-mammal calls: the Watkins Marine Mammal Sound Database
("Best of" cuts, WHOI) in the BEANS benchmark packaging (Hagiwara et al.
2022), mirrored on the Hugging Face Hub as ``DBD-research-group/beans_watkins``
(parquet shards with wav bytes and a species label; 31 species).

The full dataset ships 5 parquet files: train-00000-of-00002,
train-00001-of-00002, valid, test, and train_low. train_low is excluded
here -- it is very likely a subset of train, and mixing it in would let
the same clip appear in both a training split and a held-out split,
inflating AUROC. See src/data/fetch_watkins.py.

Two-stage loading so the pipeline reproduces offline:
1. If parquet shards are present under ``$WATKINS_DIR`` or ``data/raw``,
   decode every wav (scipy), resample to the project rate, take the first
   ``clip_s`` seconds (calls are short), compute the log-mel, quantise to
   uint8 and write the cache (a few MB, checked in).
2. Otherwise load the cache.  If neither exists return None.
The WHOI site itself was under maintenance when this project was built.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np

from ..novelty.features import from_uint8, log_mel, resample, to_uint8

EXCLUDED_SHARDS = {"train_low"}


def _raw_dir(root: Path) -> Path | None:
    for cand in [os.environ.get("WATKINS_DIR"), root / "data" / "raw"]:
        if cand and Path(cand).exists() and list(Path(cand).glob("*.parquet")):
            return Path(cand)
    return None


def _shard_files(raw: Path):
    return sorted(f for f in raw.glob("*.parquet") if not any(x in f.stem for x in EXCLUDED_SHARDS))


def _decode(wav_bytes: bytes, fs: float, clip_s: float) -> np.ndarray | None:
    from scipy.io import wavfile
    try:
        sr, x = wavfile.read(io.BytesIO(wav_bytes))
    except Exception:
        return None
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(np.asarray(x).dtype, np.integer) or np.abs(x).max() > 1.5:
        x = x / 32768.0
    x = resample(x, sr, fs)
    n = int(fs * clip_s)
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _build(raw: Path, audio: dict, max_per_species: int) -> dict:
    import pyarrow.parquet as pq
    fs, clip_s = audio["fs_hz"], audio["clip_s"]
    mels, labels, keys = [], [], []
    seen_hashes: set[str] = set()
    counts: dict[str, int] = {}
    for f in _shard_files(raw):
        t = pq.read_table(f, columns=["path", "label"])
        for row in t.to_pylist():
            lab = row["label"]
            if counts.get(lab, 0) >= max_per_species:
                continue
            wav_bytes = row["path"]["bytes"]
            key = hashlib.sha1(wav_bytes).hexdigest()
            if key in seen_hashes:
                continue  # duplicate clip across shards
            x = _decode(wav_bytes, fs, clip_s)
            if x is None:
                continue
            mels.append(to_uint8(log_mel(x, fs, audio["n_mels"], audio["n_frames"], audio["n_fft"],
                                         audio["fmin_hz"], audio["fmax_hz"])))
            labels.append(lab); keys.append(key)
            seen_hashes.add(key)
            counts[lab] = counts.get(lab, 0) + 1
    return {"X": np.stack(mels), "labels": np.array(labels), "keys": np.array(keys),
            "fs": fs, "n_mels": audio["n_mels"], "n_frames": audio["n_frames"],
            "max_per_species": max_per_species}


def load_watkins(root: Path, audio: dict, max_per_species: int = 60, cache_name: str = "watkins_mel.npz"):
    cache = root / "data" / cache_name
    if not cache.exists():
        raw = _raw_dir(root)
        if raw is None:
            return None
        d = _build(raw, audio, max_per_species)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **d)
    z = np.load(cache, allow_pickle=False)
    if "fs" in z:
        meta_ok = (int(z["fs"]) == audio["fs_hz"] and int(z["n_mels"]) == audio["n_mels"]
                   and int(z["n_frames"]) == audio["n_frames"])
        if not meta_ok:
            raise ValueError(
                f"{cache} was built with different audio settings "
                f"(fs={int(z['fs'])}, n_mels={int(z['n_mels'])}, n_frames={int(z['n_frames'])}) "
                f"than the current config (fs={audio['fs_hz']}, n_mels={audio['n_mels']}, "
                f"n_frames={audio['n_frames']}). Delete the cache or point real.cache elsewhere.")
    keys = z["keys"].astype(str) if "keys" in z else None
    return from_uint8(z["X"]), z["labels"].astype(str), keys


def load_watkins_waveforms(root: Path, fs: float, max_s: float, max_per_species: int):
    """Loads raw (resampled, cropped) waveforms for a pretrained-embedding
    backbone. Uses scipy.signal.resample_poly (anti-aliased) rather than the
    linear-interpolation resample() used by the mel pipeline, since a
    frozen pretrained model is more sensitive to aliasing artefacts."""
    from scipy.signal import resample_poly
    import pyarrow.parquet as pq

    raw = _raw_dir(root)
    if raw is None:
        return None
    waves, labels, keys = [], [], []
    seen_hashes: set[str] = set()
    counts: dict[str, int] = {}
    for f in _shard_files(raw):
        t = pq.read_table(f, columns=["path", "label"])
        for row in t.to_pylist():
            lab = row["label"]
            if counts.get(lab, 0) >= max_per_species:
                continue
            wav_bytes = row["path"]["bytes"]
            key = hashlib.sha1(wav_bytes).hexdigest()
            if key in seen_hashes:
                continue
            from scipy.io import wavfile
            try:
                sr, x = wavfile.read(io.BytesIO(wav_bytes))
            except Exception:
                continue
            x = np.asarray(x, dtype=np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if np.issubdtype(np.asarray(x).dtype, np.integer) or np.abs(x).max() > 1.5:
                x = x / 32768.0
            if sr != fs:
                g = np.gcd(int(sr), int(fs))
                x = resample_poly(x, int(fs) // g, int(sr) // g)
            n = int(fs * max_s)
            x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
            waves.append(x.astype(np.float32)); labels.append(lab); keys.append(key)
            seen_hashes.add(key)
            counts[lab] = counts.get(lab, 0) + 1
    return waves, np.array(labels), np.array(keys)
