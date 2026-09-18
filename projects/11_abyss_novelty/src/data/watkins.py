"""Real marine-mammal calls: the Watkins Marine Mammal Sound Database
("Best of" cuts, WHOI) in the BEANS benchmark packaging (Hagiwara et al.
2022), mirrored on the Hugging Face Hub as ``DBD-research-group/beans_watkins``
(parquet shards with wav bytes and a species label; 31 species).

Two-stage loading so the pipeline reproduces offline:
1. If parquet shards are present under ``$WATKINS_DIR`` or ``data/raw``,
   decode every wav (scipy), resample to the project rate, take the first
   ``clip_s`` seconds (calls are short), compute the log-mel, quantise to
   uint8 and write ``data/watkins_mel.npz`` (a few MB, checked in).
2. Otherwise load the cache.  If neither exists return None.
The WHOI site itself was under maintenance when this project was built.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np

from ..novelty.features import from_uint8, log_mel, resample, to_uint8


def _raw_dir(root: Path) -> Path | None:
    for cand in [os.environ.get("WATKINS_DIR"), root / "data" / "raw"]:
        if cand and Path(cand).exists() and list(Path(cand).glob("*.parquet")):
            return Path(cand)
    return None


def _build(raw: Path, audio: dict, max_per_species: int) -> dict:
    import pyarrow.parquet as pq
    from scipy.io import wavfile
    fs, clip_s = audio["fs_hz"], audio["clip_s"]
    mels, labels = [], []
    counts: dict[str, int] = {}
    for f in sorted(raw.glob("*.parquet")):
        t = pq.read_table(f, columns=["path", "label"])
        for row in t.to_pylist():
            lab = row["label"]
            if counts.get(lab, 0) >= max_per_species:
                continue
            try:
                sr, x = wavfile.read(io.BytesIO(row["path"]["bytes"]))
            except Exception:
                continue
            x = np.asarray(x, dtype=np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if np.issubdtype(np.asarray(x).dtype, np.integer) or np.abs(x).max() > 1.5:
                x = x / 32768.0
            x = resample(x, sr, fs)
            n = int(fs * clip_s)
            x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
            mels.append(to_uint8(log_mel(x, fs, audio["n_mels"], audio["n_frames"], audio["n_fft"],
                                         audio["fmin_hz"], audio["fmax_hz"])))
            labels.append(lab); counts[lab] = counts.get(lab, 0) + 1
    return {"X": np.stack(mels), "labels": np.array(labels)}


def load_watkins(root: Path, audio: dict, max_per_species: int = 60):
    cache = root / "data" / "watkins_mel.npz"
    if not cache.exists():
        raw = _raw_dir(root)
        if raw is None:
            return None
        d = _build(raw, audio, max_per_species)
        np.savez_compressed(cache, **d)
    z = np.load(cache, allow_pickle=False)
    return from_uint8(z["X"]), z["labels"].astype(str)
