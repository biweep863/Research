"""Speech (LibriSpeech) and noise (DEMAND) corpora -> in-memory int16 pools.

Downloads are idempotent and cached under ``data/raw``; decoded pools are
cached as ``.npz`` under ``data/cache``. Without network or files the loaders
fall back to the synthetic generators in :mod:`synth` and mark the pool
``synthetic=True`` so every result file records what it was trained on.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .synth import pink_noise, voiced_speech

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{subset}.tar.gz"
DEMAND_URL = "https://zenodo.org/records/1227121/files/{env}_16k.zip"
CIPIC_URL = "https://sofacoustics.org/data/database/cipic/{subject}.sofa"
SPLIT_SEED = {"train": 0, "val": 1, "test": 2, "heldout": 3}  # str hash() is salted per process


class Pool:
    """Concatenated int16 audio with segment boundaries (one entry per file)."""

    def __init__(self, audio: np.ndarray, starts: np.ndarray, lengths: np.ndarray, synthetic: bool = False):
        self.audio, self.starts, self.lengths, self.synthetic = audio, starts, lengths, synthetic

    def crop(self, n: int, rng) -> np.ndarray:
        """Random ``n``-sample float32 crop from a file at least ``n`` long (or tiled)."""
        ok = np.flatnonzero(self.lengths >= n)
        i = rng.choice(ok) if len(ok) else rng.integers(len(self.lengths))
        L = int(self.lengths[i])
        seg = self.audio[self.starts[i] : self.starts[i] + L]
        if L < n:
            seg = np.tile(seg, n // L + 1)
            L = len(seg)
        o = rng.integers(0, L - n + 1)
        return seg[o : o + n].astype(np.float32) / 32768.0

    @property
    def hours(self) -> float:
        return float(self.lengths.sum()) / 16000 / 3600


def _download(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[corpora] downloading {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
        return True
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[corpora] download failed ({e})")
        return False


def _to_pool(chunks: list[np.ndarray], synthetic=False) -> Pool:
    lengths = np.array([len(c) for c in chunks], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    return Pool(audio, starts, lengths, synthetic)


def _i16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


# ------------------------------------------------------------------ speech
def speaker_split(speakers: list[str], fracs, seed: int) -> dict[str, list[str]]:
    """Deterministic disjoint train/val/test speaker split."""
    rng = np.random.default_rng(seed)
    s = sorted(speakers)
    rng.shuffle(s)
    n = len(s)
    a = int(round(fracs[0] * n))
    b = a + int(round(fracs[1] * n))
    return {"train": s[:a], "val": s[a:b], "test": s[b:]}


def librispeech_files(subset: str, raw: Path) -> dict[str, list[tuple[str, bytes]]] | None:
    tgz = raw / f"{subset}.tar.gz"
    if not _download(LIBRISPEECH_URL.format(subset=subset), tgz):
        return None
    out: dict[str, list] = {}
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".flac"):
                spk = m.name.split("/")[-3]
                out.setdefault(spk, []).append((m.name, tf.extractfile(m).read()))
    return out


def load_speech(split: str, cfg: dict, root: Path, max_hours: float | None = None) -> Pool:
    """Speech pool for ``split`` in {train, val, test}, speaker-disjoint."""
    import soundfile as sf

    d = cfg["data"]["speech"]
    cache = root / "cache" / f"speech_{split}_{'-'.join(d[split])}.npz"
    if cache.exists():
        z = np.load(cache)
        return Pool(z["audio"], z["starts"], z["lengths"], bool(z["synthetic"]))
    chunks = []
    for subset in d[split]:
        files = librispeech_files(subset, root / "raw")
        if files is None:
            chunks = []
            break
        shared = sum(subset in d[s] for s in ("train", "val", "test")) > 1
        spk = list(files)
        if shared:
            spk = speaker_split(spk, d["speaker_split"], cfg["seed"])[split]
        for s in sorted(spk):
            for _, blob in sorted(files[s]):
                x, fs = sf.read(io.BytesIO(blob), dtype="int16")
                assert fs == cfg["fs"], fs
                chunks.append(x)
    synthetic = not chunks
    if synthetic:
        print(f"[corpora] speech '{split}': falling back to synthetic voiced speech")
        rng = np.random.default_rng(SPLIT_SEED[split])
        chunks = [_i16(0.1 * voiced_speech(4.0, cfg["fs"], f0=rng.uniform(90, 230), seed=i)) for i in range(60)]
    pool = _to_pool(chunks, synthetic)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, audio=pool.audio, starts=pool.starts, lengths=pool.lengths, synthetic=synthetic)
    return pool


# ------------------------------------------------------------------- noise
def load_noise(split: str, cfg: dict, root: Path) -> Pool:
    """DEMAND channel-1 noise. ``train``/``val``/``test`` use the training
    environments split in time (first 70 % / next 10 % / last 20 %);
    ``heldout`` uses the held-out environments in full."""
    import soundfile as sf

    d = cfg["data"]["noise"]
    envs = d["heldout_envs"] if split == "heldout" else d["envs"]
    cache = root / "cache" / f"noise_{split}_{'-'.join(envs)}.npz"
    if cache.exists():
        z = np.load(cache)
        return Pool(z["audio"], z["starts"], z["lengths"], bool(z["synthetic"]))
    chunks = []
    for env in envs:
        zp = root / "raw" / f"{env}_16k.zip"
        if not _download(DEMAND_URL.format(env=env), zp):
            chunks = []
            break
        with zipfile.ZipFile(zp) as z:
            name = next(n for n in z.namelist() if n.endswith("ch01.wav"))
            x, fs = sf.read(io.BytesIO(z.read(name)), dtype="int16")
        assert fs == cfg["fs"], fs
        n = len(x)
        bounds = {"train": (0, 0.7), "val": (0.7, 0.8), "test": (0.8, 1.0), "heldout": (0, 1.0)}[split]
        chunks.append(x[int(bounds[0] * n) : int(bounds[1] * n)])
    synthetic = not chunks
    if synthetic:
        print(f"[corpora] noise '{split}': falling back to synthetic pink noise")
        rng = np.random.default_rng(100 + SPLIT_SEED[split])
        chunks = [_i16(0.05 * pink_noise(30 * cfg["fs"], rng)) for _ in range(4)]
    pool = _to_pool(chunks, synthetic)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, audio=pool.audio, starts=pool.starts, lengths=pool.lengths, synthetic=synthetic)
    return pool


def fetch_hrtf(name: str, root: Path) -> None:
    if name.startswith("subject_"):
        _download(CIPIC_URL.format(subject=name), root / "raw" / f"{name}.sofa")
