"""Signal and ASR metrics (SI-SDR as in projects 15/16; ESTOI via pystoi; WER via jiwer)."""

from __future__ import annotations

import re
import unicodedata

import numpy as np


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    t = a * ref
    return float(10 * np.log10(np.sum(t**2) / (np.sum((est - t) ** 2) + 1e-12) + 1e-12))


def snr_db(s: np.ndarray, n: np.ndarray, active: np.ndarray | None = None) -> float:
    """Speech power over its active samples / noise power over the whole window."""
    ps = np.mean(s[active] ** 2) if active is not None and active.any() else np.mean(s**2)
    return float(10 * np.log10(ps / (np.mean(n**2) + 1e-20) + 1e-20))


def estoi(est: np.ndarray, ref: np.ndarray, fs: int) -> float:
    from pystoi import stoi

    return float(stoi(ref, est, fs, extended=True))


def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9ñ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_NUM = {"17": "diecisiete", "11": "once", "9": "nueve"}


def wer(ref: str, hyp: str) -> float:
    import jiwer

    r = normalize_text(ref)
    h = normalize_text(hyp)
    for k, v in _NUM.items():
        h = re.sub(rf"\b{k}\b", v, h)
        r = re.sub(rf"\b{k}\b", v, r)
    if not r:
        return float("nan")
    return float(jiwer.wer(r, h)) if h else 1.0
