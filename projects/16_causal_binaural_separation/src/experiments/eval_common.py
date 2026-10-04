"""Run a model over a fixed set and score every utterance (parallel on CPU)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.binaural import cue_errors, residual_cue_shift
from src.analysis.metrics import estoi, pesq_wb, si_sdr_binaural

METRICS = ("si_sdr", "si_sdr_i", "estoi", "pesq", "d_itd_us", "d_ild_db", "d_ild_tf_db", "d_ipd_rad",
           "res_d_itd_us", "res_d_ild_db")


def run_padded(fn, x: torch.Tensor, hop: int) -> torch.Tensor:
    """Pad to a whole number of hops, run, crop back."""
    T = x.shape[-1]
    pad = (-T) % hop
    y = fn(F.pad(x, (0, pad)) if pad else x)
    return y[..., :T]


@torch.no_grad()
def enhance_set(fn, hop: int, mix: torch.Tensor, device, batch: int = 16) -> np.ndarray:
    outs = []
    for i in range(0, mix.shape[0], batch):
        outs.append(run_padded(fn, mix[i : i + batch].to(device), hop).float().cpu())
    return torch.cat(outs).numpy()


def _score_one(args):
    est, mix, ref, fs, with_estoi = args
    r = {"si_sdr": si_sdr_binaural(est, ref)}
    r["si_sdr_i"] = r["si_sdr"] - si_sdr_binaural(mix, ref)
    r["estoi"] = estoi(est, ref, fs) if with_estoi else float("nan")
    r["pesq"] = pesq_wb(est, ref, fs)
    r.update(cue_errors(est, ref, fs))
    r.update(residual_cue_shift(est, mix, ref, fs))
    return r


def score_set(est: np.ndarray, mix: np.ndarray, ref: np.ndarray, fs: int, with_estoi: bool = True) -> dict:
    """Per-utterance metrics -> {metric: list}."""
    jobs = [(est[i], mix[i], ref[i], fs, with_estoi) for i in range(est.shape[0])]
    workers = min(8, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 8:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_score_one, jobs, chunksize=4))
    else:
        rows = [_score_one(j) for j in jobs]
    return {k: [float(r[k]) for r in rows] for k in METRICS}


def summarize(per_utt: dict) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
    from utils.stats import bootstrap_ci

    out = {}
    for k, v in per_utt.items():
        a = np.asarray(v, float)
        a = a[np.isfinite(a)]
        out[k] = list(bootstrap_ci(a)) if len(a) else [float("nan")] * 3
    return out
