"""Chatbot Arena human preference votes (lmarena-ai/arena-human-preference-55k,
Apache-2.0; Chiang et al. 2024).  57k votes between 64 LLMs; ties are kept as
half a win for each side.  We only need model-level counts, cached in
data/arena_counts.npz (a few KB, checked in); the raw 184 MB CSV is read once
from $ARENA_CSV or data/raw/train.csv if present.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv"


def _raw(project_root: Path) -> Path | None:
    for c in [os.environ.get("ARENA_CSV"), project_root / "data" / "raw" / "train.csv"]:
        if c and Path(c).exists():
            return Path(c)
    return None


def build_cache(csv: Path, out: Path) -> None:
    import pandas as pd
    d = pd.read_csv(csv, usecols=["model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"])
    names = sorted(set(d.model_a) | set(d.model_b))
    ix = {m: k for k, m in enumerate(names)}
    a = d.model_a.map(ix).to_numpy(); b = d.model_b.map(ix).to_numpy()
    wa = d.winner_model_a.to_numpy(); wb = d.winner_model_b.to_numpy(); t = d.winner_tie.to_numpy()
    # Vote-level arrays (i<j orientation, outcome = 1 if j wins, 0.5 tie, 0 if i wins).
    i = np.minimum(a, b); j = np.maximum(a, b)
    j_won = np.where(b == j, wb, wa)
    outcome = np.where(t == 1, 0.5, j_won.astype(float))
    np.savez_compressed(out, names=np.array(names), i=i.astype(np.int16), j=j.astype(np.int16),
                        outcome=outcome.astype(np.float32))


def load_arena(project_root: Path):
    cache = project_root / "data" / "arena_counts.npz"
    if not cache.exists():
        raw = _raw(project_root)
        if raw is None:
            return None
        build_cache(raw, cache)
    z = np.load(cache, allow_pickle=False)
    return {"names": list(z["names"]), "i": z["i"].astype(int), "j": z["j"].astype(int), "outcome": z["outcome"].astype(float)}


def edge_flow(i, j, outcome, n, min_votes: int = 20, prior: float = 0.5):
    """Aggregate votes to edges: log-odds of j over i with a +prior smoothing."""
    key = i * n + j
    uniq, inv = np.unique(key, return_inverse=True)
    wins = np.bincount(inv, weights=outcome); tot = np.bincount(inv).astype(float)
    keep = tot >= min_votes
    edges = np.stack([uniq[keep] // n, uniq[keep] % n], 1)
    y = np.log((wins[keep] + prior) / (tot[keep] - wins[keep] + prior))
    return edges, y, tot[keep], wins[keep]
