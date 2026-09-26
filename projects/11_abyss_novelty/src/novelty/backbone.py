"""Frozen pretrained audio embedding, used only on the real-data path
(src/experiments/run_real.py) as an alternative to the from-scratch CNN.
See doc/05_decisions.md for why AST (MIT/ast-finetuned-audioset-10-10-0.4593)
was chosen: native to `transformers` (no torchaudio/librosa dependency --
its feature extractor falls back to a numpy fbank), not gated, small
(86.6M params), and fits easily in 12GB VRAM. Fine-tuning is out of scope
for this pass; the backbone is used purely as a frozen embedding.
"""
from __future__ import annotations

import numpy as np
import torch


def embed_clips(waves: list[np.ndarray], model_id: str, fs: int, device: str = "cpu",
                 batch: int = 16) -> np.ndarray:
    """waves: list of 1-D float32 arrays at `fs` Hz. Returns (N, D) float32
    pooled embeddings from the backbone's pooler output."""
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    embeds = []
    with torch.no_grad():
        for i in range(0, len(waves), batch):
            chunk = [np.asarray(w, dtype=np.float32) for w in waves[i:i + batch]]
            inputs = extractor(chunk, sampling_rate=fs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state.mean(dim=1)
            embeds.append(pooled.float().cpu().numpy())
    return np.concatenate(embeds, axis=0)


def load_or_build_embeddings(root, rcfg: dict, device: str = "cpu"):
    """Loads (or builds+caches) backbone embeddings for the real Watkins
    clips. Returns (E, labels, keys) or None if raw data isn't available."""
    from pathlib import Path
    from ..data.watkins import load_watkins_waveforms

    root = Path(root)
    model_id = rcfg["backbone"]
    slug = model_id.split("/")[-1].replace(".", "_")
    cache = root / "data" / f"watkins_emb_{slug}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["E"].astype(np.float32), z["labels"].astype(str), z["keys"].astype(str)

    fs = rcfg.get("backbone_fs_hz", 16000)
    max_s = rcfg.get("backbone_max_s", 10.0)
    loaded = load_watkins_waveforms(root, fs, max_s, rcfg["cache_max_per_species"])
    if loaded is None:
        return None
    waves, labels, keys = loaded
    E = embed_clips(waves, model_id, fs, device, rcfg.get("backbone_batch", 16))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, E=E.astype(np.float16), labels=labels, keys=keys)
    return E, labels, keys
