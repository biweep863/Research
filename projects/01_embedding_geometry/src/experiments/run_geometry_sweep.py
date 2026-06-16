"""Run the context-length × layer geometry sweep.

Usage:
    python -m src.experiments.run_geometry_sweep --config configs/local_small.yaml
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Make `src.*` and shared utils importable when run as a script
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))                                  # project root
sys.path.insert(0, str(HERE.parent / "shared"))                # shared utils

import numpy as np
import yaml
from tqdm import tqdm

from utils.seeds import set_seed
from utils.model_loader import load_hf
from utils.hooks import hidden_states
from src.data.load_long_docs import load_documents
from src.analysis.intrinsic_dimension import twonn, effective_rank
from src.analysis.anisotropy import anisotropy
from src.analysis.curvature import local_pca_curvature


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    docs = load_documents(cfg)
    print(f"[sweep] loaded {len(docs)} documents")

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    for m in cfg["models"]:
        name, short = m["name"], m["short"]
        print(f"[sweep] loading model {name}")
        lm = load_hf(name, device=cfg.get("device", "auto"), dtype=cfg.get("dtype"))
        n_layers = lm.model.config.num_hidden_layers
        rows = []
        for L in cfg["context_lengths"]:
            print(f"[sweep]  ctx={L}")
            per_layer = {l: [] for l in range(n_layers + 1)}
            for doc in tqdm(docs, leave=False):
                hs, real_len = hidden_states(lm.model, lm.tokenizer, doc,
                                             max_length=L, device=lm.device)
                for l, h in enumerate(hs):
                    # h: (1, seq, hidden) -> (seq, hidden); cap tokens
                    x = h[0].numpy()
                    cap = min(int(cfg["tokens_for_metrics"]), x.shape[0])
                    per_layer[l].append(x[:cap])
            for l in range(n_layers + 1):
                x = np.concatenate(per_layer[l], axis=0)
                # subsample for tractability
                if x.shape[0] > 4000:
                    idx = np.random.choice(x.shape[0], 4000, replace=False)
                    x = x[idx]
                rows.append({
                    "model": short,
                    "ctx": L,
                    "layer": l,
                    "n_tokens": int(x.shape[0]),
                    "twonn": twonn(x),
                    "anisotropy": anisotropy(x, n_pairs=int(cfg["random_pairs"])),
                    "curvature": local_pca_curvature(x, k=int(cfg["neighborhood_k"])),
                    "eff_rank": effective_rank(x),
                })
        out_path = out_root / f"sweep_{short}.json"
        out_path.write_text(json.dumps(rows, indent=2))
        print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
