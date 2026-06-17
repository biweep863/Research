"""For each (prompt, true_cont, false_cont) item:
- forward the model on prompt+true_cont and prompt+false_cont
- compute persistent-homology summaries on every layer's attention matrix
- compare total persistence and betti curves between truthful and hallucinated
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.seeds import set_seed
from utils.model_loader import best_device, best_dtype
from src.analysis.topology import attention_distance, persistence, betti_curve
from transformers import AutoModelForCausalLM, AutoTokenizer
from types import SimpleNamespace


@torch.no_grad()
def attentions(model, tok, text, device, max_tokens):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    out = model(**enc, output_attentions=True)
    return [a[0].cpu().numpy() for a in out.attentions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    pairs = cfg["truthful_pairs"]
    if not pairs:
        # offline fallback if config has none (gpu config does this)
        from src.data.truthful_fallback import fallback
        pairs = fallback()
    print(f"[tda] {len(pairs)} pairs")

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.0, 1.0, cfg["filtration"]["thresholds"])
    max_dim = cfg["filtration"]["max_dimension"]

    for m in cfg["models"]:
        dev = best_device(cfg.get("device", "auto"))
        dt = best_dtype(dev, cfg.get("dtype"))
        tok = AutoTokenizer.from_pretrained(m["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            m["name"], torch_dtype=dt, attn_implementation="eager")
        model.to(dev); model.eval()

        rows = []
        for i, p in enumerate(tqdm(pairs, desc=m["short"])):
            for kind, cont in [("truthful", p["true_cont"]),
                               ("hallucinated", p["false_cont"])]:
                text = p["prompt"] + cont
                attns = attentions(model, tok, text, dev,
                                   cfg["corpus"]["max_tokens"])
                for l, A in enumerate(attns):
                    D = attention_distance(A)
                    pers = persistence(D, max_dim=max_dim)
                    betti = betti_curve(D, thresholds, max_dim=max_dim)
                    rows.append({
                        "pair_idx": i,
                        "kind": kind,
                        "layer": l,
                        **pers,
                        **betti,
                        "thresholds": thresholds.tolist(),
                    })
        out = out_root / f"tda_{m['short']}.json"
        def _default(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            raise TypeError(o)
        out.write_text(json.dumps(rows, indent=2, default=_default))
        print(f"[tda] wrote {out}")


if __name__ == "__main__":
    main()
