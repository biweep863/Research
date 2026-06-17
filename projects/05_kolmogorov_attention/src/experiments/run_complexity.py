"""For each prompt: compute Kolmogorov-proxy complexity and per-layer
attention-density metrics on a HuggingFace causal LM.
Fit a regression: density ~ complexity, per layer."""
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
from utils.model_loader import load_hf
from src.data.prompts import generate
from src.analysis.complexity import (
    gzip_complexity, compression_ratio, density_summary,
)


@torch.no_grad()
def collect_attention(model, tok, text, device, max_tokens):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    out = model(**enc, output_attentions=True)
    # tuple of (n_layers,) tensors with shape (1, n_heads, q, k)
    return [a[0].cpu().numpy() for a in out.attentions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    items = generate(cfg, seed=cfg.get("seed", 0))
    print(f"[komp] generated {len(items)} prompts")

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from utils.model_loader import best_device, best_dtype
    from types import SimpleNamespace

    for m in cfg["models"]:
        # force eager attention so out.attentions is populated
        dev = best_device(cfg.get("device", "auto"))
        dt = best_dtype(dev, cfg.get("dtype"))
        tok = AutoTokenizer.from_pretrained(m["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            m["name"], torch_dtype=dt, attn_implementation="eager")
        model.to(dev); model.eval()
        lm = SimpleNamespace(model=model, tokenizer=tok, device=dev, dtype=dt)
        rows = []
        max_tokens = cfg["corpus"]["max_tokens"]
        for it in tqdm(items, desc=m["short"]):
            text = it["text"]
            K = gzip_complexity(text)
            ratio = compression_ratio(text)
            attns = collect_attention(lm.model, lm.tokenizer, text, lm.device, max_tokens)
            for l, a in enumerate(attns):
                d = density_summary(a)
                rows.append({
                    "family": it["family"],
                    "kolmogorov_proxy": K,
                    "compression_ratio": ratio,
                    "layer": l,
                    **d,
                })
        out = out_root / f"complexity_{m['short']}.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"[komp] wrote {out}")


if __name__ == "__main__":
    main()
