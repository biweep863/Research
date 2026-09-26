"""Attention-head ablation on GPT-2 / Pythia, measured against (a) logic-accuracy
and (b) language-modeling cross-entropy on a small text snippet.

We patch the output of `attn.c_proj` (GPT-2) or `dense` (Pythia) one head at a
time by zeroing the corresponding hidden-dim slice. Heads whose ablation
disproportionately hurts logic accuracy versus LM perplexity are flagged.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np, torch, yaml
from tqdm import tqdm
from utils.seeds import set_seed
from utils.model_loader import load_hf
from src.data.synth_logic import generate


def get_attn_modules(model):
    """Return list[(layer_idx, head_dim, n_heads, module)] for each transformer block.
    Works for GPT-2 (attn.c_proj) and GPT-NeoX/Pythia (attention.dense) by hooking
    the *input* to the output projection, shape (batch, seq, n_heads * head_dim)."""
    pairs = []
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for i, block in enumerate(model.transformer.h):
            n_heads = block.attn.num_heads if hasattr(block.attn, "num_heads") else block.attn.n_head
            head_dim = block.attn.head_dim if hasattr(block.attn, "head_dim") else (block.attn.embed_dim // n_heads)
            pairs.append((i, head_dim, n_heads, block.attn.c_proj))
    elif hasattr(model, "gpt_neox"):
        for i, block in enumerate(model.gpt_neox.layers):
            n_heads = block.attention.config.num_attention_heads
            head_dim = block.attention.head_size
            pairs.append((i, head_dim, n_heads, block.attention.dense))
    return pairs


@torch.no_grad()
def score_logic(model, tok, items, device, ablate=None):
    """Score binary classification by comparing log-prob of ' valid' vs ' invalid'
    at the next-token position."""
    handle = None
    if ablate is not None:
        layer_i, head_i, head_dim = ablate
        def hook(module, input, output_ignored=None):
            x = input[0].clone()
            start = head_i * head_dim
            end = start + head_dim
            x[..., start:end] = 0.0
            return (x,) + input[1:]
        pairs = get_attn_modules(model)
        handle = pairs[layer_i][3].register_forward_pre_hook(hook)
    try:
        correct = 0
        for it in items:
            enc = tok(it.prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            out = model(**enc)
            logits = out.logits[0, -1]
            # for English: compare " valid" vs " invalid"; for conlang " vir" vs " kal"
            if it.surface == "english":
                a, b = " valid", " invalid"
            else:
                a, b = " vir", " kal"
            ai = tok(a, add_special_tokens=False)["input_ids"][0]
            bi = tok(b, add_special_tokens=False)["input_ids"][0]
            pred_valid = logits[ai] > logits[bi]
            if (pred_valid and it.label == "valid") or (not pred_valid and it.label == "invalid"):
                correct += 1
        return correct / len(items)
    finally:
        if handle is not None:
            handle.remove()


@torch.no_grad()
def lm_loss(model, tok, text, device, ablate=None):
    handle = None
    if ablate is not None:
        layer_i, head_i, head_dim = ablate
        def hook(module, input, output_ignored=None):
            x = input[0].clone()
            start = head_i * head_dim
            end = start + head_dim
            x[..., start:end] = 0.0
            return (x,) + input[1:]
        pairs = get_attn_modules(model)
        handle = pairs[layer_i][3].register_forward_pre_hook(hook)
    try:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model(**enc, labels=enc["input_ids"])
        return float(out.loss.item())
    finally:
        if handle is not None:
            handle.remove()


LM_TEXT = (
    "The transformer architecture has become the dominant approach for natural "
    "language processing tasks. It relies on self-attention to model long-range "
    "dependencies between tokens in a sequence, enabling efficient parallelisation."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_items", type=int, default=80, help="cap items for ablation eval")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    items = generate(cfg, seed=cfg.get("seed", 0))[: args.max_items]
    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    for m in cfg["models"]:
        lm = load_hf(m["name"], device=cfg.get("device", "auto"), dtype=cfg.get("dtype"))
        pairs = get_attn_modules(lm.model)
        if not pairs:
            print("[ablate] no attention modules detected, skipping")
            continue
        baseline_acc = score_logic(lm.model, lm.tokenizer, items, lm.device, ablate=None)
        baseline_lm = lm_loss(lm.model, lm.tokenizer, LM_TEXT, lm.device, ablate=None)
        print(f"[ablate] baseline acc={baseline_acc:.3f} lm={baseline_lm:.3f}")

        rows = []
        for layer_i, head_dim, n_heads, _ in pairs:
            for h in tqdm(range(n_heads), desc=f"layer {layer_i}", leave=False):
                acc = score_logic(lm.model, lm.tokenizer, items, lm.device,
                                  ablate=(layer_i, h, head_dim))
                lml = lm_loss(lm.model, lm.tokenizer, LM_TEXT, lm.device,
                              ablate=(layer_i, h, head_dim))
                rows.append({"layer": layer_i, "head": h,
                             "delta_acc": acc - baseline_acc,
                             "delta_lm": lml - baseline_lm})
        out = out_root / f"ablate_{m['short']}.json"
        out.write_text(json.dumps({"baseline_acc": baseline_acc,
                                   "baseline_lm": baseline_lm,
                                   "rows": rows}, indent=2))
        print(f"[ablate] wrote {out}")


if __name__ == "__main__":
    main()
