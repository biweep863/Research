"""Evaluate quantized models on the symbolic math task and compute logit MSE / KL vs FP32."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import numpy as np, torch, yaml
from tqdm import tqdm

from utils.seeds import set_seed
from utils.model_loader import load_hf
from src.data.load_math import load_all
from src.experiments.quantize_models import clone_and_quantize


@torch.no_grad()
def greedy_answer(model, tok, prompt, device, max_new=6):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()


def extract_number(s: str):
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else None


@torch.no_grad()
def logit_distance(model_ref, model_q, tok, items, device, n_samples, max_length):
    mse_sum = 0.0
    kl_sum = 0.0
    n = 0
    for it in items[:n_samples]:
        enc = tok(it.prompt, return_tensors="pt", truncation=True,
                  max_length=max_length).to(device)
        a = model_ref(**enc).logits[0, -1].float()
        b = model_q(**enc).logits[0, -1].float()
        mse_sum += float(((a - b) ** 2).mean().item())
        # KL(p || q)
        p = torch.softmax(a, dim=-1).clamp_min(1e-12)
        q = torch.softmax(b, dim=-1).clamp_min(1e-12)
        kl_sum += float((p * (p.log() - q.log())).sum().item())
        n += 1
    return mse_sum / max(1, n), kl_sum / max(1, n)


def accuracy(model, tok, items, device):
    correct = 0
    for it in items:
        out = greedy_answer(model, tok, it.prompt, device)
        pred = extract_number(out)
        if pred is not None and pred == it.answer:
            correct += 1
    return correct / max(1, len(items))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0))

    items = load_all(cfg, seed=cfg.get("seed", 0))
    print(f"[math] loaded {len(items)} items")

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    for m in cfg["models"]:
        lm = load_hf(m["name"], device=cfg.get("device", "auto"), dtype=cfg.get("dtype"))
        rows = []
        # reference fp32 accuracy on items (cap by mse_eval n_samples for cost on local)
        ref_items = items[: cfg["mse_eval"]["n_samples"] * 2]
        baseline_acc = accuracy(lm.model, lm.tokenizer, ref_items, lm.device)
        print(f"[math] {m['short']} fp32 baseline acc {baseline_acc:.3f}")
        for bits in cfg["bit_widths"]:
            print(f"[math]  bits={bits}")
            qm = clone_and_quantize(lm.model, bits)
            acc = accuracy(qm, lm.tokenizer, ref_items, lm.device)
            mse, kl = logit_distance(lm.model, qm, lm.tokenizer, ref_items, lm.device,
                                     n_samples=cfg["mse_eval"]["n_samples"],
                                     max_length=cfg["mse_eval"]["max_length"])
            rows.append({"bits": bits, "acc": acc, "logit_mse": mse, "logit_kl": kl})
            del qm
        out = out_root / f"math_eval_{m['short']}.json"
        out.write_text(json.dumps({"baseline_acc": baseline_acc, "rows": rows}, indent=2))
        print(f"[math] wrote {out}")


if __name__ == "__main__":
    main()
