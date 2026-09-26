"""For each (prompt, true_cont, false_cont) item:
- forward the model on prompt+true_cont and prompt+false_cont
- compute persistent-homology summaries on every layer's attention matrix
- compare total persistence and betti curves between truthful and hallucinated
"""
from __future__ import annotations
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
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
from src.analysis.topology import attention_distance, summarise
from src.data.truthful_fallback import fallback
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def attentions(model, tok, text, input_device, max_tokens):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(input_device)
    out = model(**enc, output_attentions=True)
    n_tokens = enc["input_ids"].shape[1]
    return [a[0].float().cpu().numpy() for a in out.attentions], n_tokens


def _resolve_pairs(cfg, tokenizer):
    pairs = cfg.get("truthful_pairs") or []
    if pairs:
        return pairs
    src = cfg.get("pair_source")
    if src:
        from src.data.truthfulqa_loader import load_pairs
        out_root = Path(cfg.get("output_dir", "results"))
        cache = out_root.parent / "data" / f"truthfulqa_pairs_{cfg['_current_short']}.json"
        return load_pairs(src, tokenizer, cfg.get("seed", 0), cache)
    return fallback()


def _load_model(m: dict, dt: torch.dtype):
    """Loads a model per its config entry. Plain models use .to(device); models
    with device_map/quantization use HF's own placement (no .to() call)."""
    quant = m.get("quantization", "none")
    quant_cfg = None
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig
        if quant == "4bit":
            quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    kwargs = dict(torch_dtype=dt, attn_implementation="eager")
    if "device_map" in m:
        kwargs["device_map"] = m["device_map"]
    if "max_memory" in m:
        kwargs["max_memory"] = m["max_memory"]
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg

    model = AutoModelForCausalLM.from_pretrained(m["name"], **kwargs)
    model.eval()
    placed = "device_map" in m or quant_cfg is not None
    return model, placed


def _topology_row(args):
    D, thresholds, max_dim, meta = args
    out = summarise(D, thresholds, max_dim)
    out.update(meta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N pairs (smoke test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 0), deterministic=cfg.get("deterministic", False))

    out_root = Path(cfg.get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.0, 1.0, cfg["filtration"]["thresholds"])
    max_dim = cfg["filtration"]["max_dimension"]
    n_workers = cfg.get("tda", {}).get("n_workers", 1)
    drop_first = cfg.get("graph", {}).get("drop_first_token_variant", False)

    for m in cfg["models"]:
        cfg["_current_short"] = m["short"]
        dev = best_device(cfg.get("device", "auto"))
        dt = best_dtype(dev, cfg.get("dtype"))
        tok = AutoTokenizer.from_pretrained(m["name"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model, placed = _load_model(m, dt)
        if not placed:
            model.to(dev)
        input_device = dev if not placed else model.get_input_embeddings().weight.device

        pairs = _resolve_pairs(cfg, tok)
        if args.limit:
            pairs = pairs[: args.limit]
        print(f"[tda] {m['short']}: {len(pairs)} pairs")

        jobs = []
        for i, p in enumerate(tqdm(pairs, desc=f"{m['short']} forward")):
            for kind, cont in [("truthful", p["true_cont"]),
                               ("hallucinated", p["false_cont"])]:
                text = p["prompt"] + cont
                attns, n_tokens = attentions(model, tok, text, input_device,
                                              cfg["corpus"]["max_tokens"])
                for l, A in enumerate(attns):
                    D = attention_distance(A)
                    meta = {"pair_idx": i, "kind": kind, "layer": l,
                            "n_tokens": n_tokens, "thresholds": thresholds.tolist()}
                    jobs.append((D, thresholds, max_dim, dict(meta)))
                    if drop_first and D.shape[0] > 1:
                        D_ns = D[1:, 1:]
                        meta_ns = dict(meta)
                        meta_ns["nosink"] = True
                        jobs.append((D_ns, thresholds, max_dim, meta_ns))

        rows = []
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for r in tqdm(ex.map(_topology_row, jobs, chunksize=8),
                              total=len(jobs), desc=f"{m['short']} ripser"):
                    rows.append(r)
        else:
            for j in tqdm(jobs, desc=f"{m['short']} ripser"):
                rows.append(_topology_row(j))

        # merge the no-sink variant back into its base row as extra fields
        merged = {}
        for r in rows:
            key = (r["pair_idx"], r["kind"], r["layer"])
            if r.pop("nosink", False):
                base = merged.setdefault(key, {})
                base["total_persistence_H0_nosink"] = r.get("total_persistence_H0")
                base["total_persistence_H1_nosink"] = r.get("total_persistence_H1")
            else:
                base = merged.setdefault(key, {})
                base.update(r)
        rows = list(merged.values())

        quant = m.get("quantization", "none")
        suffix = f"_{quant}" if quant not in ("none", None) else ""
        out = out_root / f"tda_{m['short']}{suffix}.json"

        def _default(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            raise TypeError(o)
        out.write_text(json.dumps(rows, indent=2, default=_default))
        print(f"[tda] wrote {out}")

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
