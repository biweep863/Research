from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "shared"))

from utils.model_loader import best_device  # noqa: E402
from utils.seeds import set_seed  # noqa: E402

from ..data.doc_generator import make_doc  # noqa: E402
from ..data.renderers import render_all  # noqa: E402


def setup(desc: str):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="configs/local_small.yaml")
    ap.add_argument("--models", nargs="*", help="subset of model shorts to run")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    set_seed(cfg.get("seed", 0))
    out = ROOT / cfg.get("results_dir", "results")
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in cfg["models"] if not args.models or m["short"] in args.models]
    return cfg, out, models


def corpus(cfg):
    """n_sections may be a list: doc i gets n_sections[i % len] (a document-size sweep)."""
    kw = dict(cfg["corpus"]["doc"])
    sizes = kw.pop("n_sections")
    sizes = sizes if isinstance(sizes, list) else [sizes]
    return [make_doc(cfg["seed"] * 1000 + i, n_sections=sizes[i % len(sizes)], **kw)
            for i in range(cfg["corpus"]["n_docs"])]


def rendered(cfg, docs):
    """Render (cached on disk as JSON) every read format for every doc."""
    work = ROOT / cfg["work_dir"]
    cache = work / "rendered.json"
    key = {"seed": cfg["seed"], **cfg["corpus"]}
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        if c["key"] == key:
            return c["docs"]
    out = {d.doc_id: render_all(d, work / "files") for d in docs}
    cache.write_text(json.dumps({"key": key, "docs": out}, ensure_ascii=False), encoding="utf-8")
    return out


def write(path: Path, obj) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=1, default=conv, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------- models
class LM:
    """A chat model with timed prefill and greedy generation (prefix KV reuse)."""

    def __init__(self, spec: dict, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.spec = spec
        self.device = best_device(device)
        self.tok = AutoTokenizer.from_pretrained(spec["name"])
        kw = {"dtype": torch.float16 if spec.get("dtype", "fp16") == "fp16" else torch.bfloat16,
              "attn_implementation": "sdpa"}
        if spec.get("quant") == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=kw["dtype"])
            kw["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(spec["name"], **kw)
        if spec.get("quant") != "4bit":
            self.model.to(self.device)
        self.model.eval()
        self.eos = [i for i in [self.tok.eos_token_id] + spec.get("extra_eos_ids", []) if i is not None]

    def chat_ids(self, messages: list[dict]) -> list[int]:
        kw = {"enable_thinking": False} if self.spec.get("no_think") else {}
        s = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def prefill(self, ids: list[int]):
        """Run the prompt once; returns (cache, seconds)."""
        from transformers import DynamicCache
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        cache = DynamicCache()
        self._sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            self.model(input_ids=x, past_key_values=cache, use_cache=True)
        self._sync()
        return cache, time.perf_counter() - t0

    def generate(self, ids: list[int], max_new_tokens: int, prefix_cache=None) -> tuple[str, int, float, bool]:
        """Greedy decode. Returns (text, new tokens, seconds, hit the token limit)."""
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        kw = {}
        if prefix_cache is not None:
            kw["past_key_values"] = copy.deepcopy(prefix_cache)
        self._sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None, top_k=None,
                                      eos_token_id=self.eos, pad_token_id=self.tok.pad_token_id or self.eos[0],
                                      **kw)
        self._sync()
        dt = time.perf_counter() - t0
        new = out[0, len(ids):].tolist()
        hit = len(new) >= max_new_tokens and (not new or new[-1] not in self.eos)
        return self.tok.decode(new, skip_special_tokens=True), len(new), dt, hit

    def free(self):
        del self.model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def common_prefix(seqs: list[list[int]]) -> int:
    n = min(len(s) for s in seqs)
    for i in range(n):
        if any(s[i] != seqs[0][i] for s in seqs[1:]):
            return i
    return n
