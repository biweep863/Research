"""Math evaluation datasets.

- Synthetic symbolic regression: arithmetic problems with deterministic ground
  truth (always available, offline).
- GSM8K and Hendrycks MATH (subsets): loaded via `datasets` if available.
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List


@dataclass
class MathItem:
    prompt: str
    answer: str
    source: str


def synthetic(cfg, seed=0) -> List[MathItem]:
    rng = random.Random(seed)
    n = int(cfg["dataset"].get("symbolic_n", 80))
    M = int(cfg["dataset"].get("max_value", 100))
    out = []
    ops = [("+", lambda a,b: a+b), ("-", lambda a,b: a-b), ("*", lambda a,b: a*b)]
    for _ in range(n):
        a = rng.randint(1, M); b = rng.randint(1, M)
        sym, fn = rng.choice(ops)
        out.append(MathItem(
            prompt=f"Q: What is {a} {sym} {b}?\nA:",
            answer=str(fn(a, b)),
            source="symbolic"))
    return out


def try_gsm8k(cfg) -> List[MathItem]:
    n = int(cfg["dataset"].get("gsm8k_n", 0))
    if n <= 0:
        return []
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=f"test[:{n}]")
        out = []
        for ex in ds:
            ans = ex["answer"].split("####")[-1].strip()
            out.append(MathItem(
                prompt=f"Question: {ex['question']}\nAnswer:",
                answer=ans, source="gsm8k"))
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[load_math] gsm8k unavailable ({e}); skipping")
        return []


def load_all(cfg, seed=0):
    return synthetic(cfg, seed=seed) + try_gsm8k(cfg)
