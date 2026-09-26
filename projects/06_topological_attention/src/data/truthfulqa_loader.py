"""Builds (prompt, true_cont, false_cont) pairs from TruthfulQA.

Mirrors the pilot's matched-surface-form design (doc/01_dataset.md): the
true and false continuation are chosen to have the same number of tokens
under the model's own tokenizer, so any topological signal is not just a
length artefact. One pair is emitted per question (independence across
pairs), and results are cached to disk so later runs are offline and
reproducible.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

_NO_COMMENT = "i have no comment"


def _clean(s: str) -> str:
    return s.strip()


def _is_usable(s: str) -> bool:
    s = _clean(s).rstrip(".!? ").lower()
    return bool(s) and s != _NO_COMMENT


def _format_cont(ans: str) -> str:
    ans = _clean(ans)
    if not ans.endswith((".", "!", "?")):
        ans += "."
    return " " + ans


def _cont_len_tokens(tok, prompt: str, cont: str) -> int:
    base = len(tok(prompt, add_special_tokens=True)["input_ids"])
    full = len(tok(prompt + cont, add_special_tokens=True)["input_ids"])
    return full - base


def build_pair(question: str, best_answer: str, correct_answers: list[str],
               incorrect_answers: list[str], tok, max_len_diff: int) -> dict | None:
    """Pick the (true, false) continuation pair with the smallest token-length
    difference for this question. Returns None if no pair is within tolerance."""
    prompt = f"Q: {question}\nA:"
    trues = [best_answer] + [a for a in correct_answers if a != best_answer]
    trues = [a for a in trues if _is_usable(a)]
    falses = [a for a in incorrect_answers if _is_usable(a)]
    if not trues or not falses:
        return None

    best = None  # (diff, true_rank, true_cont, false_cont, len_true, len_false)
    for ti, t in enumerate(trues):
        tc = _format_cont(t)
        tlen = _cont_len_tokens(tok, prompt, tc)
        for f in falses:
            fc = _format_cont(f)
            if _clean(tc).lower() == _clean(fc).lower():
                continue
            flen = _cont_len_tokens(tok, prompt, fc)
            diff = abs(tlen - flen)
            key = (diff, ti)
            if best is None or key < (best[0], best[1]):
                best = (diff, ti, tc, fc, tlen, flen)

    if best is None or best[0] > max_len_diff:
        return None
    diff, ti, tc, fc, tlen, flen = best
    return {"prompt": prompt, "true_cont": tc, "false_cont": fc,
            "len_true": tlen, "len_false": flen, "len_diff": diff}


def load_pairs(src_cfg: dict, tokenizer, seed: int, cache_path: Path) -> list[dict]:
    """Load (or build+cache) TruthfulQA pairs matched by continuation length.

    src_cfg keys: hf_id, config, split, n_pairs (-1 = all eligible),
    max_len_diff_tokens.
    """
    cache_path = Path(cache_path)
    max_diff = int(src_cfg.get("max_len_diff_tokens", 0))
    n_pairs = int(src_cfg.get("n_pairs", -1))

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("max_len_diff_tokens") == max_diff and cached.get("tokenizer") == tokenizer.name_or_path:
            pairs = cached["pairs"]
            return _sample(pairs, n_pairs, seed)

    from datasets import load_dataset
    ds = load_dataset(src_cfg.get("hf_id", "truthfulqa/truthful_qa"),
                       src_cfg.get("config", "generation"),
                       split=src_cfg.get("split", "validation"))

    pairs = []
    for q_idx, row in enumerate(ds):
        pair = build_pair(row["question"], row["best_answer"],
                           row["correct_answers"], row["incorrect_answers"],
                           tokenizer, max_diff)
        if pair is None:
            continue
        pair["q_idx"] = q_idx
        pair["category"] = row.get("category", "")
        pairs.append(pair)

    print(f"[truthfulqa_loader] {len(pairs)}/{len(ds)} questions eligible "
          f"at max_len_diff_tokens={max_diff}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "max_len_diff_tokens": max_diff,
        "tokenizer": tokenizer.name_or_path,
        "n_eligible": len(pairs),
        "pairs": pairs,
    }, indent=2))

    return _sample(pairs, n_pairs, seed)


def _sample(pairs: list[dict], n_pairs: int, seed: int) -> list[dict]:
    if n_pairs is None or n_pairs < 0 or n_pairs >= len(pairs):
        return pairs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=n_pairs, replace=False)
    idx.sort()
    return [pairs[i] for i in idx]
