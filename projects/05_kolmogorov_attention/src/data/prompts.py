"""Synthetic prompts covering a wide range of algorithmic complexity.

Kolmogorov complexity is uncomputable; we use a standard upper-bound proxy:
the size of the gzip-compressed UTF-8 representation. The wider the
compression-ratio spectrum across prompts, the cleaner the regression
against attention-density metrics.
"""
from __future__ import annotations
import random
import string


def random_uniform(n: int, rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase + string.digits, k=n))


def repeating_short(n: int, rng: random.Random) -> str:
    motif = rng.choice(["ab", "xyz", "0123"])
    return (motif * (n // len(motif) + 1))[:n]


def repeating_long(n: int, rng: random.Random) -> str:
    motif = "".join(rng.choices(string.ascii_lowercase, k=8))
    return (motif * (n // len(motif) + 1))[:n]


_ENGLISH = (
    "The quick brown fox jumps over the lazy dog. "
    "All happy families are alike; each unhappy family is unhappy in its own way. "
    "It was the best of times, it was the worst of times. "
    "Call me Ishmael. Some years ago—never mind how long precisely. ")


def english_text(n: int, rng: random.Random) -> str:
    out = ""
    while len(out) < n:
        out += _ENGLISH
    return out[:n]


def code_like(n: int, rng: random.Random) -> str:
    body = ""
    while len(body) < n:
        var = rng.choice(["x", "y", "z", "acc"])
        op = rng.choice(["+", "-", "*"])
        body += f"{var} = {var} {op} {rng.randint(1, 9)};\n"
    return body[:n]


def arithmetic(n: int, rng: random.Random) -> str:
    body = ""
    while len(body) < n:
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        body += f"{a}+{b}={a+b}\n"
    return body[:n]


def low_entropy_pattern(n: int, rng: random.Random) -> str:
    return "a" * n


FAMILIES = {
    "random_uniform":      random_uniform,
    "repeating_short":     repeating_short,
    "repeating_long":      repeating_long,
    "english_text":        english_text,
    "code_like":           code_like,
    "arithmetic":          arithmetic,
    "low_entropy_pattern": low_entropy_pattern,
}


def generate(cfg, seed: int = 0):
    rng = random.Random(seed)
    n_per = max(1, cfg["corpus"]["n_prompts"] // len(cfg["prompt_families"]))
    max_chars = cfg["corpus"]["max_tokens"] * 4  # rough char->token ratio
    items = []
    for fam in cfg["prompt_families"]:
        fn = FAMILIES[fam]
        for i in range(n_per):
            items.append({"family": fam, "text": fn(max_chars, rng)})
    rng.shuffle(items)
    return items
