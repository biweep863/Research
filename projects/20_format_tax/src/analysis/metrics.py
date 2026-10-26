"""Pure metric functions: answer scoring, fidelity, density, format tax, tests.

No model I/O here; everything is unit-tested in tests/test_metrics.py.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize(s: str) -> str:
    """Lowercase, fold accents, drop thousands separators and punctuation, collapse spaces."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s.lower())
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)        # keep decimal points only
    s = re.sub(r"[^a-z0-9.\s]", " ", s)
    return " ".join(s.split())


def first_number(s: str) -> str | None:
    m = _NUM.search(s)
    return m.group(0).replace(",", "") if m else None


def _is_number(s: str) -> bool:
    return re.fullmatch(r"-?\d+(?:\.\d+)?", s.replace(",", "")) is not None


def score_answer(pred: str, gold: str, distractor: str = "", max_words: int = 15) -> bool:
    """Exact-match style scoring robust to chatty answers.

    Numeric gold: the first number in the first answer line must equal it.
    Text gold: its normalized words must appear contiguously in the answer, the
    answer must be short (no dumping the document) and must not also name the
    distractor (compare questions).
    """
    line = next((l for l in pred.strip().splitlines() if l.strip()), "")
    if _is_number(gold):
        n = first_number(line)
        return n is not None and float(n) == float(gold.replace(",", ""))
    p, g = normalize(line), normalize(gold)
    if not g or len(p.split()) > max(max_words, 3 * len(g.split())):
        return False
    hit = re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", p) is not None
    if hit and distractor:
        d = normalize(distractor).split()[0]
        hit = re.search(rf"(?<![a-z0-9]){re.escape(d)}(?![a-z0-9])", p) is None
    return hit


def fact_recall(text: str, facts: list[str]) -> float:
    """Fraction of planted fact strings that survive in `text` (normalized substring)."""
    if not facts:
        return float("nan")
    t = normalize(text)
    t_nospace = t.replace(" ", "")
    hits = 0
    for f in facts:
        g = normalize(f)
        # PDF extraction splits words ("T ortavel") and hyphenates; accept a space-free match
        hits += (g in t) or (g.replace(" ", "") in t_nospace)
    return hits / len(facts)


def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(hyp, [ref]).score


def density(content_tokens: int, total_tokens: int) -> float:
    """Share of the tokens that carry content (content = the plain-text rendering)."""
    return content_tokens / total_tokens if total_tokens else float("nan")


def time_per_valid(total_times: list[float], valid: list[bool]) -> float:
    """Expected wall time spent per valid file: all time (incl. failures) / #valid."""
    nv = sum(valid)
    return sum(total_times) / nv if nv else float("inf")


def format_tax(route_time_per_valid: float, baseline_time_per_valid: float) -> float:
    return route_time_per_valid / baseline_time_per_valid


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar test on paired binary outcomes."""
    from scipy.stats import binomtest
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    return 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)


def wilcoxon_p(a, b) -> float:
    from scipy.stats import wilcoxon
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 2 or np.all(d == 0):
        return 1.0
    return float(wilcoxon(d).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def linfit(x, y) -> tuple[float, float, float]:
    """Least-squares y = a + b x; returns (a, b, r^2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss = ((y - y.mean()) ** 2).sum()
    return float(a), float(b), float(1 - ((y - yhat) ** 2).sum() / ss) if ss else math.nan
