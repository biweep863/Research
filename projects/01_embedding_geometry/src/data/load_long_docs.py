"""Load long documents for context-length sweeps.

Falls back to a built-in synthetic corpus (concatenated ArXiv-style abstracts) if
no internet/HF cache is available, so the local pipeline is always runnable.
"""
from __future__ import annotations
from pathlib import Path
from typing import List


_SEED_TEXT = (
    "Abstract. We study the geometry of contextual embeddings produced by autoregressive "
    "transformer language models. Recent work has shown that as the input length grows, "
    "the average pairwise cosine similarity between hidden states drifts away from zero, "
    "a phenomenon known as anisotropy. Building on Ethayarajh (2019) and Cai et al. (2021), "
    "we formalize a sequence of intrinsic-dimensional and curvature-based diagnostics that "
    "characterise how the activation manifold evolves with context length. "
    "We propose two complementary estimators: a TwoNN-based intrinsic dimension and a local-PCA "
    "curvature ratio. Across a sweep of context lengths spanning four orders of magnitude on "
    "a GPT-2 backbone, we find clear evidence of a soft transition in the middle layers near "
    "the four-thousand token mark, where intrinsic dimension drops and anisotropy spikes. "
    "We discuss implications for retrieval-augmented generation, long-context fine-tuning, and "
    "the design of attention masks. The proposed metrics generalise to instruction-tuned "
    "models and open-weight large language models. We release code and configurations.\n\n"
    "Introduction. Long-context language models have become widespread. However, the empirical "
    "observation that downstream quality degrades super-linearly with context length is poorly "
    "understood from a geometric standpoint. In this work we study the manifold structure of "
    "hidden representations and ask whether a phase transition can be detected. Our contribution "
    "is threefold: (i) we extend the anisotropy measurement of Ethayarajh to long inputs; "
    "(ii) we propose an intrinsic-dimension estimator robust to high-dimensional activations; and "
    "(iii) we relate both measurements to a discrete curvature proxy obtained from local PCA. "
    "Together these diagnostics provide a falsifiable account of representation collapse in the "
    "long-context regime.\n\n"
)


def load_documents(cfg: dict) -> List[str]:
    src = cfg.get("corpus", {}).get("source", "arxiv_local")
    n = int(cfg.get("num_documents", 8))
    if src == "hf_arxiv":
        try:
            from datasets import load_dataset
            ds = load_dataset(cfg["corpus"]["hf_dataset"], split=cfg["corpus"]["hf_split"])
            field = "article" if "article" in ds.column_names else ds.column_names[0]
            return [str(ds[i][field]) for i in range(min(n, len(ds)))]
        except Exception as e:  # noqa: BLE001
            print(f"[load_long_docs] HF load failed ({e}); using seed corpus")
    # synthetic fallback: repeat the seed text enough times to support long ctx
    blob = _SEED_TEXT * 64
    return [blob for _ in range(n)]
