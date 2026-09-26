# 05 · Decisions
- **ripser over gudhi/giotto-tda.** ripser has the cleanest distance-matrix
  API and is fast for small dense matrices.
- **Distance d = 1 - max(A, A^T).** Several alternatives (1 - A^T A,
  -log A) were considered; this one preserves the [0,1] range and is
  symmetric by construction.
- **H0 over H1 at this scale.** Short prompts give degenerate 1-complexes;
  H0 is the only stable signal at the local pilot scale.
- **N=10 deliberately.** Pilot's job is effect-size estimation for power
  analysis, not significance.

## GPU-scale run (post-pilot)

- **Pre-registered primary analysis, before running `gpu_full.yaml`:** a
  **paired** $t$-test per layer on $H_0$ total persistence
  (truthful − hallucinated, paired by `pair_idx`, since both share a
  prompt), with Holm correction across layers, restricted to pairs with
  *exactly* matched continuation token length. The pilot's own unpaired
  Welch test is kept as a secondary/legacy check for continuity, not as
  the basis for any significance claim. Secondary checks: Wilcoxon
  signed-rank, H0 normalized by token count, and a graph variant that
  drops the BOS attention-sink token. See `doc/04_results.md` for why:
  the unpaired design's own power analysis needs ~14k pairs for the
  pilot's effect size, while the paired design needs only ~200–400.
- **TruthfulQA forced continuations, not a free-generation + detector
  pipeline.** `gpu_full.yaml`'s original comment proposed sourcing pairs
  from "TruthfulQA + an offline hallucination detector." That was
  dropped: detector-labelled free generations would have noisy labels
  and unmatched lengths, undermining exactly the surface-form control
  the pilot's design depends on. Instead, `src/data/truthfulqa_loader.py`
  maps TruthfulQA's own `best_answer` → truthful continuation and the
  closest-length `incorrect_answers` entry → hallucinated continuation,
  forced onto a shared `"Q: ...\nA:"` prompt — the same forced-continuation
  design as the pilot, just sourced from a real dataset instead of 10
  hand-written examples.
- **One pair per question.** Keeps pairs independent for the paired test
  (no two pairs share underlying content).
- **Exact length match (`max_len_diff_tokens=0`) as primary.** 509/817
  TruthfulQA questions (GPT-2 tokenizer) and 490/817 (Qwen3-8B tokenizer)
  are exactly matched — comfortably above the ~400-pair target, so the
  looser `max_len_diff_tokens=1/2` tolerances were not needed.
- **Model substitution: Qwen3-8B for Meta-Llama-3-8B, then both.**
  Meta-Llama-3-8B was the intended scaled model, but its HF gated-access
  request was still pending when the run needed to start. Per this
  project's own documented fallback order, **Qwen/Qwen3-8B** (Apache-2.0,
  ungated, comparable size, and — unlike Qwen2.5 — the current generation
  at the time of this run) was substituted so the run wasn't blocked on
  an external approval of unknown duration. Access to Llama-3-8B was
  approved partway through this work; a same-pairs Llama-3-8B run was
  added afterward as a further replication check rather than a
  replacement, so the comparison across three models (GPT-2, Qwen3-8B,
  Llama-3-8B) is reported rather than any one of them alone.
- **bf16 + `device_map=auto` CPU offload, not quantization, as the
  primary loading strategy.** The 12GB-VRAM GPU can't hold an 8B model's
  weights in bf16 (~16GB) entirely on-device, but 32GB system RAM makes
  exact-precision offload feasible without accuracy loss from
  quantization. 8-bit/4-bit quantization is documented as a fallback
  (see `configs/gpu_full.yaml`) but was not needed in practice.
