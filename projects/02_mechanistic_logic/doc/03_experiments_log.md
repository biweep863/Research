# 03 · Experiments Log

| Date | Command | Outcome |
|------|---------|---------|
| 2026-06-15 | `python -m src.experiments.run_probe --config configs/local_small.yaml` | OK; 640 prompts in 15 s. probe_gpt2.json written. |
| 2026-06-15 | `python -m src.experiments.run_head_ablation --config configs/local_small.yaml --max_items 60` | OK; 144 heads × ~1 s = 2.5 min. ablate_gpt2.json written. |
| 2026-06-15 | `python -m src.figures.plot_logic --results results --out paper/figures --macros paper/macros.tex` | OK; 4 figures + 12 macros. |
| 2026-06-15 | `latexmk -pdf paper/main.tex` | OK; 412 KB PDF. |

## Headline numbers (local pilot, GPT-2)
- English probe peak: 1.000 at layer 6
- Conlang probe peak: 0.959 at layer 8
- Cross en→cl peak: 0.669
- Cross cl→en peak: 0.594
- Baseline behaviour: 51.7% (near chance — expected for un-finetuned GPT-2)
- Worst-head ablation: layer 0, head 10, ΔAcc −0.033, ΔLM +1.417
  → not a logic-specific head; ablation hurts both metrics

## GPU-scale run (2026-09-26, Pythia-2.8B, 10 templates)
| Date | Command | Outcome |
|------|---------|---------|
| 2026-09-26 | `generate()` on `configs/gpu_full.yaml`'s 10 templates | Crashed: `syllogism_celarent`/`darii`/`ferio`/`xor_chain`/`not_chain` were listed in the config but never implemented in `synth_logic.py` (only 5 of the 10 templates existed). Implemented all 5 (English + conlang), see `doc/05_decisions.md`. |
| 2026-09-26 | `run_probe.py --config configs/gpu_full.yaml` (Pythia-2.8B, 5120 items, 33 layers) | 2:39 (32 items/s on GPU, bf16 offload not needed — 2.8B fits natively). English probe peak 1.000 (layer 5), conlang peak 0.997 (layer 19). |
| 2026-09-26 | `run_head_ablation.py --config configs/gpu_full.yaml` (1024 heads = 32 layers × 32 heads, `max_items=80`) | ~37 min (2.17 s/head). Required a code fix: `get_attn_modules()` only recognised GPT-2's `transformer.h` path; added a GPT-NeoX/Pythia branch (`gpt_neox.layers[i].attention.dense`), verified against `EleutherAI/pythia-70m` before committing to the full run. |
| 2026-09-26 | `make figures` + `make paper` | 24 macros, 4-page PDF, no undefined refs. |

## GPU-scale headline numbers (Pythia-2.8B)
- English probe peak: 1.000 at layer 5; conlang peak: 0.997 at layer 19
  (both still near-perfect at 2.8B scale)
- Cross en→cl peak: **0.580**; cross cl→en peak: **0.514** — both *lower*
  than the GPT-2 pilot's 0.669/0.594, and cl→en is close to chance (0.5).
  Bigger model, weaker cross-surface transfer — the opposite of what a
  naive "scale helps generalization" story would predict.
- Baseline zero-shot logic accuracy: 0.425 (*below* chance on this 10-
  template mix — Pythia-2.8B's raw next-token preference between " valid"/
  " invalid" is not even calibrated in the right direction on average).
- Worst-head ablation: layer 14, head 14, ΔAcc −0.025, ΔLM −0.038 — both
  tiny (single-head ablation barely moves a 2.8B model's behaviour;
  corr(ΔAcc, ΔLM) across all 1024 heads ≈ 0.026, i.e. no relationship) —
  consistent with a much more distributed/redundant representation than
  GPT-2-small's, where the single worst head visibly hurt both metrics.
