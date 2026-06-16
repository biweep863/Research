# 03 · Experiments Log

| Date | Command | Outcome |
|------|---------|---------|
| 2026-06-15 | `python -m src.experiments.run_probe --config configs/local_small.yaml` | OK; 640 prompts in 15 s. probe_gpt2.json written. |
| 2026-06-15 | `python -m src.experiments.run_head_ablation --config configs/local_small.yaml --max_items 60` | OK; 144 heads × ~1 s = 2.5 min. ablate_gpt2.json written. |
| 2026-06-15 | `python -m src.figures.plot_logic --results results --out paper/figures --macros paper/macros.tex` | OK; 4 figures + 12 macros. |
| 2026-06-15 | `latexmk -pdf paper/main.tex` | OK; 412 KB PDF. |

## Headline numbers
- English probe peak: 1.000 at layer 6
- Conlang probe peak: 0.959 at layer 8
- Cross en→cl peak: 0.669
- Cross cl→en peak: 0.594
- Baseline behaviour: 51.7% (near chance — expected for un-finetuned GPT-2)
- Worst-head ablation: layer 0, head 10, ΔAcc −0.033, ΔLM +1.417
  → not a logic-specific head; ablation hurts both metrics
