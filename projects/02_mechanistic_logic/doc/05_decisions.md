# 05 · Decisions

- **Manual hooks vs TransformerLens.** Chose pure HF forward hooks for the
  local pipeline to avoid an extra dependency; TransformerLens drop-in is a
  follow-up.
- **Zero ablation vs mean ablation.** Zero ablation is closer to a
  causal-effect intervention and avoids leaking statistics from other
  inputs into the patched head. Mean ablation is exposed as a config flag
  (`ablation.baseline: mean`).
- **Why exhaustive enumeration of 144 heads.** GPT-2-small is small enough
  that the full grid runs in ~2.5 minutes; this avoids any selection bias
  that would come from a screening heuristic.
- **Probe model.** Logistic regression with L2 only. More expressive probes
  (MLPs) inflate accuracy spuriously on small training sets.

## GPU-scale additions

- **Five templates were listed in `gpu_full.yaml` but never implemented.**
  The config specified 10 templates (adding `syllogism_celarent`,
  `syllogism_darii`, `syllogism_ferio`, `xor_chain`, `not_chain` to the 5
  local ones), but `synth_logic.py` only ever implemented the 5 — running
  the GPU config would `KeyError` immediately. Implemented all 5 (English
  + conlang): celarent/darii/ferio follow the real categorical-syllogism
  moods (universal-negative + universal/particular-affirmative premises);
  xor_chain and not_chain follow the existing and/or templates' plain
  declarative style. The conlang versions approximate categorical moods as
  signed implication chains (matching how `cl_syllogism_barbara` already
  simplifies a syllogism to `a→b, b→c ⊢ a→c`) rather than modelling true
  quantifiers, which the invented conlang's minimal grammar has no way to
  express — a structural simplification relative to English, noted rather
  than hidden.
- **GPT-NeoX/Pythia branch in the ablation hook.** `get_attn_modules()`
  only recognised `model.transformer.h` (GPT-2's module path); Pythia is
  GPT-NeoX, with attention at `model.gpt_neox.layers[i].attention` and no
  `num_attention_heads` instance attribute (only on `.config`). Without
  this branch, `run_head_ablation.py` silently skipped ablation entirely
  ("no attention modules detected") instead of erroring — easy to miss.
  Verified against the tiny `EleutherAI/pythia-70m` before running the
  full 1024-head sweep on Pythia-2.8B.
