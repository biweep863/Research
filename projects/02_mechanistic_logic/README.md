# Project 2 — Mechanistic Interpretability of Logical-Symbolic Reasoning

## Question
Does a transformer carry a linearly decodable representation of logical
validity that is independent of natural-language surface form?

## Quick start
```bash
source ../../.venv/bin/activate
make test        # data tests
make probe       # per-layer logistic probes (English + conlang)
make ablate      # exhaustive per-head ablation on GPT-2
make figures     # figures + LaTeX macros
make paper       # paper/main.pdf
```

## Scaling
```bash
make all CONFIG=configs/gpu_full.yaml  # Pythia-2.8B, 10 templates, larger N
```

## Files
- `src/data/conlang.py`, `src/data/synth_logic.py` — dataset generator
- `src/experiments/run_probe.py` — per-layer probes
- `src/experiments/run_head_ablation.py` — per-head ablation
- `src/figures/plot_logic.py` — figures + macros
- `paper/main.pdf` — two-column scientific paper
- `doc/` — methodology notes & experiments log
