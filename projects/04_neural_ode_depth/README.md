# Project 4 — Neural-ODE Depth in Transformers

## Question
Does treating transformer depth as a Neural-ODE solver budget improve the
latency/robustness trade-off vs. discrete depth?

## Quick start
```bash
source ../../.venv/bin/activate
make test
make all       # train + sweep + figures + paper
```
Local config trains a ~56k-param model on a synthetic copy task in ~90 s
on CPU. GPU config scales to a 256-dim sort task.

## Files
- `src/model.py` — ODE-block transformer
- `src/data/synth_seq.py` — copy / sort task generator
- `src/experiments/train_node.py` — training + solver sweep
- `src/figures/plot_node.py` — figures + macros
- `paper/main.pdf` — two-column paper
