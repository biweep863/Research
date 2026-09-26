# Project 14 — Bigger populations for rugged landscapes: does population genetics explain genetic-algorithm sizing?

## Question
Genetic algorithms borrow their metaphor from evolution but rarely borrow
population genetics' own quantitative theory of when selection stops
dominating over chance. Does a GA on a rugged fitness landscape need a
population above some critical size $N_c$ to reliably find the optimum, and
does the classic population-genetics mechanism (a shrinking per-mutation
selection coefficient, via Kimura's fixation formula) explain why $N_c$
grows with ruggedness?

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/14_drift_selection
make test    # 15 tests (<3 s)
make run     # threshold sweep + effective-size check -> results/*.json (~1.5 min)
make figures # 3 figures + paper/macros.tex
make paper   # compile paper/main.pdf
```

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
Genome length 16→20 bits, 4→10 epistasis levels, a finer/wider population
grid (up to N=4096), 5→30 independent landscape draws per K, 100→1000
replicates per cell. The whole GA is one vectorised torch implementation
(population × replicate as a single batched tensor), so scaling is a
`device: cuda` config change, not a code change.

## Files
- `src/landscape/nk.py` — NK fitness landscape (Kauffman & Levin 1987), vectorised, exact brute-force optimum
- `src/ga/simple_ga.py` — batched-replicate GA (tournament selection, uniform crossover, bit-flip mutation, no elitism)
- `src/analysis/effective_size.py` — temporal Ne estimator, DFE-based selection coefficient, Kimura fixation probability, logistic threshold fit
- `src/experiments/run_{threshold_sweep,effective_size}.py` — write `results/*.json`
- `src/figures/plot_drift.py` — 3 figures + macros
- `paper/main.tex` — two-column paper
- `doc/` — overview, data, methodology, experiments log, results, decisions

## Headline
Population size needed to reliably find the optimum grows sharply with
landscape ruggedness — ~8785x from $K=0$ to $K=9$ at GPU scale ($N_c$: 3.7
→ 32,506), first seen as a 175x increase over a 4-point local pilot. But
the textbook population-genetics story for *why* — that epistasis shrinks
the selection coefficient of individual mutations, letting drift dominate
— is directly falsified: the measured selection coefficient $s(K)$ *grows*
monotonically with $K$ (0.034→0.115 across all 10 GPU-scale points), so
Kimura's fixation formula predicts $N_c$ should *fall*, not rise. The
population-size threshold is real; the classic single-locus mechanism for
it is not the explanation, and the falsification is cleaner (fully
monotonic) with more data, not weaker. See `doc/04_results.md`.
