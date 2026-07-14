# Project 7 — Optimal Stopping: the 37% rule, the parking problem, and RL agents

## Question
When should you stop looking? We place the secretary problem (the "37% rule"),
the parking problem, and two variants in one MDP, and ask whether model-free
agents rediscover the optimal stopping thresholds from reward alone.

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/07_optimal_stopping
make test           # closed forms vs Monte Carlo / brute force
make run            # Monte-Carlo + RL agent experiments
make figures        # regenerate figures + paper/macros.tex
make paper          # compile paper/main.pdf
```

## Scaling up
```bash
make run     CONFIG=configs/gpu_full.yaml
make figures CONFIG=configs/gpu_full.yaml
```
Wider pool sizes, occupancies, costs, and more seeds/episodes — no code
changes; everything is gated behind the config. (CPU-only: no GPU needed.)

## Files
- `src/envs/{secretary,parking}.py` — synthetic environments + fast simulators
- `src/analysis/analytic.py` — closed forms and exact backward-induction DP
- `src/analysis/policies.py` — tabular Q-learning and REINFORCE agents
- `src/experiments/run_montecarlo.py` — closed-form + simulated curves
- `src/experiments/run_agents.py` — trains agents, scores regret vs optima
- `src/figures/plot_stopping.py` — figures + LaTeX macros
- `paper/main.tex` — two-column scientific paper
- `doc/` — step-by-step methodology, experiments log, decisions
