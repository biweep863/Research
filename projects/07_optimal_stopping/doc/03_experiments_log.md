# 03 · Experiments Log

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-07-13 | `local_small.yaml` | `python -m src.experiments.run_montecarlo --config configs/local_small.yaml` | OK; ~95 s. Secretary curves (n=10/100/1000), parking DP vs naive-37% across p, cost variants. Output `results/montecarlo.json`. |
| 2026-07-13 | `local_small.yaml` | `python -m src.experiments.run_agents --config configs/local_small.yaml` | OK; ~57 s. Q-learning + REINFORCE over 5 seeds. Output `results/agents.json`. |
| 2026-07-13 | — | `python -m src.figures.plot_stopping ...` | OK; 5 PDF figures + 20 macros. |
| 2026-07-13 | — | `latexmk -pdf main.tex` | OK; two-column `paper/main.pdf`, refs resolved. |

## Observations
- **37% is asymptotic.** Optimal skip fraction: 0.300 (n=10) → 0.370 (n=100)
  → 0.368 (n=1000). For small menus the cutoff and success probability are
  both above 1/e.
- **Naive 37% on parking is catastrophic and worsens with occupancy:** regret
  ~0% at p=0.3 but ~1602% at p=0.9. When spots are usually free you should hold
  out for the destination, not grab the first free spot after a fixed cutoff.
- **Search cost pulls the cutoff down:** optimal x* falls from 1/e (c=0) to
  0.130 at c=0.005 (n=100).
- **Q-learning recovers the optima:** secretary greedy-policy regret ~1.4% at
  n=100; parking exact (0% regret) at p=0.9, within ~17% in the sparse p=0.3
  regime where the acceptance boundary is a single hard-to-estimate step.
- **REINFORCE** finds the right secretary regime but with high variance
  (~18% regret at n=100) and cannot represent the non-stationary parking
  optimum — a policy-class limitation, documented on purpose.
- Early runs mis-summarised the parking agent by extracting a single "distance
  threshold"; the optimum is per-step, so we switched to scoring the agent's
  full greedy policy exactly (see `05_decisions.md`).
