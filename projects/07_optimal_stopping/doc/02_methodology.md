# 02 · Methodology

## Closed forms and exact DP (`analysis/analytic.py`)

### Secretary (rank-only)
Reject-first-`r` success probability
`P(r,n) = (r/n) Σ_{i=r+1}^{n} 1/(i-1)`; asymptotically `P(x) = -x ln x`, max at
`x = 1/e`. An arbitrary per-position take-mask `S` is scored exactly by
`P = Σ_{i∈S} (1/n) Π_{j<i, j∈S} (j-1)/j`, using record independence — this lets
us evaluate a *learned* (possibly non-threshold) policy without simulation.

### Parking (backward induction)
`V_t = p·min(c_t, V_{t+1}) + (1-p)·V_{t+1}`, with `V_T = fail_cost`. Park at a
free spot iff `c_t ≤ V_{t+1}`. The threshold is per-step (not one distance):
near the end of the street you accept anything rather than fail.

### Variants
Search cost: payoff `P(win) − c·E[#inspected]`, with `E[#inspected]` in closed
form. Full information: the win ceiling rises to ≈0.58 (Gilbert–Mosteller).

## Agents (`analysis/policies.py`)
- **Tabular Q-learning** over discrete decision states (record position;
  street step), two actions, ε-greedy with linearly annealed ε and learning
  rate. The greedy policy is extracted as a mask and scored *exactly*.
- **REINFORCE** on a one-parameter sigmoid threshold; the learned scalar is the
  cutoff. Well-specified for the secretary rule; deliberately mis-specified for
  the non-stationary parking optimum.

Regret is `|achieved − optimal| / |optimal|` in success probability (secretary)
or expected cost (parking).

## Pipeline
```
envs/ (secretary, parking)
        │
        ▼
analytic.py  ── exact optima ──┐
        │                      │
run_montecarlo.py       run_agents.py  (train + score vs optima)
        │                      │
results/montecarlo.json  results/agents.json
        │                      │
        └────── plot_stopping.py ──────┘
                     │
             figures + paper/macros.tex → paper/main.pdf
```
