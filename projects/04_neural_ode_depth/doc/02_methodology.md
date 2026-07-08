# 02 · Methodology

## Architecture
A single pre-norm transformer block
`f_θ = FFN ∘ LN ∘ (Attn ∘ LN + id)` serves as the vector field.

- **Continuous mode:** integrate $h(0) \mapsto h(1)$ via
  `torchdiffeq.odeint` with one of {Euler, RK4, dopri5}. NFE
  (number of function evaluations) is set by the solver budget.
- **Discrete mode:** apply $h_{k+1} = h_k + f_\theta(h_k)$ for
  $k = 0, \dots, D-1$. NFE equals $D$.

Both modes share parameters; only the unroll differs.

## Training
Fixed cheap continuous schedule (RK4, step 0.25, 4 NFE). AdamW,
lr $3\times 10^{-3}$, batch 64, 1500 steps. Next-token cross-entropy.

## Evaluation sweep
- **ODE solvers (7 configs):** Euler at {0.5, 0.25, 0.125},
  RK4 at {0.25, 0.125}, dopri5 at tolerances {1e-2, 1e-4}.
- **Discrete baselines (3 configs):** depth {2, 4, 8}.

## Metrics
- **Accuracy** on supervised positions.
- **Loss** (next-token cross-entropy; floor ≈ 1.47).
- **Latency** (wall-clock, 256-example batch).
- **NFE** (integer count of vector-field evaluations).

## Pipeline
```
gen_task.py
        │
        ▼
train_ode.py — one continuous block, cheap-schedule training
        │
        ▼
sweep_eval.py — for each (mode, budget): loss, acc, latency, NFE
        │
        ▼
results/sweep.json
        │
        ▼
plot_pareto.py — figures + LaTeX macros
        │
        ▼
paper/main.pdf
```
