# 02 · Methodology

## Pipeline
```
OSM tiles ──► data/osm_raw.json ──► corridor.py (inventory) ──► network.py
                                                                   │  per direction: cells, capacities,
                                                                   │  ramps (feeders), exit splits β
            ┌──────────────────────────────────────────────────────┤
            ▼                       ▼                              ▼
   analysis/queues.py        sim/ctm.py (CTM)             sim/des.py (DES)
   tandem ρ, Kingman         peak dynamics, spillback     validates closed forms
            └───────────┬───────────┴──────────────┬───────────────┘
                        ▼                          ▼
          experiments/run_*.py ──► results/*.json ──► figures/plot_corridor.py
                                                      ──► paper/figures/*.pdf + paper/macros.tex ──► main.pdf
```

## Capacity
Q_i = s0 · (ℓ_exp + ℓ_lat · g_i) · (1 − f · n_acc,i), s0 = 1900 veh/h/lane (HCM).
Feeder capacity: signal s0(1 − g − 0.1) per lane; ramp 1600; give-way 700.

## Demand
q = lanes · r_class · τ (τ = 1 terminal, 0.3 crossing at base). Exits: β_c =
min(κ · attr_c / F_c, 0.5), κ by bisection so Σ exits = Σ side inflow.
Peak: trapezoid, 45 % off-peak → 1.0 over 30 min, plateau 2 h, 5 h horizon.

## Closed forms (`analysis/queues.py`, `capacity.py`, `fluid.py`)
- M/M/1: L = ρ/(1−ρ). M/D/1: Lq = ρ²/(2(1−ρ)). Erlang C via Erlang-B recursion.
- Kingman: Wq ≈ ρ/(1−ρ) · (ca² + cs²)/2 · 1/μ.
- Tandem: λ_k = min(λ_{k−1}, Q_{k−1})(1 − β_{k−1}) + a_k; ρ_k = λ_k / Q_k.
- Webster: d = 0.9 [C(1−g)²/(2(1−gx)) + x²/(2q(1−x))]; HCM d1 + d2 for x ≥ 1.
- Fluid: A(t), D(t) = min(A, D + μ dt); delay = ∫(A − D) dt.

## CTM (`sim/ctm.py`)
Triangular FD: vf 60 km/h, w 20 km/h, kj 150 veh/km/lane, dt 6 s (dx 100 m).
S_i = min(n_i, Q_i dt · (1 − drop·[queued])), R_i = min(s0 ℓ dt, w/vf (N − n)).
Merge: Daganzo priority p = ℓ_main/(ℓ_main + ℓ_ramps). Diverge: partial FIFO,
exits = β S (φ · pass + 1 − φ). ALINEA: r ← r + K (0.9 kc − k_down), K = 35.
Delay = Σ (n − outflow) dt + Σ ramp queues dt. Active bottleneck: k_i > kc ≥ k_{i+1}.

## Predictors and scoring (`experiments/corridor_eval.py`, `run_montecarlo.py`)
Per segment between junctions: local lane ratio, cumulative lane ratio, cumulative
feeder lanes, 1/min capacity, max ρ. Truth: segment with the largest share of time
hosting an active bottleneck in the CTM. Scores: hit rate (top segment = truth),
Spearman vs active share and vs delay (ties make Spearman weak; hit rate is primary),
max ρ ≥ 1 as a congestion classifier (accuracy, AUC).

## Statistics
Monte Carlo: log-uniform per-lane class rates, uniform turn fractions, through
demand, g/C, friction, φ, capacity drop. Bootstrap 95 % CIs (`shared/utils/stats.py`).
