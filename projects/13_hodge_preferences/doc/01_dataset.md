# 01 · Data

## Synthetic preferences (`src/data/synthetic.py`)
- **Items.** n items with centred features x_i ~ N(0, I_d). Local run: n = 40, d = 8.
- **Comparison graph.** Erdős–Rényi with density p, plus a random spanning path so the graph is connected. Local run: p = 0.2, giving ~190 edges and dimensions ≈ 39 / 110 / 40 (grad / curl / harm).
- **Truth.** g*(x_i, x_j) = β_g [s(x_j) − s(x_i)] + β_c tanh(x_iᵀ R x_j).
  - s is a random smooth utility (tanh net plus a linear term).
  - R is a random skew matrix.
  - With centred features the skew term is (almost) divergence-free, so it lives in cycle space.
  - β = (1, 2) gives ≈ 59 / 30 / 11 % grad / curl / harm energy on training edges; the energies are measured by projection.
- **Held-out pairs.** 30 % of the non-edges. Their targets are defined because the truth is a function of features.
- **Noise.** Gaussian noise on the flow (σ ∈ {0, 0.5, 1, 2}), or Bernoulli votes from sigmoid(g*) with a fraction ε of votes flipped (ε ∈ {0, …, 0.4}; 30 votes per edge in `run_noise`, 200 in `run_dynamics`).

## Chatbot Arena (`src/data/arena.py`)
- **Source.** `lmarena-ai/arena-human-preference-55k` on Hugging Face (Apache-2.0, public). `lmsys/chatbot_arena_conversations` is gated, so we did not use it.
- **Size.** 57 477 votes between 64 LLMs (20 064 A, 19 652 B, 17 761 ties).
- **Cache.** `data/arena_counts.npz` (125 KB) holds vote-level (i, j, outcome) with ties = 0.5; the raw CSV is git-ignored.
- **Analysis graph.** Pairs with ≥ 20 votes: 63 models, 640 edges, 52 200 votes. Dimensions are 62 grad / 576 curl / 2 harm.
- **Split.** 20 % of votes per pair are held out (7 126 decisive held-out votes).

## Configs
`configs/local_small.yaml` runs in ~13 min on a laptop CPU (the 5-minute template target is exceeded; noise and Arena sweeps dominate). `configs/gpu_full.yaml` adds n = 80, d up to 64, depth 3, width up to 4096, 7 α values, 10 seeds, 20 000 steps and 2 000 bootstrap draws.
