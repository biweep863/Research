# 02 · Methodology

## Attention to distance
Per layer $\ell$, collect the head-averaged attention matrix
$G^{(\ell)} \in \mathbb{R}^{T \times T}$, symmetrise as
$G \leftarrow \max(G, G^\top)$, and convert to a metric:
$d^{(\ell)}_{ij} = 1 - G^{(\ell)}_{ij}$ with $d^{(\ell)}_{ii} = 0$.

## Vietoris–Rips persistent homology
Feed $d^{(\ell)}$ to `ripser` and compute the persistence diagram in
dimensions $0$ and $1$. Extract:
- **Total persistence** in $H_k$:
  $\sum_i (\text{death}_i - \text{birth}_i)$ over finite bars.
- **Betti curve** $\beta_k(t)$: number of bars alive at threshold
  $t \in \{0, 1/32, \dots, 1\}$.

## Statistical test
Per layer, run a Welch two-sample $t$-test on total persistence in
$H_0$ between truthful and hallucinated conditions. Report the
per-layer $p$-value as $-\log_{10} p$.

## Model
GPT-2 small (12 blocks), `attn_implementation="eager"` to expose
attention.

## Pipeline
```
pairs.json (10 curated pairs)
        │
        ▼
run_attn_capture.py — forward each (prompt+continuation), grab attention
        │
        ▼
compute_ph.py — Vietoris–Rips PH per layer (ripser)
        │
        ▼
stats.py — per-layer Welch t-test on H0 total persistence
        │
        ▼
plot_ph.py — figures + LaTeX macros
        │
        ▼
paper/main.pdf
```
