# 02 · Methodology

## Complexity proxy
For each prompt $x$:
- $C(x) := |\text{gzip}(x)|$
- $\rho(x) := C(x)/|x|$ (the headline scalar; $\rho \in (0,1]$).

Gzip is a cheap, tokenizer-independent Kolmogorov-complexity upper
bound.

## Attention-density metrics
For layer $\ell$ with attention tensor
$A^{(\ell)} \in \mathbb{R}^{H \times Q \times K}$, we collapse each
row $a$ to three scalars:
- **Entropy** $-\sum_k a_k \log a_k$
- **Gini** $\in [0, 1]$ (0 uniform, 1 Dirac)
- **Sparsity** — fraction of keys carrying 90% of the mass.

Averaged over heads and queries:
$\overline{H}_\ell = \mathbb{E}[\text{Entropy}]$, etc.

## Regression
Per layer $\ell$:
$\overline{H}_\ell(x) = \alpha_\ell + \beta_\ell\, \rho(x) + \varepsilon$
Report $R^2_\ell$, slope, intercept across the prompt corpus.

## Model
GPT-2 small (12 blocks, 12 heads, $d = 768$), run with
`attn_implementation="eager"` to expose attention tensors.

## Pipeline
```
gen_prompts.py
        │
        ▼
run_attn_sweep.py — per prompt × layer: attention metrics
        │
        ▼
compute_complexity.py — gzip ratio per prompt
        │
        ▼
fit_regression.py — R², slope per layer
        │
        ▼
plot_r2.py — figures + LaTeX macros
        │
        ▼
paper/main.pdf
```
