# 02 · Methodology

```
configs/*.yaml ─► src/physics (closed forms) ──────────────► results/budget.json
              ─► src/sim (species → propagate → log-mel) ─┐
data/watkins_mel.npz (real log-mel) ───────────────────────┤
                                                            ▼
                           src/novelty: models (CNN, AE, PCA+LR) → scores → cluster
                           src/survey: community → estimators → allocate
                                                            │
                           src/experiments/run_* ──► results/*.json
                           src/figures/plot_abyss ─► paper/figures/*.pdf + paper/macros.tex
                           latexmk ──► paper/main.pdf
```

## Physics (`src/physics`)
- Light: E(z) = E0 e^(−Kd z); reach = ln(E0/floor)/Kd (halved for two-way);
  euphotic = ln(100)/Kd; bioluminescence: photons·A/(4πr²)·e^(−Kd r) ≥ threshold.
- Radio: δ = 1/√(π f μ0 σ); reach = δ·M/8.686 for margin M dB.
- Sound: SL − TL − (NL − DI) ≥ DT; TL = 20 log r (+ cylindrical beyond 4 km in
  hybrid mode) + Thorp α(f)·r; NL = Wenz spectrum level + 10 log(0.23 f);
  range = largest r satisfying the inequality on a log grid.
- eDNA: 1-D plume C(x) = S/(u·π·2Dx/u)·e^(−kx/u); footprint = largest x with
  C ≥ floor; memory = 1/k.

## Open-set pipeline (`src/novelty`)
- Split: 35 % of known clips held out; all unknown clips in the test set.
- CNN: 3×[conv3×3–BN–ReLU–maxpool] (16, 32, 64) → global mean → Linear(64)+ReLU
  → head; Adam 3e-3, wd 1e-4, 12 epochs (synthetic) / 50 (real), batch 32/16,
  Gaussian input noise 0.05 as augmentation.
- AE: conv encoder to 32-d latent, transposed-conv decoder; MSE; 10 epochs.
- PCA(40) + multinomial logistic regression on flattened log-mel.
- Scores: msp = 1 − max softmax; energy = −logsumexp; knn = cosine distance
  to the 5th nearest training embedding; maha = min class Mahalanobis with
  shared covariance (+1e-3 I); ae = mean squared reconstruction error;
  pca_msp = 1 − max PCA+LR probability.
- Metrics: AUROC, AUPR (unknown positive), FPR@95 % TPR; known-class accuracy;
  bootstrap CI (300 resamples) on the best score's AUROC.
- Flag + cluster: threshold = 95th percentile of known test scores on the kNN
  score; agglomerative (average linkage, cosine) on L2-normalised embeddings;
  k ∈ [2, 10] by silhouette; ARI vs true labels among flagged clips.

## Survey (`src/survey`)
- Log-series relative abundances p_k ∝ x^k/k, x = 1 − 1/α, permuted.
- Detections ~ Poisson(rate · hours · p_i · S); species accumulation = # with
  count > 0; Chao1 = S_obs + f1²/(2 f2) (bias-corrected when f2 = 0) with
  Chao 1987 variance; Good–Turing discovery probability = f1/n.
- Allocation: expected species at a site E[S(h)] = Σ_i (1 − e^(−λ_i h)) under
  the prior over habitats; greedy 10 h steps until the budget is spent;
  realised species by simulation; compared with uniform hours per site.

## Verification (`tests/test_model.py`, 10 tests, ~6 s)
Light/radio closed forms vs hand values; Thorp monotone; sonar range solves
the equation to 0.2 dB; hybrid ≥ spherical; eDNA memory and footprint order;
mel filterbank normalisation and 1 kHz peak position; uint8 round-trip; the
four call families are mutually distinct (corr < 0.9); CNN known accuracy >
0.5 and kNN/Mahalanobis AUROC > 0.7 on a tiny set; clustering recovers 4
separated groups (ARI > 0.95); Chao1 and Good–Turing on hand-computed counts;
greedy allocation with a perfect prior beats uniform.
