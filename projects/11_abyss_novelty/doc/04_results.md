# 04 · Results

Headline numbers (local config; all emitted as macros in `paper/macros.tex`).

## Physical sensing budget
| modality | reach |
|---|---|
| ocean-colour imagery (1/Kd, type I) | 45 m |
| ideal two-way optics, type I / III | 576 / 127 m |
| bioluminescence camera, in situ | 36 m |
| radar 10 GHz skin depth | 2.5 mm |
| radio 1 MHz skin depth | 25 cm |
| ELF 3 Hz, 60 dB margin | 1.0 km |
| eDNA plume (k = 0.05 /h) | 396 m, 20 h memory |
| sound: fish chorus 130 dB @ 300 Hz | 115 m |
| sound: click 210 dB @ 20 kHz | 12 km |
| sound: baleen 180 dB @ 50 Hz | 23 km spherical / 121 km hybrid |

Only low-frequency sound crosses the abyss; quiet animals (100 dB) are audible
within ~3 m.

## Open-set novelty, synthetic (12 known + 5 unknown, 40 clips each, SNR ~22 dB)
| score | AUROC |
|---|---|
| kNN (k=5) | **0.88** (CI 0.85–0.91), FPR@95 = 53 % |
| Mahalanobis | **0.88** |
| energy | 0.75 |
| max-softmax | 0.70 |
| autoencoder | 0.56 |
| PCA + logistic | 0.49 |

Known-class accuracy 0.77. Range ceiling 10 → 300 km (SNR 34 → 16 dB):
Mahalanobis 0.92 → 0.86, kNN 0.91 → 0.83, max-softmax 0.92 → 0.68. Four known
species: Mahalanobis 0.93; 15 clips/species: 0.87 (max-softmax 0.51).

## Open-set novelty, real Watkins (796 clips, 29 species, 6 held out, 3 draws)
Known accuracy 0.56. AUROC: max-softmax **0.69 ± 0.07**, energy 0.67, PCA
0.61, kNN 0.59, Mahalanobis 0.49, AE 0.42. Cluster ARI 0.30 mean. The
embedding is the bottleneck.

## From flagged clips to species (kNN score, 95th-percentile flag)
| U | flagged | precision | recall | k̂ | k_true | ARI |
|---|---|---|---|---|---|---|
| 2 | 50 | 0.84 | 0.53 | 4 | 2 | 0.98 |
| 3 | 78 | 0.90 | 0.58 | 4 | 3 | 0.95 |
| 5 | 183 | 0.96 | 0.88 | 2 | 5 | 0.31 |
| 8 | 277 | 0.97 | 0.84 | 3 | 8 | 0.43 |

## How many remain, where to look (300 species, log-series α = 40)
| hours | seen | Chao1 | P(next new) |
|---|---|---|---|
| 10 | 22 | 47 ± 21 | 19.4 % |
| 100 | 64 | 95 ± 18 | 3.8 % |
| 1000 | 125 | 160 ± 17 | 0.5 % |
| 3000 | 157 | 191 ± 16 | 0.2 % |

Allocation of 300 h over 20 sites (seamount 300 vs plain 60 species): greedy
on the satellite prior vs uniform = −1 % (q = 0.5), +1 % (0.7), **+13 %**
(0.9), +17 % (1.0).

## Interpretation
The abyss can only be heard, sampled or visited; the satellite chooses the
site. Given an embedding that separates species, distance-based open-set
scores flag the unknown well and group a few new species almost perfectly;
obtaining that embedding for real calls needs pretraining beyond what a
deep-sea catalogue provides. And whatever the model, rare-species statistics
set the pace of discovery. See `paper/main.pdf`.

## GPU-scale results (full Watkins dataset, pretrained backbone, larger sweeps)

### Open-set novelty, synthetic (40 known + 15 unknown, up to 200 clips, 18-point sweep)
AUROC ranges 0.72–0.97 across the sweep (base: known-class acc 0.92, best
score Mahalanobis 0.86); the qualitative ordering (embedding-geometry scores
> confidence scores > autoencoder/PCA) from the local pilot holds at scale.
Full sweep (18 runs) completed in ~17 minutes on GPU.

### Open-set novelty, real Watkins — the CNN-vs-pretrained-backbone comparison
Full dataset: 1,668 clips, 31 species, 8 held out, 10 holdout draws.

| | from-scratch CNN (300 epochs) | AST embedding (frozen, probe only) |
|---|---|---|
| Known-class accuracy | 0.70 | **0.88** |
| max-softmax AUROC | 0.66 ± 0.08 | **0.83 ± 0.02** |
| energy AUROC | 0.60 ± 0.09 | **0.84 ± 0.03** |
| kNN AUROC | 0.61 ± 0.05 | **0.73 ± 0.04** |
| Mahalanobis AUROC | 0.61 ± 0.06 | **0.74 ± 0.07** |
| Best score | msp, 0.66 | energy, **0.84** |

Paired per-draw delta on the best score: **+0.125 ± 0.041, positive in
10/10 draws** (range +0.047 to +0.197). This confirms the local pilot's own
diagnosis — "the bottleneck is the embedding, not the scores" — and closes
most of the gap to the synthetic-data ceiling (AUROC 0.88) without any
fine-tuning: the backbone is used frozen, as a drop-in embedding swap.
`ae` and `pca_msp` are identical between the two rows by construction (both
scores are computed from the mel features, not from either embedding —
documented in `doc/05_decisions.md`).

### Cluster and survey at scale
U=2: ARI 0.91; U=3: 0.52; U=5: 0.94; U=8: 0.76 (compare to the local pilot's
0.98/0.95/0.31/0.43 — the qualitative pattern of "small U clusters near-
perfectly, mid-range U is noisier" persists, but the exact ARI per U is not
stable across the 40× larger species/clip counts, consistent with a single
clustering replicate per U rather than a converged estimate).

1000-species survey: seen 220/1000 at 1000 sensor-hours, Chao1 275 ± 21,
P(next detection is new) falls from 44% at 1h to <0.1% at 10,000h+. Greedy
allocation on the satellite habitat prior ranges from **−3% at q=0.34**
(worse than uniform — a low-quality prior can actively hurt) to +13% at
q≥0.9, extending the local pilot's monotone q=0.5→1.0 trend down into the
regime where the prior is actively misleading.
