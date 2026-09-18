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
