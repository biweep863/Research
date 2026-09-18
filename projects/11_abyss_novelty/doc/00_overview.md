# 00 · Overview

**Question.** Could satellites or other remote sensors reveal unknown animals
below the abyssal zone, and could an AI recognise a creature nobody has
catalogued? (ideas/ideas.txt: "detectar criaturas debajo del agua a una
distancia después del abyss zone con IA".)

**Physics first.** Seawater ends every satellite modality within a few hundred
metres: ocean colour integrates ~45 m, an ideal two-way optical link reaches
~0.6 km in the clearest water, radar skin depth is millimetres and 3 Hz ELF
radio dies by ~1 km. Only sound crosses the abyss (a 180 dB baleen call at
50 Hz: ~120 km with deep-water spreading); eDNA integrates ~400 m and ~20 h;
a bioluminescent flash is visible ~36 m from an in-situ camera. The satellite's
job is bathymetry and surface productivity — *where to look*.

**Hypothesis.** Given sensors in the water, an open-set model trained on known
species can flag clips that belong to none of them (distance-based scores on a
learned embedding > confidence-based scores > reconstruction error), degrade
gracefully with range, and group the flagged clips into a handful of candidate
species; survey statistics (Chao1, Good–Turing) then say how many remain and a
satellite habitat prior improves where effort goes.

**Falsification.** If embedding-space scores were no better than max-softmax
or than chance on the synthetic soundscape; if novelty did not fall with
range; if clusters of flagged clips bore no relation to the unknown species;
or if Chao1 did not increase toward the truth with effort.

**Five stages.** (1) physical budget, (2) synthetic soundscape + real Watkins
calls, (3) open-set novelty with six scores and three sweeps, (4) clustering
flagged clips, (5) richness estimation and budget allocation.

**Headline.** Embedding-space scores reach AUROC 0.88 (kNN/Mahalanobis) vs 0.70
(max-softmax) and 0.56 (autoencoder) on synthetic data; Mahalanobis falls only
0.92 → 0.86 from 10 km to 300 km ceilings; 2–3 unknown species cluster at
ARI ≥ 0.95, 5–8 merge (ARI 0.3–0.4). On real Watkins calls a from-scratch CNN
reaches 0.56 accuracy and 0.69 AUROC — the embedding is the bottleneck. Chao1
tracks unseen richness; P(next detection is new) falls 19 % → 0.5 % from 10 to
1000 h; a 90 %-reliable habitat prior buys +13 % species per budget, a coin
flip buys nothing.

**Reproduce.**
```bash
cd projects/11_abyss_novelty
make test      # ~6 s
make all       # ~5 min: 5 experiments, 6 figures, 122 macros, paper/main.pdf
```
