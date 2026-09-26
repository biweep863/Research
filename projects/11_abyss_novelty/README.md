# Project 11 — Nothing sees the abyss but sound: can AI discover unknown deep-sea species?

## Question
Could satellites (or any remote sensor) reveal unknown animals below the
abyssal zone (> 4,000 m), and could an AI recognise a creature nobody has
catalogued? Part 1 answers the first with closed-form physics; Part 2 builds an
open-set AI on simulated and real bioacoustics; Part 3 asks how many species
remain and where to spend sensor hours.

## The five stages
1. **Physical sensing budget** — light (Beer–Lambert, Jerlov types), radio/radar
   (skin depth), sound (passive sonar equation, Thorp absorption, Wenz noise,
   spherical/hybrid spreading), eDNA (shed–decay–transport plume); what the
   satellite *can* do (bathymetry, surface productivity → habitat prior).
2. **Synthetic deep-sea soundscape** — species = parametric call families
   (tonal, click, drum, burst) with species-specific parameters, propagated to
   a hydrophone; 64-band log-mel clips.
3. **Open-set novelty** — CNN embedding trained on known species; six novelty
   scores (max-softmax, energy, kNN, Mahalanobis, autoencoder, PCA baseline);
   sweeps over range/SNR, number of known species and clips; the same pipeline
   on real Watkins marine-mammal calls (BEANS packaging).
4. **From "novel" to "how many"** — flag clips above the 95th percentile of
   known scores, cluster their embeddings, compare with true unknown species.
5. **How many remain, where to look** — log-series community, Chao1 and
   Good–Turing vs sensor-hours; greedy allocation of a sensor budget over a
   satellite habitat prior of varying reliability.

## Quick start
```bash
# from repo root
source .venv/bin/activate
cd projects/11_abyss_novelty
make test           # 13 tests: physics closed forms, mel, species, scores, clustering, estimators (~6 s)
make run            # 5 experiments -> results/*.json (~4.5 min on a laptop CPU)
make figures        # 6 figures + paper/macros.tex
make paper          # compile paper/main.pdf
```
Real calls come from `data/watkins_mel.npz` (3 MB log-mel cache, checked in).
To rebuild it from the BEANS-Watkins parquet shards on Hugging Face
(`DBD-research-group/beans_watkins`), put them in `data/raw/` or set
`WATKINS_DIR`. Without cache or shards the real-data experiment is skipped and
says so.

## Scaling up
```bash
make all CONFIG=configs/gpu_full.yaml
```
40 + 15 synthetic species, 200 clips each, finer sweeps, 10 real holdout draws
with 300-epoch training, 1000-species survey with three habitats — config-gated.
The intended next step for real data (a pretrained audio embedding instead of
the from-scratch CNN) is a model swap in `src/novelty/models.py`.

## Files
- `src/physics/{light,radio,acoustics,edna}.py` — closed-form budgets
- `src/sim/{species,propagate,dataset}.py` — call families, propagation + Wenz noise, clip builder
- `src/data/watkins.py` — BEANS-Watkins loader and log-mel cache
- `src/novelty/{features,models,scores,cluster}.py` — numpy mel, torch CNN/AE + sklearn baseline, six scores, clustering
- `src/survey/{community,estimators,allocate}.py` — log-series sampling, Chao1/Good–Turing, greedy allocation
- `src/experiments/run_{budget,novelty,real,cluster,survey}.py` — write `results/*.json`
- `src/figures/plot_abyss.py` — 6 figures + 122 LaTeX macros
- `paper/main.tex` — two-column paper (7 sections)
- `doc/` — overview, data, methodology, experiments log, results, decisions

## Headline
No optical or radio signal passes ~0.6 km even with an ideal detector; only
low-frequency sound crosses the abyss (a 180 dB call at 50 Hz: ~120 km). On a
physically simulated soundscape, embedding-space novelty scores flag unknown
species at AUROC 0.88 (max-softmax 0.70, autoencoder 0.56), degrade gracefully
with range, and cluster 2–3 new species almost perfectly (ARI ≥ 0.95) but merge
larger sets. On real Watkins calls a from-scratch CNN reaches only 0.66–0.69
AUROC — the bottleneck is the embedding, not the scores; swapping in a frozen
pretrained audio backbone (AST) raises it to 0.84, positive in 10/10 held-out
draws, without any fine-tuning. Chao1 tracks unseen richness; the probability
that the next detection is new falls from 19 % at 10 h to 0.5 % at 1000 h; a
90 %-reliable satellite habitat prior buys +13 % species per budget, while an
unreliable one (34 %) costs −3 % relative to no prior at all.

## Scaling up (results)
Full run on 4 of 5 BEANS-Watkins shards (1,668 clips, 31 species) with the
AST backbone: see `doc/04_results.md` for the complete CNN-vs-backbone table,
the 18-point synthetic sweep, and the 1000-species survey.
