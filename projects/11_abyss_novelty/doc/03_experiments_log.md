# 03 · Experiments log

All runs use `configs/local_small.yaml` (seed 0) on a laptop CPU.

| Date | Config | Command | Outcome |
|------|--------|---------|---------|
| 2026-09-15 | local_small | `make test` | 10 passed, 6 s (after fixing three test expectations, see below) |
| 2026-09-15 | local_small | `run_budget` (1 s) | optics ≤ 576 m even ideal; radar skin 2.5 mm; ELF reach 1 km; baleen 23 km spherical / 121 km hybrid; click 12 km; fish 115 m; eDNA footprint ~400 m, memory 20 h |
| 2026-09-15 | local_small | `run_novelty` (1:22) | base: acc 0.77, kNN 0.88 (CI 0.85–0.91), Maha 0.88, msp 0.70, energy 0.75, AE 0.56, PCA 0.49; range 10→300 km: Maha 0.92→0.86 |
| 2026-09-15 | local_small | `run_real` (2:11) | 796 clips / 29 species; acc 0.52–0.62; msp 0.69±0.07 best; kNN 0.59; Maha 0.49; AE 0.42; cluster ARI 0.02–0.63 |
| 2026-09-15 | local_small | `run_cluster` (0:55) | U=2: ARI 0.98, k̂=4; U=3: 0.95, k̂=4; U=5: 0.31, k̂=2; U=8: 0.43, k̂=3; flag precision ≥ 0.84 |
| 2026-09-15 | local_small | `run_survey` (3 s) | seen 22/64/125/157 of 300 at 10/100/1000/3000 h; Chao1 47/95/160/191; P(new) 19 %→0.2 %; greedy gain −1/+1/+13/+17 % at q = 0.5/0.7/0.9/1.0 |
| 2026-09-15 | local_small | `make figures` | 6 figures, 122 macros |
| 2026-09-15 | local_small | `make paper` | 7 pages, no overfull boxes, no undefined refs |

`make all` ≈ 5 min end to end.

## Observations
- **YAML exponents.** `1.0e10` parses as a *string* in PyYAML (needs
  `1.0e+10`); the budget crashed until all exponents got explicit signs.
- **First CNN did not learn.** 30 epochs at batch 64 on ~100 clips is ~50
  gradient steps; known-class accuracy was chance. Batch 32 / 30 epochs on
  ~500 clips gives 0.97; the final config uses 12 epochs to fit the time
  budget (accuracy 0.77–0.99 depending on SNR).
- **Time budget.** The first full sweep took 5:16 (novelty) + 4:34 (cluster).
  Trimming to 40 clips/species, 12 CNN / 10 AE epochs, 8 sweep points and one
  clustering replicate brought `make run` to ~4.5 min.
- **Real data is hard from scratch.** 12→40→80 epochs moved known accuracy
  0.34→0.48→0.58 and max-softmax AUROC 0.58→0.52→0.76; we run 50 epochs
  (0.56 accuracy, 0.69 AUROC). Embedding-space scores that win on synthetic
  data are near chance here — the embedding has not learned species geometry.
  Reported as the honest pilot; the fix (pretrained audio backbone) is the
  scaled configuration.
- **Silhouette under-counts.** With 5–8 unknown species the silhouette picks
  2–3 clusters (it groups by call family). Reported, not tuned away.
- **Allocation gain needed contrast.** With plain richness 120 and 10
  replicates the greedy gain was noisy (+1 % at q = 0.9). Richness 60 and 40
  replicates give the monotone curve; q = 0.5 correctly shows no gain.
- Watkins soccer-style gotcha: the HF mirror's `label` uses underscores and
  commas in species names (e.g. `Grampus,_Rissos_Dolphin`); kept verbatim.
- **Run-to-run jitter.** A clean `make all` re-run reproduced every physics
  and survey number exactly but moved the synthetic novelty AUROCs by ±0.01
  (kNN 0.88 → 0.89) and the base closed-set accuracy from 0.77 to 0.94 (a
  12-epoch CNN sits on a steep part of its learning curve): torch's CPU
  thread scheduling is not bit-deterministic. The novelty ordering of the
  scores was identical in both runs.
  The paper always quotes the macros of the last run; doc numbers may differ
  in the second decimal.
