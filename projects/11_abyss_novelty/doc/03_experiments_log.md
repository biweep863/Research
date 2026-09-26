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

## GPU-scale run (2026-09-26)
| Config | Command | Outcome |
|--------|---------|---------|
| gpu_full | env setup (Python 3.12, torch+cu128, RTX 3060 12GB confirmed) | Windows Smart App Control blocked scipy/sklearn native DLLs; disabled with user's explicit approval (was in Evaluation mode). |
| gpu_full | `fetch_watkins.py` (4 of 5 shards, ~1.2GB, `train_low` excluded) + mel cache rebuild | 1,668 clips / 31 species (vs. 796/29 locally). |
| gpu_full | AST embedding build (1,668 clips, 768-d) | ~2 min on GPU. |
| gpu_full | `run_real.py` smoke test (`--limit 1`) | CNN acc 0.80, best msp 0.68; AST-probe acc 0.87, best msp 0.82 (paired_delta +0.144) — first sign the backbone swap helps. |
| gpu_full | `run_real.py`, 10 draws — **first attempt killed after 5+ hours with 0 CPU progress** | A PowerShell `Start-Process -RedirectStandardOutput` pipe deadlocked against the process's own output (a known Windows issue). Fixed by launching all subsequent long jobs via a `.bat` file with native shell (`>`) redirection instead of PowerShell's pipe-based redirection. |
| gpu_full | `run_real.py`, 10 draws (relaunched) | CNN mean best-score AUROC 0.66 (acc 0.70); AST-probe mean best-score AUROC 0.84 (acc 0.88). AST beats CNN on the paired best-score comparison in **10/10 draws**, delta +0.047 to +0.197 (mean +0.125, std 0.041). |
| gpu_full | `run_novelty.py`, full 18-point sweep — **initially very slow on CPU** (CNN training in `run_novelty.py`/`run_cluster.py` had no `device` passthrough; the base point alone ran >45 min with no result) | Fixed by adding the same `device` passthrough already used in `run_real.py`, then relaunched on GPU: **all 18 points finished in ~17 minutes** (vs. an extrapolated many-hour CPU run). AUROCs 0.72–0.97 across sweeps, known-class accuracy 0.36–1.00 depending on clips/species. |
| gpu_full | `run_cluster.py` (GPU) | U=2: ARI 0.91; U=3: 0.52; U=5: 0.94; U=8: 0.76. |
| gpu_full | `run_survey.py` (numpy, CPU) | 1000-species survey; seen 220/1000 at 1000h, Chao1 275±21; greedy allocation gain ranges from **−3% at q=0.34** (worse than uniform) to +13% at q≥0.9. |
| gpu_full | `make figures` | Two bugs found and fixed while rebuilding: `plot_abyss.py` read `rows['clips_15']`, absent from the original `[10,20,50,100,200]` sweep (added 15 to the sweep); `WORDS` digit-to-word map lacked entries for the new `prior_quality` tags (`034`,`06`,`08`,`095`) and asserted on any leftover digit (added a digit-by-digit spelling fallback instead of hardcoding every possible tag). |
