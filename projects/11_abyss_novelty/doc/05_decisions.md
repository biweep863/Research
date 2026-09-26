# 05 · Decisions & Alternatives

## Answer the physics before building the AI
The idea as posed ("from satellite images") is physically impossible below a
few hundred metres. Rather than dismiss it, we quantify every modality with
closed forms and reframe the satellite's role as habitat prior. This is the
paper's first result and it shapes the rest.

## Closed forms, not a propagation code
Beer–Lambert, skin depth, sonar equation with Thorp/Wenz, and a 1-D eDNA
plume are textbook and reproducible in milliseconds. A ray/PE propagation
model (Bellhop, RAM) would change ranges by factors, not orders of magnitude,
and is a data swap in the scaled config. We report both spherical and hybrid
spreading so the reader sees the sensitivity.

## Ideal-detector optics on purpose
The 1e-9 W/m² floor is far better than any real satellite sensor; using it
makes the "nothing optical reaches 1 km" claim robust to any objection about
better cameras.

## Simulate species as parametric call families
No public recording of undescribed species exists by definition. Parametric
families with species-specific parameter draws give ground truth for
"unknown" while keeping physical propagation and noise real. Four families
cover the mechanisms that dominate deep-sea sound; more are a config list.

## Real data as an honesty check, not the headline
Watkins/BEANS is shallow-water marine mammals, not the abyss, and only two of
four shards were used to keep the cache small (3 MB). Its role is to show the
gap between a physically clean simulator and real recordings — and it does:
the from-scratch CNN embedding fails to organise species, and the paper says
so instead of hiding the number. The remedy (pretrained backbone) is stated
as the scaled configuration rather than bolted on.

## Six scores, including two that lose
Max-softmax and energy (confidence-based), kNN and Mahalanobis (embedding
geometry), autoencoder (classifier-free) and a PCA baseline. Keeping the
losers is the point: the ordering (geometry > confidence > reconstruction)
is the transferable finding, and the autoencoder's failure ("a new call is
still a call") is physically meaningful.

## From-scratch CNN, 12 epochs
Chosen for the ≤ 5-minute budget; a pretrained audio model would dominate on
real data but would make the study depend on a large download. The trade-off
is documented in the log and the limitations.

## Flag at the 95th percentile of known scores
A fixed-false-positive rule is what a monitoring system would use; it trades
recall (0.5–0.9) for precision (≥ 0.84) and keeps the clustering step clean.

## Silhouette for k, and its failure reported
Silhouette is the standard unsupervised choice. It over-splits at U = 2–3 and
under-splits at U = 5–8 (grouping by family). We did not switch to a
criterion tuned to the answer; the failure is where the human enters.

## Chao1 and Good–Turing rather than a fitted rarefaction curve
Both are non-parametric, closed-form, and standard in ecology; Chao1 is a
lower bound and behaves like one in our curves. Fisher's log-series is the
classic many-rare-species model and makes the "each new species costs
exponentially more hours" conclusion explicit.

## Greedy allocation with a stated prior quality
The value of the satellite is the value of its prior; sweeping the prior's
reliability from a coin flip to perfect makes that explicit (no gain at 0.5,
+13 % at 0.9). A full Bayesian adaptive survey is a natural extension.

## Numbers are macros
Every number in the paper comes from `paper/macros.tex`; LaTeX command names
cannot contain digits, so the figure script maps them to words.

## Pretrained backbone: AST, not a from-scratch fix
The local pilot's own limitation ("the fix is a pretrained audio embedding")
is implemented as `MIT/ast-finetuned-audioset-10-10-0.4593`, frozen, feeding
a deterministic sklearn probe (`fit_probe`) rather than another from-scratch
torch classifier — this avoids the CPU/GPU training non-determinism already
noted in `doc/03_experiments_log.md` for the CNN path. `ae` and `pca_msp` are
reused from the CNN path's mel features rather than recomputed on the
backbone embedding, since neither score is architecture-specific (both look
at mel-space reconstruction/variance, not the classifier's own embedding
geometry); this is stated rather than silently assumed.

## train_low excluded, not "the full dataset"
BEANS-Watkins ships five shards; `train_low` is very likely a subset of
`train` (unconfirmed from the dataset card alone), and including it risked
duplicate clips crossing the train/held-out boundary, artificially inflating
AUROC. sha1-based deduplication across the four shards that *are* used
catches exact duplicates but not re-cut segments of the same recording — a
residual leakage risk, listed rather than assumed away.

## GPU passthrough required a real fix, not just a flag
`run_novelty.py` and `run_cluster.py` build a torch `CNNClassifier`/`ConvAE`
per sweep point but never passed a `device` argument to `train_classifier`
(added in `models.py` for the `run_real.py` backbone comparison) — so the
first `gpu_full` sweep ran entirely on CPU as if the GPU flag didn't exist,
taking a projected many hours for 18 points. The 17-minute GPU sweep
required plumbing `resolve_device(cfg)` through both entry points, not just
enabling `device: cuda` in the config.
