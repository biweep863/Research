# 04 · Results

## Headline (local pilot: L=16, K∈{0,2,4,6}, 5 landscapes/K, 100 replicates)

| K | $N_c$ (95% CI) | max $P(\text{success})$ observed |
|---|---|---|
| 0 | 2.7 (2.4–2.8) | 1.00 (at N=256) |
| 2 | 13.4 (12.3–14.5) | 0.98 |
| 4 | 212.3 (180.2–265.3) | 0.53 |
| 6 | 469.4 (387.8–598.0) | 0.37 |

$N_c$ grows ~175x from K=0 to K=6. Comparing a linear fit ($N_c = a+bK$,
AIC 37.2) against a log-linear fit ($\log N_c = a+bK$, AIC 41.9): **AIC
prefers the linear fit** at this resolution — the log-linear scaling this
project's hypothesis predicted is not supported by the local pilot. Four
K-points is a weak test either way; see the GPU-scale run for a 10-point
grid with 6x more landscape averaging.

## The mechanism test: does Kimura's formula, fed the measured s(K), predict the observed $N_c(K)$?

| K | $s(K)$ | $N_c^{\text{Kimura}} = 1/s(K)$ | $N_c$ (measured, from GA runs) |
|---|---|---|---|
| 0 | 0.0379 | 26.4 | 2.7 |
| 2 | 0.0748 | 13.4 | 13.4 |
| 4 | 0.1024 | 9.8 | 212.3 |
| 6 | 0.1178 | 8.5 | 469.4 |

**This is the project's central, honest negative result.** $s(K)$ — the
mean beneficial single-mutation effect size, measured directly from each
landscape's distribution of fitness effects — *increases* with K (fit:
$s(K) \approx e^{-3.13 + 0.19K}$), not decreases as the hypothesis in
`doc/00_overview.md` predicted. Consequently $N_c^{\text{Kimura}}(K)$
*falls* with K (26.4 → 8.5), the opposite direction from the measured
$N_c(K)$ (2.7 → 469.4), which rises sharply. The two columns don't just
disagree in magnitude — they disagree in *sign of trend*, so this is not
"the theory is roughly right but needs a scale factor": the single-locus,
Kimura-style mechanism for why bigger populations are needed on rugged
landscapes is falsified by this measurement.

**What this means.** The population-size threshold itself looks real
(the $P(\text{success})$ vs $N$ curves in `paper/figures/success_vs_n.pdf`
show a genuine sigmoid-like transition, not noise), but the reason is very
likely *not* "epistasis makes any single beneficial mutation weaker and
therefore easier for drift to lose" (the classical single-locus
population-genetics story). A more plausible mechanism, not tested
quantitatively here: higher K multiplies the number of local optima
(a standard NK-model fact), so bigger $N$ helps mainly by maintaining
enough simultaneous diversity across the population to keep multiple
basins of attraction explored at once — a *multi-peak metastability*
argument, not a single-locus drift/selection balance. Testing that
directly (e.g. counting how many distinct local optima a population's
diversity spans over time) is a natural follow-up, not done in this pass.

## Effective population size sanity check
$N_e$ estimated from allele-frequency trajectories at $N=100$ (no
elitism) sits at 28–36 across K=0–6 — the same order of magnitude as the
census population size $N=100$, as expected for a simple Wright-Fisher-like
GA without artificial diversity preservation. This confirms the GA's
population dynamics are not doing something wildly non-Wright-Fisher-like
(e.g. an $N_e \gg N$ from some hidden diversity-boosting effect), which
would have undermined using Kimura's formula as a comparison point at all.

## GPU-scale run (L=20, K=0..9, 10 landscapes/K, 1000 replicates)

| K | $N_c$ | $s(K)$ | $N_c^{\text{Kimura}}$ |
|---|---|---|---|
| 0 | 3.7 | 0.0336 | 29.7 |
| 1 | 9.8 | 0.0519 | 19.3 |
| 2 | 40.1 | 0.0583 | 17.2 |
| 3 | 367.5 | 0.0691 | 14.5 |
| 4 | 1374.6 | 0.0776 | 12.9 |
| 5 | 14490.6 | 0.0884 | 11.3 |
| 6 | 2409.2 | 0.0938 | 10.7 |
| 7 | 3591.2 | 0.1017 | 9.8 |
| 8 | 25350.6 | 0.1066 | 9.4 |
| 9 | 32506.0 | 0.1149 | 8.7 |

$N_c$ grows ~8785x from K=0 to K=9 (first seen as a 175x increase over
K∈{0,2,4,6} in the local pilot). Linear vs log-linear AIC: **181.5 vs
194.7 — linear still preferred**, at more than double the K-resolution of
the local pilot. This is not "the pilot was underpowered and log-linear is
actually right" — the same conclusion holds up under a much stronger test.
$N_c(K)$ is still non-monotonic at high K (K=5 exceeds K=6) even after
averaging 10 independent landscape draws per K, which either means
landscape-instance variance needs substantially more than 10 draws to
average out at high K, or the true $N_c(K)$ relationship genuinely isn't
smooth — not distinguishable with the current data.

**The mechanism falsification replicates, and more cleanly.** $s(K)$
increases *monotonically* across all 10 points (vs. the local pilot's
4-point version, which was monotonic but noisier), so $N_c^{\text{Kimura}}$
falls monotonically from 29.7 to 8.7 while the measured $N_c$ rises three
to four orders of magnitude over the same K range — the same falsification
as the local pilot, now with much stronger statistical support that it
isn't a small-sample fluke.

See `paper/main.pdf` for figures (success probability vs N per K with
fitted $N_c$; $N_c(K)$ with both fits overlaid; $s(K)$ decay/growth and the
two independent $N_c(K)$ estimates side by side).
