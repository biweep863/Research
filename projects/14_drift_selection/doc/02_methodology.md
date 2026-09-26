# 02 · Methodology

## Pipeline
```
NKLandscape(L, K, seed) --brute_force_optimum()--> exact global optimum
        |
        v
run_ga(landscape, N, R, generations, ...) --batched (R,N,L) tensor--> RunResult
        |                                                  |
        v                                                  v
run_threshold_sweep.py: P(success) vs N, per K       run_effective_size.py: s(K), Ne(K)
        |                                                  |
        v                                                  v
   N_c(K) via logistic fit                        N_c^Kimura(K) = 1/s(K)
        \_______________________________  ________________/
                                        \/
                             plot_drift.py --> figures + macros.tex --> paper/main.pdf
```

## GA operators
Generational, no elitism (see `doc/05_decisions.md`): 2-way tournament
selection, uniform crossover (rate `p_crossover` per locus), independent
bit-flip mutation (rate `p_mutation` per locus). All batched across a
`(replicates, population, L)` tensor in `src/ga/simple_ga.py`, so the same
code runs on CPU or GPU by moving tensors to `device`.

## Success criterion
A replicate "succeeds" at generation $g$ if its population's best fitness
reaches the landscape's exact global optimum within `success_tol` (fitness
values are landscape-mean contributions in $[0,1)$; `success_tol=1e-6`
tolerates only floating-point noise, not near-misses). $P(\text{success})$
at a given $(N,K)$ is the fraction of replicates that ever succeed within
the generation budget, averaged over `landscapes_per_k` independent
landscape draws (doc/05_decisions.md explains why the average over
landscapes, not a single draw, is the primary quantity).

## Threshold fit ($N_c$)
`fit_threshold` (`src/analysis/effective_size.py`) fits a logistic
regression of the (binomial-count-weighted) success indicator on
$\log_2 N$, and reports the population size where the fitted curve crosses
$P=0.5$. A bootstrap over the trial-level binomial noise (`n_boot=500` by
default) gives a 95% CI on $N_c$.

## Selection coefficient $s(K)$ and Kimura's formula
$s(K)$ is estimated by sampling random genomes, flipping one random locus,
and averaging the *positive* relative fitness changes
$(f_{\text{mutant}} - f_{\text{parent}}) / f_{\text{parent}}$ across
samples (`dfe_selection_coefficient`) — the mean beneficial effect size of
the landscape's distribution of fitness effects (DFE), a standard
population-genetics object. Kimura's (1962) haploid Wright-Fisher diffusion
fixation probability,
$$u(s,N) = \frac{1-e^{-2s}}{1-e^{-2Ns}},$$
gives $N_c^{\text{Kimura}}(K) = 1/s(K)$ as the population size at which
$Ns=1$, the conventional drift/selection crossover.

## Effective population size (secondary check)
$N_e$ is estimated from the GA's own allele-frequency trajectory at one
tracked locus via the standard temporal (F-statistic) estimator,
$N_e = \bar p(1-\bar p) / (2 \,\mathrm{Var}(\Delta p))$ (Krimbas & Tsakas
1971; Waples 1989), to confirm the GA's population behaves like a finite
Wright-Fisher population of roughly the size it is (not, say, wildly larger
due to some hidden source of extra diversity-preservation the simple GA
doesn't have — e.g. elitism, which was deliberately excluded).

## Linear vs log-linear fit of $N_c(K)$
Two 2-parameter fits of the (K, $N_c$) pairs — $N_c = a + bK$ and $\log N_c
= a + bK$ — compared by AIC (`_aic` in `run_threshold_sweep.py`), computed
on the residuals in raw $N_c$ units for both (so the comparison is on the
same scale, not an apples-to-oranges log-space vs linear-space RSS).
