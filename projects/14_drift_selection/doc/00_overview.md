# 00 · Overview

## Question
Does a genetic algorithm (GA) on a rugged fitness landscape need a
population above a critical size $N_c$ to reliably reach the optimum, and
does population genetics' own theory of genetic drift explain why $N_c$
grows with ruggedness?

## Hypothesis
GA population sizing is normally set by empirical rules of thumb (Goldberg
& Rudnick's schema-survival/noise-to-signal arguments), not by the
population-genetics formalism that genetic algorithms are metaphorically
named after. Population genetics has a precise answer to "when does chance
dominate over selection": the fixation probability of a mutation with
selection advantage $s$ in a population of size $N$ (Kimura 1962,
diffusion approximation) transitions from the neutral value $1/N$ to the
selection-dominated value $2s$ around $Ns \approx 1$. If a GA behaves like
a finite Wright-Fisher population, there should be a critical population
size $N_c \approx 1/s$ below which the search is drift-dominated (loses
beneficial mutations by chance) and above which selection reliably fixes
them.

The mechanistic prediction this project tests: on an NK landscape (Kauffman
& Levin 1987), higher epistasis $K$ is known to multiply the number of
local optima roughly exponentially. If that also means the *typical single-
mutation fitness effect* shrinks with $K$ (more interference between loci
diluting any one flip's effect), then $s(K)$ should decay, and $N_c(K) =
1/s(K)$ should grow — plausibly log-linearly in $K$, since an exponential
decay in $s(K)$ gives a log-linear $N_c(K)$.

## Falsification criterion
The hypothesis is wrong if either: (a) $N_c$ does not grow with $K$ at all
(no drift-dominated failure mode exists), or (b) $N_c$ grows but the
measured selection coefficient $s(K)$ does not shrink accordingly — i.e.
Kimura's formula, fed the *measured* $s(K)$, predicts the wrong direction
of change in $N_c$. This project finds (b): see `doc/04_results.md`.

## Novelty / delta
Goldberg & Rudnick's classic population-sizing theory (1991) reasons about
schema-decision noise, not drift/selection balance in the population-
genetics sense. "Effective population size" is a standard idea in
evolutionary-algorithm theory (e.g. in the context of genetic drift in GAs,
Rogers & Prügel-Bennett and others), but the specific test run here — an
independently-measured landscape selection coefficient $s(K)$ fed into
Kimura's formula, cross-checked against a directly-fitted $N_c(K)$ from GA
runs, on an NK-model ruggedness sweep — is an inference made for this
project, not a result recalled from a specific paper. It should be read as
a hypothesis this project tests and reports on, not a literature summary.
