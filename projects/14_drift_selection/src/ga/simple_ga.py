"""A canonical generational GA (tournament selection, uniform crossover,
bit-flip mutation, no elitism) batched across independent replicates as a
single (R, N, L) tensor, so R independent runs execute as one vectorised
op -- the same code scales from a handful of replicates on CPU to thousands
on GPU by changing only the config.

No elitism is a deliberate choice (doc/05_decisions.md): elitism is an
artificial guarantee against losing the optimum that has no population-
genetics analogue, and this project is explicitly testing whether a GA
reproduces population-genetics drift/selection behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..landscape.nk import NKLandscape


@dataclass
class RunResult:
    best_ever: torch.Tensor        # (R,) best fitness seen per replicate
    gen_to_success: torch.Tensor   # (R,) generation index of first success, -1 if never
    freq_history: torch.Tensor     # (G+1, R) allele frequency at a tracked locus, per replicate
    final_pop_mean_fitness: torch.Tensor  # (R,)


def _tournament_select(fitness: torch.Tensor, rng: torch.Generator, device: str) -> torch.Tensor:
    """fitness: (R, N). Returns (R, N) parent indices via 2-way tournament."""
    R, N = fitness.shape
    a = torch.randint(0, N, (R, N), generator=rng, device=device)
    b = torch.randint(0, N, (R, N), generator=rng, device=device)
    fa = torch.gather(fitness, 1, a)
    fb = torch.gather(fitness, 1, b)
    return torch.where(fa >= fb, a, b)


def run_ga(landscape: NKLandscape, N: int, R: int, generations: int, p_c: float, p_m: float,
           optimum_fitness: float, success_tol: float, seed: int, device: str,
           tracked_locus: int = 0) -> RunResult:
    """All randomness is drawn by a generator on `device` directly (not a
    CPU generator + transfer): for large (R, N, L) tensors on CUDA, CPU-side
    RNG generation followed by a host-to-device copy was measured to be the
    dominant cost (N=1024, R=1000, L=20: 37.6s/call vs 0.34s at N=4 --
    quadratic-looking blowup traced to this, not to the generation loop
    itself). This means CPU and CUDA runs of the same seed are not expected
    to produce bit-identical trajectories, only same-device reproducibility
    (see tests/test_ga.py), which was never guaranteed cross-device anyway."""
    L = landscape.L
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    pop = (torch.rand(R, N, L, generator=gen, device=device) < 0.5).long()

    best_ever = torch.zeros(R, device=device) - 1.0
    gen_to_success = torch.full((R,), -1, dtype=torch.long, device=device)
    freq_history = torch.zeros(generations + 1, R, device=device)

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    for g in range(generations + 1):
        fit = landscape.fitness(pop)                       # (R, N)
        gen_best, _ = fit.max(dim=1)
        newly_solved = (gen_best >= optimum_fitness - success_tol) & (gen_to_success < 0)
        gen_to_success = torch.where(newly_solved, torch.full_like(gen_to_success, g), gen_to_success)
        best_ever = torch.maximum(best_ever, gen_best)
        freq_history[g] = pop[..., tracked_locus].float().mean(dim=1)

        if g == generations:
            break

        parent_idx = _tournament_select(fit, rng, device)                       # (R, N)
        parents = torch.gather(pop, 1, parent_idx.unsqueeze(-1).expand(-1, -1, L))

        shuffle = torch.argsort(torch.rand(R, N, generator=rng, device=device), dim=1)
        mates = torch.gather(parents, 1, shuffle.unsqueeze(-1).expand(-1, -1, L))
        cross_mask = torch.rand(R, N, L, generator=rng, device=device) < p_c
        children = torch.where(cross_mask, mates, parents)

        mut_mask = torch.rand(R, N, L, generator=rng, device=device) < p_m
        pop = children ^ mut_mask.long()

    return RunResult(best_ever=best_ever, gen_to_success=gen_to_success,
                     freq_history=freq_history, final_pop_mean_fitness=fit.mean(dim=1))
