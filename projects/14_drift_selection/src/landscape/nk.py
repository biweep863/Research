"""NK fitness landscapes (Kauffman & Levin 1987), adjacent-neighbourhood
variant: locus i's fitness contribution depends on its own bit and the next
K loci (circular). Each locus has an independent random contribution table,
so K controls epistasis/ruggedness directly: K=0 is purely additive (a
single global optimum in expectation); larger K packs in more local optima.

Fully vectorised in torch so the same code runs on CPU (local_small) or GPU
(gpu_full) by moving tensors to `device` -- no separate numpy/torch path.
"""
from __future__ import annotations

import torch


class NKLandscape:
    def __init__(self, L: int, K: int, seed: int, device: str = "cpu"):
        self.L, self.K, self.device = L, K, device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.table = torch.rand(L, 2 ** (K + 1), generator=g).to(device)
        neighbours = (torch.arange(L).unsqueeze(1) + torch.arange(K + 1).unsqueeze(0)) % L
        self.neighbours = neighbours.to(device)              # (L, K+1)
        weights = (2 ** torch.arange(K, -1, -1)).to(device)  # (K+1,)
        self._weights = weights

    def fitness(self, genomes: torch.Tensor) -> torch.Tensor:
        """genomes: (..., L) uint8/bool/float in {0,1}. Returns (...,) mean
        contribution over loci, in [0, 1)."""
        bits = genomes.long()
        gathered = bits[..., self.neighbours]                 # (..., L, K+1)
        idx = (gathered * self._weights).sum(-1)               # (..., L)
        contrib = self.table[torch.arange(self.L, device=self.device), idx]
        return contrib.mean(-1)

    def brute_force_optimum(self, batch: int = 1 << 16) -> tuple[torch.Tensor, float]:
        """Exact global optimum by exhaustive enumeration (2**L genomes).
        Only meaningful while 2**L is small enough to enumerate -- both
        configs keep L modest for exactly this reason."""
        n = 1 << self.L
        best_fit = -1.0
        best_genome = None
        for start in range(0, n, batch):
            end = min(start + batch, n)
            ints = torch.arange(start, end, device=self.device)
            bits = ((ints.unsqueeze(1) >> torch.arange(self.L - 1, -1, -1, device=self.device)) & 1)
            fit = self.fitness(bits)
            m = fit.max().item()
            if m > best_fit:
                best_fit = m
                best_genome = bits[fit.argmax()].clone()
        return best_genome, best_fit
