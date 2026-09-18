"""Where the resident goes.

The resident lives on floor ``h`` of a building with floors 1..N (1 = lobby).
They alternate *outbound* trips from home and *return* trips to home:

* outbound, with probability ``alpha``: to the lobby (leaving the building);
  otherwise to another floor f != h, f >= 2, drawn uniformly (or weighted);
* return: from wherever they are back to h.

This chaining makes every trip pair symmetric by construction (the car has to
come and get you at the lobby just as often as at home) and it is what makes
the "car stays where it last stopped" parking policy well defined.
"""
from __future__ import annotations

import numpy as np


class TripGenerator:
    def __init__(self, n_floors: int, home: int, alpha: float, rng: np.random.Generator,
                 weights: np.ndarray | None = None):
        assert 1 <= home <= n_floors
        self.N, self.h, self.alpha, self.rng = n_floors, home, alpha, rng
        others = np.array([f for f in range(2, n_floors + 1) if f != home])
        self.others = others
        if weights is not None:
            w = np.asarray(weights, dtype=float)[others - 1]
            self.w = w / w.sum() if w.sum() > 0 else None
        else:
            self.w = None
        self.loc = home

    def next(self) -> tuple[int, int, str]:
        """Return (origin, destination, kind) and move the resident there."""
        if self.loc == self.h:
            if self.rng.random() < self.alpha or len(self.others) == 0:
                dest, kind = 1, "lobby_out"
            else:
                dest = int(self.rng.choice(self.others, p=self.w)); kind = "intra_out"
        else:
            dest, kind = self.h, ("lobby_back" if self.loc == 1 else "intra_back")
        o, self.loc = self.loc, dest
        return o, dest, kind


def od_distribution(n_floors: int, home: int, alpha: float) -> list[tuple[int, int, float]]:
    """Exact probability of each (origin, destination) pair over a long
    sequence of chained trips: outbound and return each have mass 1/2."""
    others = [f for f in range(2, n_floors + 1) if f != home]
    rows = []
    p_lobby = alpha if others else 1.0
    rows.append((home, 1, 0.5 * p_lobby)); rows.append((1, home, 0.5 * p_lobby))
    for f in others:
        p = 0.5 * (1 - p_lobby) / len(others)
        rows.append((home, f, p)); rows.append((f, home, p))
    return rows
