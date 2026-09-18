"""The stairs: an alternative with an up/down asymmetry.

Walking down a flight costs about 10 s, walking up 15-20 s (and people are
rarely willing to climb more than a few floors).  The resident takes the
stairs when the *expected* elevator time for the trip exceeds the stairs time
and the climb is within their limit.  Decisions are made on expectations
(they do not know where the car is), so the rule is a pure function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stairs:
    t_up: float = 15.0        # s per floor climbed
    t_down: float = 10.0      # s per floor descended
    max_climb: float = 3      # floors the resident is willing to climb (inf = any)
    max_descend: float = float("inf")

    @classmethod
    def from_cfg(cls, d: dict | None) -> "Stairs | None":
        if not d or not d.get("enabled", True):
            return None
        return cls(t_up=d.get("t_up", 15.0), t_down=d.get("t_down", 10.0),
                   max_climb=float(d.get("max_climb", 3)), max_descend=float(d.get("max_descend", "inf")))

    def time(self, origin: int, dest: int) -> float:
        n = dest - origin
        if n > 0:
            return np.inf if n > self.max_climb else n * self.t_up
        if n < 0:
            return np.inf if -n > self.max_descend else -n * self.t_down
        return 0.0

    def take_stairs(self, origin: int, dest: int, elevator_expected: float) -> bool:
        return self.time(origin, dest) < elevator_expected
