"""Elevator kinematics: how long a car takes to move n floors and to stop.

Real cars follow a trapezoidal (or S-curve) velocity profile: accelerate at
``a`` to rated speed ``v``, cruise, decelerate.  For a run of distance d the
exact trapezoid time is  d/v + v/a  when the car reaches rated speed
(d >= v^2/a) and  2 sqrt(d/a)  otherwise (CIBSE Guide D; Barney & Al-Sharif).

The simulator moves floor by floor, so it uses the linearised form
``run(n) = n * H / v + v / a`` -- a per-floor cruise time plus one
start/stop penalty per run.  For rated speeds up to ~2.5 m/s the two agree to
within a fraction of a second on one-floor runs (see ``trapezoid_error``),
and the closed-form model uses the same linearised time so that the
discrete-event simulator can be validated against it exactly at zero traffic.

A *stop* costs the door cycle (open + close) plus a transfer time per
passenger boarding or alighting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Kinematics:
    v: float = 1.6            # rated speed, m/s
    a: float = 1.0            # acceleration, m/s^2
    floor_h: float = 3.3      # floor-to-floor height, m
    t_door: float = 4.0       # door open + close cycle, s
    t_pass: float = 1.2       # transfer time per passenger, s

    @classmethod
    def from_cfg(cls, d: dict) -> "Kinematics":
        return cls(v=d.get("v", 1.6), a=d.get("a", 1.0), floor_h=d.get("floor_h", 3.3),
                   t_door=d.get("t_door", 4.0), t_pass=d.get("t_pass", 1.2))

    def run(self, n_floors) -> np.ndarray | float:
        """Linearised run time for |n| floors (0 for n = 0)."""
        n = np.abs(np.asarray(n_floors, dtype=float))
        return np.where(n > 0, n * self.floor_h / self.v + self.v / self.a, 0.0)

    def run_trapezoid(self, n_floors) -> np.ndarray | float:
        """Exact trapezoidal-profile time, for reference and tests."""
        d = np.abs(np.asarray(n_floors, dtype=float)) * self.floor_h
        reach = d >= self.v ** 2 / self.a
        return np.where(d > 0, np.where(reach, d / self.v + self.v / self.a, 2 * np.sqrt(d / self.a)), 0.0)

    def stop(self, n_pass: int = 1) -> float:
        """Time cost of one stop with ``n_pass`` passengers transferring."""
        return self.t_door + self.t_pass * n_pass

    def trapezoid_error(self, max_floors: int = 5) -> float:
        n = np.arange(1, max_floors + 1)
        return float(np.max(np.abs(self.run(n) - self.run_trapezoid(n))))
