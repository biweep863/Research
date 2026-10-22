"""Vehicle-level discrete-event queues used to validate the closed forms.

* multi_server_fifo: c identical lanes, FIFO, Poisson arrivals, exponential or
  deterministic service (Kiefer-Wolfowitz workload recursion).
* signal_queue: one approach at a fixed-time signal; vehicles discharge at the
  saturation headway only during green (Webster's setting).
"""
from __future__ import annotations

import heapq

import numpy as np


def multi_server_fifo(lam: float, mu: float, c: int, n: int, rng: np.random.Generator,
                      service: str = "exp", warmup: float = 0.1) -> dict:
    """Mean wait in queue (h) and P(wait > 0) after discarding a warm-up fraction."""
    arr = np.cumsum(rng.exponential(1 / lam, n))
    if service == "exp":
        svc = rng.exponential(1 / mu, n)
    else:
        svc = np.full(n, 1 / mu)
    free = [0.0] * c                     # time each server becomes free (min-heap)
    heapq.heapify(free)
    wait = np.empty(n)
    for i in range(n):
        f = heapq.heappop(free)
        start = max(arr[i], f)
        wait[i] = start - arr[i]
        heapq.heappush(free, start + svc[i])
    w = wait[int(warmup * n):]
    return {"Wq": float(w.mean()), "Pw": float((w > 1e-12).mean()), "waits": w}


def signal_queue(q_vph: float, s_vph: float, cycle_s: float, g_over_c: float, n: int,
                 rng: np.random.Generator, warmup: float = 0.1) -> float:
    """Mean delay (s) of Poisson arrivals at a fixed-time signal.

    Green occupies the first g C seconds of each cycle; a vehicle departs at
    the later of its arrival and the previous departure + saturation headway,
    pushed to the next green start if that falls in red.  Delay is measured
    against free passage, so a vehicle arriving on green to an empty queue
    has zero delay.
    """
    h = 3600.0 / s_vph
    green = g_over_c * cycle_s
    arr = np.cumsum(rng.exponential(3600.0 / q_vph, n))
    last = -np.inf
    delay = np.empty(n)
    for i, a in enumerate(arr):
        t = max(a, last + h)
        phase = t % cycle_s
        if phase >= green:
            t += cycle_s - phase
        delay[i] = t - a
        last = t
    return float(delay[int(warmup * n):].mean())
