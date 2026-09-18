"""Discrete-event elevator simulator: K cars, collective control, nearest-car
dispatch, Poisson background passengers, and one instrumented resident.

Cars move floor by floor (per-floor cruise time H/v, plus one start/stop
penalty v/a per run) and stop for the door cycle plus a transfer time per
passenger.  Collective control: a car keeps its direction while there are
calls ahead (car calls of passengers on board, or hall calls in its
direction), reverses when there are calls only behind, and otherwise goes to
its parking floor (lobby / middle) or stays put (last).  A new hall call wakes
the nearest idle car; moving cars pick up hall calls they pass in their
direction.  Capacity is enforced.

Everything the paper measures comes from the *resident's* trips: wait, ride,
number of intermediate stops while on board, and the positions of the cars at
the moment the resident pressed the button.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..model.kinematics import Kinematics


@dataclass
class Passenger:
    pid: int
    t_arrive: float
    origin: int
    dest: int
    resident: bool = False
    kind: str = ""
    t_board: float = np.nan
    t_alight: float = np.nan
    stops_on_board: int = 0
    car: int = -1

    @property
    def direction(self) -> int:
        return 1 if self.dest > self.origin else -1


@dataclass
class Car:
    cid: int
    pos: int
    dir: int = 0                     # -1, 0, +1
    moving: bool = False
    onboard: list = field(default_factory=list)
    car_calls: set = field(default_factory=set)
    park_target: int | None = None
    busy_until: float = 0.0
    n_runs: int = 0
    n_stops: int = 0
    floors_travelled: int = 0


class Building:
    def __init__(self, n_floors: int, n_cars: int, kin: Kinematics, parking: str = "lobby",
                 capacity: int = 10, park_floor: int | None = None):
        self.N, self.K, self.kin, self.parking, self.capacity = n_floors, n_cars, kin, parking, capacity
        if park_floor is None:
            park_floor = 1 if parking == "lobby" else int(round((n_floors + 1) / 2))
        self.park_floor = park_floor
        self.cars = [Car(c, park_floor) for c in range(n_cars)]
        self.waiting: dict[int, list[Passenger]] = {f: [] for f in range(1, n_floors + 1)}
        self.events: list = []
        self._seq = 0
        self.t = 0.0
        self.done: list[Passenger] = []

    # ------------------------------------------------------------------ #
    # Event plumbing
    # ------------------------------------------------------------------ #
    def push(self, t: float, kind: str, payload):
        self._seq += 1
        heapq.heappush(self.events, (t, self._seq, kind, payload))

    def add_passenger(self, p: Passenger):
        self.push(p.t_arrive, "arrive", p)

    def run(self, until: float):
        while self.events and self.events[0][0] <= until:
            t, _, kind, payload = heapq.heappop(self.events)
            self.t = t
            if kind == "arrive":
                self._on_arrive(payload)
            elif kind == "car":
                self._on_car(payload)
        self.t = until

    # ------------------------------------------------------------------ #
    # Calls
    # ------------------------------------------------------------------ #
    def _pending_hall(self):
        return [f for f, q in self.waiting.items() if q]

    def _calls_ahead(self, car: Car) -> bool:
        if car.dir == 0:
            return False
        for f in car.car_calls:
            if (f - car.pos) * car.dir > 0:
                return True
        for f in self._pending_hall():
            if (f - car.pos) * car.dir > 0:
                return True
        return False

    def _calls_behind(self, car: Car) -> bool:
        for f in car.car_calls:
            if (f - car.pos) * car.dir < 0:
                return True
        for f in self._pending_hall():
            if (f - car.pos) * car.dir < 0:
                return True
        return False

    def _on_arrive(self, p: Passenger):
        self.waiting[p.origin].append(p)
        # Wake the nearest idle car (nearest-car dispatch).
        idle = [c for c in self.cars if c.dir == 0 and not c.moving]
        if idle:
            car = min(idle, key=lambda c: (abs(c.pos - p.origin), c.cid))
            self._dispatch(car, p.origin)
        # else: a moving car will pick it up in passing or after reversing.

    def _dispatch(self, car: Car, floor: int):
        if car.pos == floor:
            self._stop(car)                     # doors open right away
        else:
            car.dir = 1 if floor > car.pos else -1
            self._depart(car)

    # ------------------------------------------------------------------ #
    # Car dynamics
    # ------------------------------------------------------------------ #
    def _depart(self, car: Car):
        """Start a run from a standstill: first floor costs cruise + start penalty."""
        car.moving = True
        car.n_runs += 1
        self.push(self.t + self.kin.floor_h / self.kin.v + self.kin.v / self.kin.a, "car", (car.cid, "step"))

    def _on_car(self, payload):
        cid, what = payload
        car = self.cars[cid]
        if what == "step":
            car.pos += car.dir
            car.floors_travelled += 1
            if self._should_stop(car):
                car.moving = False
                self._stop(car)
            else:
                self.push(self.t + self.kin.floor_h / self.kin.v, "car", (cid, "step"))
        elif what == "resume":
            self._decide(car)

    def _should_stop(self, car: Car) -> bool:
        f = car.pos
        if f in car.car_calls:
            return True
        if f == 1 or f == self.N:
            return bool(self.waiting[f]) or not self._calls_ahead(car)
        if any(q.direction == car.dir for q in self.waiting[f]) and len(car.onboard) < self.capacity:
            return True
        if self.waiting[f] and not self._calls_ahead(car):
            return True                          # reversal floor: take them
        if car.park_target == f and not car.onboard and not car.car_calls:
            return True
        return False

    def _stop(self, car: Car):
        """Handle a stop at car.pos: alight, board, dwell, then decide."""
        f = car.pos
        car.n_stops += 1
        n_transfer = 0
        stay = []
        for p in car.onboard:
            if p.dest == f:
                p.t_alight = self.t
                self.done.append(p); n_transfer += 1
            else:
                stay.append(p); p.stops_on_board += 1
        car.onboard = stay
        car.car_calls.discard(f)
        # Direction for boarding: keep it if calls ahead, else the waiting majority.
        if car.dir == 0 or not (car.car_calls or self._calls_ahead(car)):
            q = self.waiting[f]
            if q:
                car.dir = q[0].direction
        boarded, rest = [], []
        for p in self.waiting[f]:
            if p.direction == car.dir and len(car.onboard) + len(boarded) < self.capacity:
                boarded.append(p)
            else:
                rest.append(p)
        self.waiting[f] = rest
        for p in boarded:
            p.t_board = self.t; p.car = car.cid
            car.onboard.append(p); car.car_calls.add(p.dest); n_transfer += 1
        if n_transfer > 0 or car.park_target == f:
            dwell = self.kin.stop(max(n_transfer, 1)) if n_transfer else self.kin.t_door
        else:
            dwell = self.kin.t_door
        if car.park_target == f:
            car.park_target = None
        self.push(self.t + dwell, "car", (car.cid, "resume"))

    def _decide(self, car: Car):
        """After a dwell: continue, reverse, park, or idle."""
        if car.dir == 0:
            car.dir = 1
        if self._calls_ahead(car):
            self._depart(car); return
        if self._calls_behind(car):
            car.dir = -car.dir; self._depart(car); return
        # Any pending hall call anywhere (e.g. left behind by capacity)?
        pend = self._pending_hall()
        if pend:
            f = min(pend, key=lambda x: abs(x - car.pos))
            if f != car.pos:
                car.dir = 1 if f > car.pos else -1; self._depart(car); return
            self._stop(car); return
        # Nothing to do: park or idle.
        if self.parking in ("lobby", "middle") and car.pos != self.park_floor:
            car.park_target = self.park_floor
            car.dir = 1 if self.park_floor > car.pos else -1
            self._depart(car); return
        car.dir = 0; car.moving = False; car.park_target = None

    # ------------------------------------------------------------------ #
    def car_positions(self) -> list[int]:
        return [c.pos for c in self.cars]


def poisson_background(rate_per_min: float, horizon_s: float, n_floors: int, alpha: float,
                       rng: np.random.Generator, pid0: int = 0) -> list[Passenger]:
    """Background passengers: Poisson arrivals; lobby trips (up from or down to
    floor 1) with probability alpha, otherwise between two floors >= 2."""
    out = []
    if rate_per_min <= 0:
        return out
    n = rng.poisson(rate_per_min * horizon_s / 60.0)
    times = np.sort(rng.uniform(0, horizon_s, n))
    floors = np.arange(2, n_floors + 1)
    for i, t in enumerate(times):
        if rng.random() < alpha or len(floors) < 2:
            f = int(rng.choice(floors))
            o, d = (1, f) if rng.random() < 0.5 else (f, 1)
        else:
            o, d = rng.choice(floors, size=2, replace=False)
        out.append(Passenger(pid0 + i, float(t), int(o), int(d), kind="bg"))
    return out
