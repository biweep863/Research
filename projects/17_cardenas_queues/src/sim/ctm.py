"""Cell Transmission Model (Daganzo 1994, 1995) for one directional carriageway.

The carriageway is a line of cells of length dx = vf * dt.  Express and lateral
(frontage) lanes are aggregated into one cross-section whose boundary capacity
is s0 * (lanes_exp * g_exp + lanes_lat * g_lat) * friction, reduced by
`cap_drop` while the queue sits behind it (the two-capacity phenomenon,
Hall & Agyemang-Duah 1991; Chung et al. 2007).  Feeder approaches
are point queues ("ramps") that merge at the upstream boundary of a cell with
Daganzo's priority merge; exits are fixed split fractions at the downstream
boundary with a partially FIFO diverge: with fifo = 1 a blocked continuation
also holds back exiting vehicles (Daganzo 1995), with fifo = 0 exiting
vehicles leave freely (they use the outer lane), in between a mix.

All state is in vehicles, rates in veh/h, time step dt in hours.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CTMParams:
    vf_kmh: float = 60.0         # free-flow speed of an urban arterial
    w_kmh: float = 20.0          # backward wave speed
    kj_vpkm: float = 150.0       # jam density per lane
    s0: float = 1900.0           # saturation flow per lane (veh/h)
    dt_s: float = 6.0            # time step -> dx = vf dt = 100 m at 60 km/h
    fifo: float = 1.0            # 1 = FIFO diverge, 0 = exits never blocked
    cap_drop: float = 0.0        # discharge loss when the cell upstream of a boundary is queued

    @property
    def dt_h(self) -> float:
        return self.dt_s / 3600.0

    @property
    def dx_km(self) -> float:
        return self.vf_kmh * self.dt_h

    @property
    def kc_vpkm(self) -> float:
        """Critical density per lane of the triangular fundamental diagram."""
        return self.s0 / self.vf_kmh


@dataclass
class Carriageway:
    """Geometry and control of one direction, in travel order (cell 0 = upstream)."""
    lanes: np.ndarray            # total lanes per cell (express + lateral)
    cap_vph: np.ndarray          # downstream-boundary capacity per cell (signals, friction applied)
    beta: np.ndarray             # exit fraction at downstream boundary of each cell
    ramp_cell: np.ndarray        # cell each ramp enters (upstream boundary)
    ramp_cap_vph: np.ndarray     # ramp discharge capacity
    ramp_lanes: np.ndarray       # for merge priority
    sink_cap_vph: float = np.inf
    meta: dict = field(default_factory=dict)


@dataclass
class Metering:
    """ALINEA (Papageorgiou 1991) local ramp metering on a subset of ramps."""
    ramps: np.ndarray            # indices of metered ramps
    gain_vph: float = 35.0       # K_R in veh/h per (veh/km/lane) of density error (~70 veh/h per % occupancy)
    target_frac: float = 0.9     # target density as a fraction of critical
    r_min_frac: float = 0.15     # minimum metering rate as a fraction of ramp capacity


def simulate(cw: Carriageway, ramp_demand_vph: np.ndarray, p: CTMParams,
             metering: Metering | None = None, record_every: int = 10) -> dict:
    """Run the CTM.

    ramp_demand_vph: (T, R) demand of each ramp per step (ramp 0 is usually the
    upstream boundary entering cell 0).  Returns delays (veh h), recorded
    density (veh/km/lane) and ramp queues, outflow and conservation terms.
    """
    T, R = ramp_demand_vph.shape
    n_c = len(cw.lanes)
    dt = p.dt_h
    dx = p.dx_km
    N = p.kj_vpkm * cw.lanes * dx                     # jam storage per cell (veh)
    Qcell = p.s0 * cw.lanes * dt                      # cell capacity (no signal) per step
    Qb = cw.cap_vph * dt                              # boundary capacity per step
    wv = p.w_kmh / p.vf_kmh
    ramp_cap = cw.ramp_cap_vph * dt
    beta = np.clip(cw.beta, 0.0, 0.9)

    # Group ramps by cell once; mainline merge priority from lanes.
    ramp_cell = cw.ramp_cell.astype(int)
    lanes_in = np.zeros(n_c)
    np.add.at(lanes_in, ramp_cell, cw.ramp_lanes)
    p_main = np.where(lanes_in > 0, cw.lanes / (cw.lanes + lanes_in), 1.0)
    p_main[0] = 0.0                                   # nothing upstream of cell 0

    n = np.zeros(n_c)
    q = np.zeros(R)
    meter_rate = None
    if metering is not None and len(metering.ramps):
        meter_rate = cw.ramp_cap_vph[metering.ramps].astype(float).copy()   # veh/h
    kc = p.kc_vpkm

    delay_main = 0.0
    delay_ramp = 0.0
    delay_main_cell = np.zeros(n_c)
    delay_ramp_each = np.zeros(R)
    total_in = total_out = total_exit = 0.0
    rec_k, rec_q = [], []

    for t in range(T):
        arr = ramp_demand_vph[t] * dt
        queued = n > kc * cw.lanes * dx
        S = np.minimum(n, Qb * np.where(queued, 1 - p.cap_drop, 1.0))   # dx = vf dt -> v n = n
        Rcv = np.minimum(Qcell, wv * (N - n))
        S_cont = (1 - beta) * S

        Sr = np.minimum(q + arr, ramp_cap)
        if meter_rate is not None:
            # ALINEA on density of the receiving cell.
            cells = ramp_cell[metering.ramps]
            k = n[cells] / (cw.lanes[cells] * dx)
            err = metering.target_frac * kc - k
            cap_m = cw.ramp_cap_vph[metering.ramps]
            meter_rate = np.clip(meter_rate + metering.gain_vph * err, metering.r_min_frac * cap_m, cap_m)
            Sr[metering.ramps] = np.minimum(Sr[metering.ramps], meter_rate * dt)
        Sr_cell = np.zeros(n_c)
        np.add.at(Sr_cell, ramp_cell, Sr)

        # Mainline sending into cell j comes from cell j-1.
        S_up = np.concatenate([[0.0], S_cont[:-1]])
        free = S_up + Sr_cell <= Rcv
        y_main = np.where(free, S_up,
                          np.median(np.stack([S_up, Rcv - Sr_cell, p_main * Rcv]), axis=0))
        y_main = np.clip(y_main, 0.0, S_up)
        y_rcell = np.where(free, Sr_cell, np.maximum(Rcv - y_main, 0.0))
        y_rcell = np.minimum(y_rcell, Sr_cell)
        share = np.divide(y_rcell, Sr_cell, out=np.zeros(n_c), where=Sr_cell > 0)
        y_r = Sr * share[ramp_cell]

        # Last cell discharges to the sink.
        last_cont = min(S_cont[-1], cw.sink_cap_vph * dt)
        cont_out = np.concatenate([y_main[1:], [last_cont]])
        # Exits: FIFO share is held back in proportion to the blocked continuation.
        pass_frac = np.divide(cont_out, S_cont, out=np.ones(n_c), where=S_cont > 0)
        exits = beta * S * (p.fifo * pass_frac + (1 - p.fifo))
        out_total = cont_out + exits

        delay_main_cell += (n - out_total) * dt
        n = n + y_main + y_rcell - out_total
        q = q + arr - y_r
        delay_ramp_each += q * dt

        total_in += arr.sum()
        total_out += last_cont
        total_exit += exits.sum()
        if t % record_every == 0:
            rec_k.append(n / (cw.lanes * dx))
            rec_q.append(q.copy())

    delay_main = float(delay_main_cell.sum())
    delay_ramp = float(delay_ramp_each.sum())
    return {
        "delay_main_vh": delay_main,
        "delay_ramp_vh": delay_ramp,
        "delay_total_vh": delay_main + delay_ramp,
        "delay_main_cell_vh": delay_main_cell,
        "delay_ramp_vh_each": delay_ramp_each,
        "density": np.array(rec_k),
        "ramp_queue": np.array(rec_q),
        "stored_end": float(n.sum() + q.sum()),
        "total_in": total_in, "total_out": total_out, "total_exit": total_exit,
        "record_dt_h": dt * record_every,
    }


def active_bottlenecks(density: np.ndarray, kc: float, lo: float = 1.0, hi: float = 1.0) -> np.ndarray:
    """Fraction of recorded time each boundary i -> i+1 is an active bottleneck:
    cell i congested (k > lo * kc) while cell i+1 is uncongested (k < hi * kc)."""
    up = density[:, :-1] > lo * kc
    down = density[:, 1:] < hi * kc
    return np.concatenate([(up & down).mean(axis=0), [0.0]])
