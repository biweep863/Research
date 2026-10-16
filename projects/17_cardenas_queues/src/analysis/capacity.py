"""Capacity and signalised-delay formulas (veh/h, s).

Saturation flow per lane (HCM base 1900 pc/h/ln), lane-group capacity with a
green ratio, Webster's (1958) uniform + random delay and the HCM incremental
delay term, which stays finite when the degree of saturation x >= 1.
"""
from __future__ import annotations

import math


def lane_group_capacity(lanes: float, g_over_c: float = 1.0, s0: float = 1900.0) -> float:
    """Capacity of a lane group: lanes * s0 * g/C."""
    return lanes * s0 * g_over_c


def corridor_capacity(lanes_express: float, lanes_lateral: float, g_express: float,
                      g_lateral: float, s0: float = 1900.0) -> float:
    """Capacity of a cross-section with express lanes and frontage (lateral) lanes,
    each with its own green ratio (1.0 where it is not signalised)."""
    return s0 * (lanes_express * g_express + lanes_lateral * g_lateral)


def webster_delay(cycle_s: float, g_over_c: float, q_vph: float, cap_vph: float) -> float:
    """Webster's average delay per vehicle (s); inf when x >= 1."""
    x = q_vph / cap_vph
    if x >= 1:
        return math.inf
    q = q_vph / 3600.0
    d1 = cycle_s * (1 - g_over_c) ** 2 / (2 * (1 - g_over_c * x))
    d2 = x ** 2 / (2 * q * (1 - x)) if q > 0 else 0.0
    return 0.9 * (d1 + d2)


def hcm_delay(cycle_s: float, g_over_c: float, q_vph: float, cap_vph: float,
              period_h: float = 0.25) -> float:
    """HCM-2000 control delay (s/veh) = uniform d1 + incremental d2, valid for x >= 1.

    d1 = 0.5 C (1-g/C)^2 / (1 - min(1,x) g/C)
    d2 = 900 T [(x-1) + sqrt((x-1)^2 + 4x/(cap T))]   (k=0.5, I=1)
    """
    x = q_vph / cap_vph
    d1 = 0.5 * cycle_s * (1 - g_over_c) ** 2 / (1 - min(1.0, x) * g_over_c)
    T = period_h
    d2 = 900 * T * ((x - 1) + math.sqrt((x - 1) ** 2 + 4 * x / (cap_vph * T)))
    return d1 + d2


def minor_stream_capacity(lanes: float, conflicting_vph: float, tc_s: float = 6.2,
                          tf_s: float = 3.3) -> float:
    """HCM gap-acceptance capacity of an unsignalised minor approach (veh/h):
    c = v_c exp(-v_c tc/3600) / (1 - exp(-v_c tf/3600)), per lane."""
    vc = max(conflicting_vph, 1e-6)
    c = vc * math.exp(-vc * tc_s / 3600) / (1 - math.exp(-vc * tf_s / 3600))
    return lanes * c
