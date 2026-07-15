"""Space-based solar power (SBSP): the "layer in orbit" the user asked about.

Compares a ground panel with an orbital collector that beams power down. The
orbital collector's advantage is NOT mainly "catching light before the
atmosphere" -- it is (i) full solar constant instead of the day/night- and
weather-reduced surface value, and (ii) a ~99% capacity factor instead of
~25%. That advantage is then eroded by the conversion + transmission chain,
which we lay out term by term.
"""
from __future__ import annotations

from dataclasses import dataclass

from .insolation import S0
from .landuse import (
    DESERT_GHI,
    PANEL_EFFICIENCY,
    PERFORMANCE_RATIO,
    WORLD_ELECTRICITY_POWER,
    panel_power_density,
)

# --- Orbital collector chain ---------------------------------------------- #
ORBIT_AVAILABILITY = 0.99     # fraction of time in sunlight at GEO
SPACE_CELL_EFFICIENCY = 0.30  # space-grade PV efficiency
DC_TO_RF = 0.80               # DC to microwave conversion
ATM_TRANSMISSION = 0.95       # microwave transmission through the atmosphere
RECTENNA_EFFICIENCY = 0.85    # microwave to DC at the ground rectenna


@dataclass(frozen=True)
class ChainStep:
    label: str
    power: float              # W/m^2 of collector remaining after this step


def sbsp_chain() -> list[ChainStep]:
    """Delivered power per m^2 of orbital collector, step by step (W/m^2)."""
    p0 = S0 * ORBIT_AVAILABILITY
    p1 = p0 * SPACE_CELL_EFFICIENCY
    p2 = p1 * DC_TO_RF
    p3 = p2 * ATM_TRANSMISSION
    p4 = p3 * RECTENNA_EFFICIENCY
    return [
        ChainStep("collected (S0 x availability)", p0),
        ChainStep("after PV conversion", p1),
        ChainStep("after DC->RF", p2),
        ChainStep("after atmospheric transmission", p3),
        ChainStep("delivered to grid (after rectenna)", p4),
    ]


def sbsp_delivered_per_m2() -> float:
    """Grid power delivered per m^2 of orbital collector (W/m^2)."""
    return sbsp_chain()[-1].power


@dataclass(frozen=True)
class SbspComparison:
    ground_per_m2: float
    space_per_m2: float
    ratio: float
    ground_capacity_factor: float
    space_capacity_factor: float
    ground_area_for_world: float
    space_area_for_world: float


def compare_ground_vs_space() -> SbspComparison:
    ground = panel_power_density(DESERT_GHI)
    space = sbsp_delivered_per_m2()
    ground_cf = DESERT_GHI / 1000.0   # relative to 1 kW/m^2 nameplate condition
    return SbspComparison(
        ground_per_m2=ground,
        space_per_m2=space,
        ratio=space / ground,
        ground_capacity_factor=ground_cf,
        space_capacity_factor=ORBIT_AVAILABILITY,
        ground_area_for_world=WORLD_ELECTRICITY_POWER / ground,
        space_area_for_world=WORLD_ELECTRICITY_POWER / space,
    )
