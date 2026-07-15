"""Land availability and demand: what can we deploy, and how much is needed.

Turns the surface-irradiance resource into deployable-panel energy under a
"do no ecological harm" land filter (deserts + rooftops only, protected land
excluded), and compares it against real-world electricity/primary-energy
demand. All quantities are annual-mean powers in watts unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass

from .insolation import earth_surface_area

# --- Land cover (published global fractions) ------------------------------ #
LAND_FRACTION = 0.292                 # fraction of Earth surface that is land
LAND_COVER = {                        # fractions of land area (sum ~ 1.0)
    "forest": 0.31,
    "grassland_shrub": 0.28,
    "desert_barren": 0.19,
    "cropland": 0.12,
    "ice_tundra_rock": 0.09,
    "urban": 0.01,
}

# "Deploy without harming flora and fauna": only a conservative slice of
# already-barren desert (excluding dunes, protected reserves, and remote
# terrain) plus rooftops on built-up land.
USABLE_DESERT_FRACTION = 0.20
ROOFTOP_OF_URBAN = 0.30

# --- Panel performance ---------------------------------------------------- #
PANEL_EFFICIENCY = 0.20               # module efficiency
PERFORMANCE_RATIO = 0.80              # system losses (inverter, wiring, soiling)
DESERT_GHI = 250.0                    # annual-mean surface GHI at good sites, W/m^2

# --- Demand (published, approximate present-day) -------------------------- #
WORLD_PRIMARY_POWER = 1.8e13          # ~18 TW primary energy
WORLD_ELECTRICITY_POWER = 3.0e12      # ~3 TW electricity
WORLD_POPULATION = 8.0e9
N_COUNTRIES = 195
CITY_POPULATION = 1.0e6               # a reference "one-million city"


@dataclass(frozen=True)
class LandBudget:
    land_area: float
    deployable_area: float
    deployable_power: float           # annual-mean W delivered
    frac_of_primary: float
    frac_of_electricity: float
    area_for_world_electricity: float  # m^2 of panel to meet world electricity
    n_million_cities: float
    n_country_equivalents: float


def land_area() -> float:
    return earth_surface_area() * LAND_FRACTION


def deployable_area() -> float:
    """Panel-eligible land area (m^2): usable desert + urban rooftops."""
    la = land_area()
    return la * (LAND_COVER["desert_barren"] * USABLE_DESERT_FRACTION
                 + LAND_COVER["urban"] * ROOFTOP_OF_URBAN)


def panel_power_density(ghi: float = DESERT_GHI) -> float:
    """Annual-mean delivered power per m^2 of panel (W/m^2)."""
    return ghi * PANEL_EFFICIENCY * PERFORMANCE_RATIO


def land_budget() -> LandBudget:
    la = land_area()
    area = deployable_area()
    power = area * panel_power_density()
    per_capita = WORLD_ELECTRICITY_POWER / WORLD_POPULATION
    per_city = CITY_POPULATION * per_capita
    per_country = WORLD_ELECTRICITY_POWER / N_COUNTRIES
    return LandBudget(
        land_area=la,
        deployable_area=area,
        deployable_power=power,
        frac_of_primary=power / WORLD_PRIMARY_POWER,
        frac_of_electricity=power / WORLD_ELECTRICITY_POWER,
        area_for_world_electricity=WORLD_ELECTRICITY_POWER / panel_power_density(),
        n_million_cities=power / per_city,
        n_country_equivalents=power / per_country,
    )
