"""
Data loading module for cohort projections.

Provides functions to load processed data files into the formats required
by the projection engine and multi-geography orchestrator.

Modules:
    - base_population_loader: Load base population data for projections
"""

from .base_population_loader import (
    load_base_population_for_all_counties,
    load_base_population_for_county,
    load_state_age_sex_race_distribution,
)

__all__ = [
    "load_base_population_for_county",
    "load_base_population_for_all_counties",
    "load_state_age_sex_race_distribution",
]
