"""
Data loading module for cohort projections.

Provides functions to load processed data files into the formats required
by the projection engine and multi-geography orchestrator.

Modules:
    - base_population_loader: Load base population data for projections
    - census_age_sex_population: Load Census/PEP population by age and sex
"""

from .base_population_loader import (
    load_base_population_for_all_counties,
    load_base_population_for_county,
    load_state_age_sex_race_distribution,
)
from .census_age_sex_population import (
    AGE_GROUP_LABELS,
    load_census_2000_county_age_sex,
    load_census_2020_base_population,
    load_pep_2010_2019_county_age_sex,
    load_pep_2020_2024_county_age_sex,
    load_pep_2020_intercensal_county_age_sex,
)

__all__ = [
    "load_base_population_for_county",
    "load_base_population_for_all_counties",
    "load_state_age_sex_race_distribution",
    # Census/PEP population loaders
    "AGE_GROUP_LABELS",
    "load_census_2000_county_age_sex",
    "load_pep_2010_2019_county_age_sex",
    "load_pep_2020_intercensal_county_age_sex",
    "load_pep_2020_2024_county_age_sex",
    "load_census_2020_base_population",
]
