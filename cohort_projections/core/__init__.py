"""
Core cohort component projection modules.

This package contains the mathematical engine for population projections
using the cohort-component method.
"""

from .cohort_component import CohortComponentProjection
from .fertility import apply_fertility_scenario, calculate_births, validate_fertility_rates
from .migration import (
    apply_migration,
    apply_migration_scenario,
    combine_domestic_international,
    distribute_international_migration,
    validate_migration_data,
)
from .mortality import (
    apply_mortality_improvement,
    apply_survival_rates,
    calculate_life_expectancy,
    validate_survival_rates,
)

__all__ = [
    # Main projection engine
    "CohortComponentProjection",
    # Fertility functions
    "calculate_births",
    "apply_fertility_scenario",
    "validate_fertility_rates",
    # Mortality functions
    "apply_survival_rates",
    "apply_mortality_improvement",
    "validate_survival_rates",
    "calculate_life_expectancy",
    # Migration functions
    "apply_migration",
    "apply_migration_scenario",
    "validate_migration_data",
    "distribute_international_migration",
    "combine_domestic_international",
]
