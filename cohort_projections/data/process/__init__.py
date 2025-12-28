"""
Data processing module for cohort projections.

Processes raw demographic data (Census, SEER, IRS, etc.) into standardized
formats required by the projection engine.

Modules:
    - base_population: Process Census population data into cohort matrices
    - fertility_rates: Process SEER/NVSS fertility rates
    - survival_rates: Process SEER/CDC life tables into survival rates
    - migration_rates: Process IRS/ACS migration data into net migration by cohort
"""

# Base population processing
from .base_population import (
    RACE_ETHNICITY_MAP,
    create_cohort_matrix,
    get_cohort_summary,
    harmonize_race_categories,
    process_county_population,
    process_place_population,
    process_state_population,
    validate_cohort_matrix,
)

# Fertility rates processing
from .fertility_rates import (
    SEER_RACE_ETHNICITY_MAP,
    calculate_average_fertility_rates,
    create_fertility_rate_table,
    harmonize_fertility_race_categories,
    load_seer_fertility_data,
    process_fertility_rates,
    validate_fertility_rates,
)

# Migration rates processing
from .migration_rates import (
    MIGRATION_RACE_MAP,
    calculate_net_migration,
    combine_domestic_international_migration,
    create_migration_rate_table,
    distribute_migration_by_age,
    distribute_migration_by_race,
    distribute_migration_by_sex,
    get_standard_age_migration_pattern,
    load_international_migration_data,
    load_irs_migration_data,
    process_migration_rates,
    validate_migration_data,
)

# Survival rates processing
from .survival_rates import (
    SEER_MORTALITY_RACE_MAP,
    apply_mortality_improvement,
    calculate_life_expectancy,
    calculate_survival_rates_from_life_table,
    create_survival_rate_table,
    harmonize_mortality_race_categories,
    load_life_table_data,
    process_survival_rates,
    validate_survival_rates,
)

__all__ = [
    # Base population
    "harmonize_race_categories",
    "create_cohort_matrix",
    "validate_cohort_matrix",
    "process_state_population",
    "process_county_population",
    "process_place_population",
    "get_cohort_summary",
    "RACE_ETHNICITY_MAP",
    # Fertility rates
    "load_seer_fertility_data",
    "harmonize_fertility_race_categories",
    "calculate_average_fertility_rates",
    "create_fertility_rate_table",
    "validate_fertility_rates",
    "process_fertility_rates",
    "SEER_RACE_ETHNICITY_MAP",
    # Survival rates
    "load_life_table_data",
    "harmonize_mortality_race_categories",
    "calculate_survival_rates_from_life_table",
    "apply_mortality_improvement",
    "create_survival_rate_table",
    "validate_survival_rates",
    "calculate_life_expectancy",
    "process_survival_rates",
    "SEER_MORTALITY_RACE_MAP",
    # Migration rates
    "load_irs_migration_data",
    "load_international_migration_data",
    "get_standard_age_migration_pattern",
    "distribute_migration_by_age",
    "distribute_migration_by_sex",
    "distribute_migration_by_race",
    "calculate_net_migration",
    "combine_domestic_international_migration",
    "create_migration_rate_table",
    "validate_migration_data",
    "process_migration_rates",
    "MIGRATION_RACE_MAP",
]
