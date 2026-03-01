"""
Data processing module for cohort projections.

Processes raw demographic data (Census, SEER, IRS, etc.) into standardized
formats required by the projection engine.

Modules:
    - base_population: Process Census population data into cohort matrices
    - fertility_rates: Process SEER/NVSS fertility rates
    - survival_rates: Process SEER/CDC life tables into survival rates
    - migration_rates: Process IRS/ACS migration data into net migration by cohort
    - mortality_improvement: Census Bureau NP2023 ND-adjusted survival projections
    - pep_regime_analysis: Regime-aware migration analysis from Census PEP data
    - place_housing_unit_projection: Housing-unit method for place projections (ADR-060)
"""

from importlib import import_module
from types import ModuleType

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
    process_pep_migration_rates,
    validate_migration_data,
)

# Mortality improvement (Census Bureau NP2023 ND-adjusted)
from .mortality_improvement import (
    build_nd_adjusted_survival_projections,
    compute_nd_adjustment_factors,
    load_census_survival_projections,
    load_nd_baseline_survival,
    run_mortality_improvement_pipeline,
)

# PEP regime analysis
from .pep_regime_analysis import (
    DEFAULT_DAMPENING,
    DEFAULT_REGIME_WEIGHTS,
    METRO_COUNTIES,
    MIGRATION_REGIMES,
    OIL_COUNTIES,
    calculate_regime_averages,
    calculate_regime_weighted_average,
    classify_counties,
    generate_regime_analysis_report,
    load_pep_preferred_estimates,
)

# Place backtesting (PP-003 Phase 3)
from .place_backtest import (
    compute_per_place_metrics,
    compute_tier_aggregates,
    compute_variant_score,
    run_single_variant,
    select_winner,
)

# Place housing-unit method (PP-005 / ADR-060)
from .place_housing_unit_projection import (
    cross_validate_with_share_trending,
    load_housing_data,
    project_population_from_hu,
    project_pph,
    run_housing_unit_projections,
    trend_housing_units,
)

# Place projection orchestration (PP-003 Phase 2)
from .place_projection_orchestrator import (
    allocate_age_sex_detail,
    run_place_projections,
    validate_state_scenario_ordering,
    write_place_outputs,
    write_place_qa_artifacts,
    write_places_summary,
    write_run_level_metadata,
)

# Place share trending (PP-003 Phase 2)
from .place_share_trending import (
    BALANCE_KEY,
    DEFAULT_EPSILON,
    DEFAULT_LAMBDA_DECAY,
    DEFAULT_RECONCILIATION_FLAG_THRESHOLD,
    ReconciliationResult,
    apply_cap_and_redistribute,
    apply_proportional_rescaling,
    compute_recency_weights,
    fit_share_trend,
    inverse_logit,
    logit_transform,
    project_shares,
    reconcile_county_shares,
    trend_all_places_in_county,
)

# Rolling-origin backtesting (PP-005 WS-A)
from .rolling_origin_backtest import (
    aggregate_rolling_metrics,
    build_per_window_summary,
    generate_rolling_windows,
    run_rolling_origin_backtest,
    select_rolling_winner,
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


def load_example_usage_module() -> ModuleType:
    """Lazily load the base population example module."""
    return import_module("cohort_projections.data.process.example_usage")


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
    "process_pep_migration_rates",
    "MIGRATION_RACE_MAP",
    # Mortality improvement
    "load_census_survival_projections",
    "load_nd_baseline_survival",
    "compute_nd_adjustment_factors",
    "build_nd_adjusted_survival_projections",
    "run_mortality_improvement_pipeline",
    # PEP regime analysis
    "OIL_COUNTIES",
    "METRO_COUNTIES",
    "MIGRATION_REGIMES",
    "DEFAULT_REGIME_WEIGHTS",
    "DEFAULT_DAMPENING",
    "classify_counties",
    "calculate_regime_averages",
    "calculate_regime_weighted_average",
    "load_pep_preferred_estimates",
    "generate_regime_analysis_report",
    # Place backtesting
    "run_single_variant",
    "compute_per_place_metrics",
    "compute_tier_aggregates",
    "compute_variant_score",
    "select_winner",
    # Place share trending
    "BALANCE_KEY",
    "DEFAULT_EPSILON",
    "DEFAULT_LAMBDA_DECAY",
    "DEFAULT_RECONCILIATION_FLAG_THRESHOLD",
    "ReconciliationResult",
    "logit_transform",
    "inverse_logit",
    "compute_recency_weights",
    "fit_share_trend",
    "project_shares",
    "apply_proportional_rescaling",
    "apply_cap_and_redistribute",
    "reconcile_county_shares",
    "trend_all_places_in_county",
    # Rolling-origin backtesting
    "aggregate_rolling_metrics",
    "build_per_window_summary",
    "generate_rolling_windows",
    "run_rolling_origin_backtest",
    "select_rolling_winner",
    # Place housing-unit method
    "load_housing_data",
    "trend_housing_units",
    "project_pph",
    "project_population_from_hu",
    "run_housing_unit_projections",
    "cross_validate_with_share_trending",
    # Place projection orchestration
    "allocate_age_sex_detail",
    "run_place_projections",
    "validate_state_scenario_ordering",
    "write_place_qa_artifacts",
    "write_place_outputs",
    "write_places_summary",
    "write_run_level_metadata",
    # Example entrypoint
    "load_example_usage_module",
]
