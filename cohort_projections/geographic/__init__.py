"""
Geographic module for multi-geography cohort projections.

This module provides functionality to run cohort-component population projections
for multiple geographies (state, counties, places) with support for parallel
processing and hierarchical aggregation.

Key Components:
    - geography_loader: Load FIPS codes and geographic reference data
    - multi_geography: Orchestrate projections across geographies

Main Functions:
    - load_nd_counties(): Load North Dakota county reference data
    - load_nd_places(): Load North Dakota place reference data
    - load_geography_list(): Get list of FIPS codes to project
    - get_geography_name(): Get name for a FIPS code
    - get_place_to_county_mapping(): Map places to counties
    - run_single_geography_projection(): Run projection for one geography
    - run_multi_geography_projections(): Run projections for multiple geographies
    - aggregate_to_county(): Aggregate places to counties
    - aggregate_to_state(): Aggregate counties to state
    - validate_aggregation(): Validate aggregated results

Example Usage:
    ```python
    from cohort_projections.geographic import (
        load_geography_list,
        run_multi_geography_projections
    )

    # Load list of counties to project
    counties = load_geography_list('county')

    # Run projections for all counties
    results = run_multi_geography_projections(
        level='county',
        base_population_by_geography=county_populations,
        fertility_rates=nd_fertility_rates,
        survival_rates=nd_survival_rates,
        migration_rates_by_geography=county_migration_rates,
        parallel=True
    )

    # Access results
    print(f"Projected {len(results['results'])} counties")
    print(results['summary'])
    ```
"""

# Geography loader functions
from .geography_loader import (
    load_nd_counties,
    load_nd_places,
    get_place_to_county_mapping,
    load_geography_list,
    get_geography_name
)

# Multi-geography projection functions
from .multi_geography import (
    run_single_geography_projection,
    run_multi_geography_projections,
    aggregate_to_county,
    aggregate_to_state,
    validate_aggregation
)

__all__ = [
    # Geography loader
    'load_nd_counties',
    'load_nd_places',
    'get_place_to_county_mapping',
    'load_geography_list',
    'get_geography_name',
    # Multi-geography projections
    'run_single_geography_projection',
    'run_multi_geography_projections',
    'aggregate_to_county',
    'aggregate_to_state',
    'validate_aggregation',
]
