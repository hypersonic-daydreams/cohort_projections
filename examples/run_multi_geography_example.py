"""
Example: Run Multi-Geography Projections for North Dakota

Demonstrates how to use the geographic module to run cohort-component projections
for multiple geographies (counties and places) with parallel processing.

This example shows:
1. Loading geographic reference data
2. Preparing base population by geography
3. Running multi-geography projections (serial and parallel)
4. Aggregating places to counties and counties to state
5. Validating hierarchical aggregation
6. Analyzing results

Requirements:
- Processed rate files (fertility, survival, migration)
- Base population by geography
- See data/process/ modules for rate generation
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.geographic import (  # noqa: E402
    aggregate_to_county,
    aggregate_to_state,
    get_geography_name,
    load_nd_counties,
    load_nd_places,
    run_multi_geography_projections,
    run_single_geography_projection,
    validate_aggregation,
)
from cohort_projections.utils.config_loader import load_projection_config  # noqa: E402
from cohort_projections.utils.logger import get_logger_from_config  # noqa: E402

logger = get_logger_from_config(__name__)


def create_synthetic_data():
    """
    Create synthetic data for demonstration.

    In production, load actual processed data from:
    - data/processed/population/base_population.parquet
    - data/processed/fertility/fertility_rates.parquet
    - data/processed/survival/survival_rates.parquet
    - data/processed/migration/migration_rates.parquet
    """
    logger.info("Creating synthetic data for demonstration...")

    # Configuration
    config = load_projection_config()
    base_year = config["project"]["base_year"]

    # Ages, sexes, races from config
    ages = list(range(91))  # 0-90
    sexes = ["Male", "Female"]
    races = config["demographics"]["race_ethnicity"]["categories"]

    # 1. Create base population for a county (Cass County, 38017)
    base_pop_records = []
    for age in ages:
        for sex in sexes:
            for race in races:
                # Synthetic population - realistic age distribution
                if age < 20:
                    pop_base = 1000
                elif age < 40:
                    pop_base = 1500
                elif age < 65:
                    pop_base = 1200
                else:
                    pop_base = 800

                # Add some noise
                pop = int(pop_base * np.random.uniform(0.8, 1.2))

                base_pop_records.append(
                    {"year": base_year, "age": age, "sex": sex, "race": race, "population": pop}
                )

    base_population = pd.DataFrame(base_pop_records)

    # 2. Create fertility rates (ages 15-49, females only)
    fertility_records = []
    for age in range(15, 50):
        for race in races:
            # Synthetic fertility rate (births per 1000 women)
            if age < 20:
                rate = 0.020
            elif age < 30:
                rate = 0.085
            elif age < 35:
                rate = 0.095
            elif age < 40:
                rate = 0.055
            else:
                rate = 0.015

            fertility_records.append({"age": age, "race": race, "fertility_rate": rate})

    fertility_rates = pd.DataFrame(fertility_records)

    # 3. Create survival rates (all ages, both sexes, all races)
    survival_records = []
    for age in ages:
        for sex in sexes:
            for race in races:
                # Synthetic survival rate
                if age < 1:
                    rate = 0.995
                elif age < 20:
                    rate = 0.999
                elif age < 60:
                    rate = 0.998
                elif age < 80:
                    rate = 0.985
                else:
                    rate = 0.950

                survival_records.append(
                    {"age": age, "sex": sex, "race": race, "survival_rate": rate}
                )

    survival_rates = pd.DataFrame(survival_records)

    # 4. Create migration rates (net migration by cohort)
    migration_records = []
    for age in ages:
        for sex in sexes:
            for race in races:
                # Synthetic net migration (higher for working ages)
                if age < 20:
                    net_mig = np.random.randint(-5, 10)
                elif age < 40:
                    net_mig = np.random.randint(0, 50)
                elif age < 65:
                    net_mig = np.random.randint(-10, 20)
                else:
                    net_mig = np.random.randint(-5, 5)

                migration_records.append(
                    {"age": age, "sex": sex, "race_ethnicity": race, "net_migration": net_mig}
                )

    migration_rates = pd.DataFrame(migration_records)

    total_pop = base_population["population"].sum()
    logger.info(f"Created synthetic data. Base population: {total_pop:,.0f}")

    return base_population, fertility_rates, survival_rates, migration_rates


def example_1_single_geography():
    """Example 1: Run projection for a single geography."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 1: Single Geography Projection")
    logger.info("=" * 70)

    # Load synthetic data
    base_pop, fertility, survival, migration = create_synthetic_data()

    # Load config
    config = load_projection_config()

    # Run projection for Cass County (FIPS: 38017)
    logger.info("\nRunning projection for Cass County (38017)...")

    result = run_single_geography_projection(
        fips="38017",
        level="county",
        base_population=base_pop,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates=migration,
        config=config,
        save_results=False,  # Don't save for demo
    )

    # Display results
    logger.info(f"\nProjection complete for {result['geography']['name']}")
    logger.info(
        f"Base population: {result['metadata']['summary_statistics']['base_population']:,.0f}"
    )
    logger.info(
        f"Final population: {result['metadata']['summary_statistics']['final_population']:,.0f}"
    )
    logger.info(f"Growth rate: {result['metadata']['summary_statistics']['growth_rate']:+.1%}")
    logger.info(f"Processing time: {result['processing_time']:.2f} seconds")

    return result


def example_2_multiple_counties():
    """Example 2: Run projections for multiple counties."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: Multiple County Projections")
    logger.info("=" * 70)

    # Load geographic data
    counties = load_nd_counties()
    logger.info(f"\nLoaded {len(counties)} North Dakota counties")

    # For demo, use subset of counties
    county_fips = counties["county_fips"].tolist()[:5]  # First 5 counties
    logger.info(f"Demo: Projecting {len(county_fips)} counties")

    for fips in county_fips:
        logger.info(f"  - {get_geography_name(fips)}")

    # Create synthetic data for each county
    base_pop, fertility, survival, migration = create_synthetic_data()

    # Prepare data by geography (in practice, load from separate files)
    base_populations = {}
    migration_rates = {}

    for fips in county_fips:
        # For demo, use same data with slight variation
        pop_multiplier = np.random.uniform(0.5, 1.5)
        base_populations[fips] = base_pop.copy()
        base_populations[fips]["population"] = (
            base_populations[fips]["population"] * pop_multiplier
        ).astype(int)

        migration_rates[fips] = migration.copy()

    # Run projections (serial mode for demo)
    logger.info("\nRunning projections (serial mode)...")
    start_time = time.time()

    results = run_multi_geography_projections(
        level="county",
        base_population_by_geography=base_populations,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates_by_geography=migration_rates,
        fips_codes=county_fips,
        parallel=False,  # Serial for demo
        save_results=False,
    )

    elapsed = time.time() - start_time

    # Display results
    logger.info(f"\n{len(county_fips)} counties projected in {elapsed:.1f} seconds")
    logger.info(f"Successful: {results['metadata']['successful']}")
    logger.info(f"Failed: {results['metadata']['failed']}")

    logger.info("\nSummary:")
    print(results["summary"][["name", "base_population", "final_population", "growth_rate"]])

    return results


def example_3_parallel_processing():
    """Example 3: Demonstrate parallel processing speedup."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: Parallel Processing Comparison")
    logger.info("=" * 70)

    # Load counties
    counties = load_nd_counties()
    county_fips = counties["county_fips"].tolist()[:10]  # First 10 counties

    logger.info(f"\nComparing serial vs parallel for {len(county_fips)} counties")

    # Prepare data
    base_pop, fertility, survival, migration = create_synthetic_data()

    base_populations = {}
    migration_rates = {}

    for fips in county_fips:
        base_populations[fips] = base_pop.copy()
        migration_rates[fips] = migration.copy()

    # Serial execution
    logger.info("\n1. Running SERIAL projections...")
    start_time = time.time()

    results_serial = run_multi_geography_projections(
        level="county",
        base_population_by_geography=base_populations,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates_by_geography=migration_rates,
        fips_codes=county_fips,
        parallel=False,
        save_results=False,
    )

    serial_time = time.time() - start_time
    logger.info(f"   Serial time: {serial_time:.2f} seconds")

    # Parallel execution
    logger.info("\n2. Running PARALLEL projections...")
    start_time = time.time()

    results_parallel = run_multi_geography_projections(
        level="county",
        base_population_by_geography=base_populations,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates_by_geography=migration_rates,
        fips_codes=county_fips,
        parallel=True,
        max_workers=4,
        save_results=False,
    )

    parallel_time = time.time() - start_time
    logger.info(f"   Parallel time: {parallel_time:.2f} seconds")

    # Comparison
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    logger.info(f"\nSpeedup: {speedup:.1f}x")
    logger.info(f"Time saved: {serial_time - parallel_time:.2f} seconds")

    return results_serial, results_parallel


def example_4_hierarchical_aggregation():
    """Example 4: Hierarchical aggregation and validation."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Hierarchical Aggregation")
    logger.info("=" * 70)

    # Load places for a county (Cass County: 38017)
    places = load_nd_places()
    cass_places = places[places["county_fips"] == "38017"]

    logger.info(f"\nCass County contains {len(cass_places)} incorporated places:")
    for _, place in cass_places.iterrows():
        logger.info(f"  - {place['place_name']} (FIPS: {place['place_fips']})")

    # Prepare synthetic data
    base_pop, fertility, survival, migration = create_synthetic_data()

    place_fips = cass_places["place_fips"].tolist()
    base_populations = {}
    migration_rates_dict = {}

    for fips in place_fips:
        # Scale population for each place
        scale = np.random.uniform(0.2, 0.8)
        base_populations[fips] = base_pop.copy()
        base_populations[fips]["population"] = (
            base_populations[fips]["population"] * scale
        ).astype(int)
        migration_rates_dict[fips] = migration.copy()

    # Run place projections
    logger.info("\nRunning projections for places in Cass County...")

    place_results = run_multi_geography_projections(
        level="place",
        base_population_by_geography=base_populations,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates_by_geography=migration_rates_dict,
        fips_codes=place_fips,
        parallel=False,
        save_results=False,
    )

    # Aggregate places to county
    logger.info("\nAggregating places to county level...")

    county_aggregated = aggregate_to_county(place_results["results"])

    if "38017" in county_aggregated:
        aggregated_df = county_aggregated["38017"]
        base_year_agg = aggregated_df[aggregated_df["year"] == aggregated_df["year"].min()]
        total_pop = base_year_agg["population"].sum()

        logger.info(f"Cass County aggregated population (from places): {total_pop:,.0f}")

        # Validate aggregation
        logger.info("\nValidating aggregation...")

        validation = validate_aggregation(
            component_projections=place_results["results"],
            aggregated_projection=aggregated_df,
            component_level="place",
            aggregate_level="county",
            tolerance=0.01,
        )

        logger.info(f"Validation passed: {validation['valid']}")
        logger.info(f"Overall difference: {validation['overall']['percent_difference']:.3%}")

        if validation["warnings"]:
            logger.warning(f"Warnings: {validation['warnings']}")

        if validation["errors"]:
            logger.error(f"Errors: {validation['errors']}")

    return place_results, county_aggregated


def example_5_full_state_projection():
    """Example 5: Full state projection from county aggregation."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 5: Full State Projection from Counties")
    logger.info("=" * 70)

    # Load counties
    counties = load_nd_counties()
    county_fips = counties["county_fips"].tolist()[:5]  # Demo: first 5

    logger.info(f"\nProjecting {len(county_fips)} counties and aggregating to state...")

    # Prepare data
    base_pop, fertility, survival, migration = create_synthetic_data()

    base_populations = {}
    migration_rates_dict = {}

    for fips in county_fips:
        base_populations[fips] = base_pop.copy()
        migration_rates_dict[fips] = migration.copy()

    # Run county projections
    logger.info("\n1. Running county projections...")

    county_results = run_multi_geography_projections(
        level="county",
        base_population_by_geography=base_populations,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates_by_geography=migration_rates_dict,
        fips_codes=county_fips,
        parallel=True,
        max_workers=4,
        save_results=False,
    )

    # Aggregate to state
    logger.info("\n2. Aggregating counties to state level...")

    state_projection = aggregate_to_state(county_results["results"])

    # Display state results
    years = sorted(state_projection["year"].unique())
    base_year = years[0]
    final_year = years[-1]

    base_pop_state = state_projection[state_projection["year"] == base_year]["population"].sum()
    final_pop_state = state_projection[state_projection["year"] == final_year]["population"].sum()
    growth = final_pop_state - base_pop_state
    growth_rate = (growth / base_pop_state) if base_pop_state > 0 else 0

    logger.info(f"\nNorth Dakota Projection (from {len(county_fips)} counties):")
    logger.info(f"  Base year ({base_year}): {base_pop_state:,.0f}")
    logger.info(f"  Final year ({final_year}): {final_pop_state:,.0f}")
    logger.info(f"  Absolute growth: {growth:+,.0f}")
    logger.info(f"  Growth rate: {growth_rate:+.1%}")

    # Population by year
    logger.info("\nPopulation by year:")
    for year in years:
        year_pop = state_projection[state_projection["year"] == year]["population"].sum()
        logger.info(f"  {year}: {year_pop:,.0f}")

    return county_results, state_projection


def main():
    """Run all examples."""
    logger.info("=" * 70)
    logger.info("Multi-Geography Projection Examples")
    logger.info("=" * 70)

    try:
        # Example 1: Single geography
        example_1_single_geography()

        # Example 2: Multiple counties
        example_2_multiple_counties()

        # Example 3: Parallel processing
        example_3_parallel_processing()

        # Example 4: Hierarchical aggregation
        example_4_hierarchical_aggregation()

        # Example 5: Full state projection
        example_5_full_state_projection()

        logger.info("\n" + "=" * 70)
        logger.info("All examples completed successfully!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Example failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
