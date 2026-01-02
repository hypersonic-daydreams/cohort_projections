#!/usr/bin/env python3
"""
Integration Test Script for Cohort Projections.

This script runs a complete end-to-end projection using actual processed data
for Cass County, North Dakota (FIPS 38017, Fargo area).

Usage:
    python scripts/run_integration_test.py

Output:
    - Console summary of projection results
    - Saved results to data/processed/integration_test_results.csv
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cohort_projections.core.cohort_component import CohortComponentProjection  # noqa: E402


def load_processed_data():
    """Load all processed data files."""
    data_dir = PROJECT_ROOT / "data" / "raw"

    print("=" * 70)
    print("COHORT PROJECTION INTEGRATION TEST")
    print("=" * 70)
    print(f"\nLoading data files from: {data_dir}")
    print("-" * 70)

    data = {}

    # Load fertility rates
    fertility_path = data_dir / "fertility" / "asfr_processed.csv"
    data["fertility"] = pd.read_csv(fertility_path)
    print(f"  Fertility rates: {len(data['fertility'])} records loaded")

    # Load survival rates
    survival_path = data_dir / "mortality" / "survival_rates_processed.csv"
    data["survival"] = pd.read_csv(survival_path)
    print(f"  Survival rates: {len(data['survival'])} records loaded")

    # Load migration data
    migration_path = data_dir / "migration" / "nd_migration_processed.csv"
    data["migration"] = pd.read_csv(migration_path)
    print(f"  Migration data: {len(data['migration'])} records loaded")

    # Load county population
    county_path = data_dir / "population" / "nd_county_population.csv"
    data["county_population"] = pd.read_csv(county_path)
    print(f"  County population: {len(data['county_population'])} counties loaded")

    # Load age-sex-race distribution
    age_sex_path = data_dir / "population" / "nd_age_sex_race_distribution.csv"
    data["age_sex_race"] = pd.read_csv(age_sex_path)
    print(f"  Age-sex-race dist: {len(data['age_sex_race'])} records loaded")

    return data


def build_base_population(data, county_fips=38017):
    """Build base population for a county from processed data."""
    county_pop = data["county_population"]
    age_sex_race = data["age_sex_race"]

    # Get county total population
    county_row = county_pop[county_pop["county_fips"] == county_fips]
    if county_row.empty:
        raise ValueError(f"County FIPS {county_fips} not found")

    county_name = county_row["county_name"].values[0]
    total_pop = county_row["population_2024"].values[0]
    print(f"\nBuilding base population for {county_name} County (FIPS {county_fips})")
    print(f"  2024 total population: {total_pop:,}")

    # Calculate proportions from age-sex-race distribution
    total_proportion = age_sex_race["proportion"].sum()

    # Build base population by applying proportions
    base_pop_records = []

    race_map = {
        "white_nonhispanic": "White alone, Non-Hispanic",
        "black_nonhispanic": "Black alone, Non-Hispanic",
        "aian_nonhispanic": "AIAN alone, Non-Hispanic",
        "asian_nonhispanic": "Asian/PI alone, Non-Hispanic",
        "nhpi_nonhispanic": "Asian/PI alone, Non-Hispanic",
        "multiracial_nonhispanic": "Two or more races, Non-Hispanic",
        "other_nonhispanic": "Two or more races, Non-Hispanic",
        "hispanic": "Hispanic (any race)",
    }

    for _, row in age_sex_race.iterrows():
        age_group = row["age_group"]

        # Parse age group
        if age_group == "85+":
            ages = list(range(85, 91))
        elif "-" in age_group:
            age_start, age_end = map(int, age_group.split("-"))
            ages = list(range(age_start, age_end + 1))
        else:
            continue

        # Calculate population for this group
        population_in_group = total_pop * (row["proportion"] / total_proportion)
        pop_per_age = population_in_group / len(ages)

        race_ethnicity = row["race_ethnicity"]
        mapped_race = race_map.get(race_ethnicity, "Two or more races, Non-Hispanic")

        for age in ages:
            base_pop_records.append(
                {
                    "year": 2025,
                    "age": min(age, 90),
                    "sex": row["sex"].title(),
                    "race": mapped_race,
                    "population": pop_per_age,
                }
            )

    base_pop_df = pd.DataFrame(base_pop_records)

    # Aggregate duplicates
    base_pop_df = base_pop_df.groupby(["year", "age", "sex", "race"], as_index=False).agg(
        {"population": "sum"}
    )

    print(f"  Built {len(base_pop_df)} cohorts for base year 2025")
    print(f"  Total base population: {base_pop_df['population'].sum():,.0f}")

    return base_pop_df


def prepare_fertility_rates(data):
    """Prepare fertility rates for projection engine."""
    fertility_df = data["fertility"]

    race_map = {
        "total": "White alone, Non-Hispanic",
        "white_nh": "White alone, Non-Hispanic",
        "black_nh": "Black alone, Non-Hispanic",
        "aian_nh": "AIAN alone, Non-Hispanic",
        "asian_nh": "Asian/PI alone, Non-Hispanic",
        "hispanic": "Hispanic (any race)",
    }

    all_races = [
        "White alone, Non-Hispanic",
        "Black alone, Non-Hispanic",
        "AIAN alone, Non-Hispanic",
        "Asian/PI alone, Non-Hispanic",
        "Two or more races, Non-Hispanic",
        "Hispanic (any race)",
    ]

    fertility_records = []

    for _, row in fertility_df.iterrows():
        age_group = row["age"]
        race = row["race_ethnicity"]
        asfr = row["asfr"] / 1000  # Convert from per-1000 to proportion

        if race not in race_map:
            continue

        if isinstance(age_group, str) and "-" in age_group:
            age_start, age_end = map(int, age_group.split("-"))
            ages = list(range(age_start, age_end + 1))
        else:
            continue

        mapped_race = race_map[race]

        for age in ages:
            fertility_records.append(
                {
                    "age": age,
                    "race": mapped_race,
                    "fertility_rate": asfr,
                }
            )

    fertility_rate_df = pd.DataFrame(fertility_records)
    fertility_rate_df = fertility_rate_df.groupby(["age", "race"], as_index=False).agg(
        {"fertility_rate": "mean"}
    )

    # Fill missing races with proxy rates
    total_rates = fertility_rate_df[fertility_rate_df["race"] == "White alone, Non-Hispanic"].copy()

    for race in all_races:
        if race not in fertility_rate_df["race"].values:
            race_rates = total_rates.copy()
            race_rates["race"] = race
            fertility_rate_df = pd.concat([fertility_rate_df, race_rates], ignore_index=True)

    print(f"\nFertility rates prepared: {len(fertility_rate_df)} age-race combinations")

    return fertility_rate_df


def prepare_survival_rates(data):
    """Prepare survival rates for projection engine."""
    survival_df = data["survival"]

    race_map = {
        "aian_nh": "AIAN alone, Non-Hispanic",
        "asian_nh": "Asian/PI alone, Non-Hispanic",
        "black_nh": "Black alone, Non-Hispanic",
        "hispanic": "Hispanic (any race)",
        "white_nh": "White alone, Non-Hispanic",
    }

    sex_map = {"female": "Female", "male": "Male"}

    all_races = [
        "White alone, Non-Hispanic",
        "Black alone, Non-Hispanic",
        "AIAN alone, Non-Hispanic",
        "Asian/PI alone, Non-Hispanic",
        "Two or more races, Non-Hispanic",
        "Hispanic (any race)",
    ]

    survival_records = []

    for _, row in survival_df.iterrows():
        race = row.get("race_ethnicity", row.get("race"))
        sex = row.get("sex", "")

        if race not in race_map or sex not in sex_map:
            continue

        survival_records.append(
            {
                "age": int(row["age"]),
                "sex": sex_map[sex],
                "race": race_map[race],
                "survival_rate": float(row["survival_rate"]),
            }
        )

    survival_rate_df = pd.DataFrame(survival_records)

    # Fill missing race categories
    for sex in ["Male", "Female"]:
        for race in all_races:
            existing = survival_rate_df[
                (survival_rate_df["sex"] == sex) & (survival_rate_df["race"] == race)
            ]
            if existing.empty:
                proxy_rates = survival_rate_df[
                    (survival_rate_df["sex"] == sex)
                    & (survival_rate_df["race"] == "White alone, Non-Hispanic")
                ].copy()
                proxy_rates["race"] = race
                survival_rate_df = pd.concat([survival_rate_df, proxy_rates], ignore_index=True)

    print(f"Survival rates prepared: {len(survival_rate_df)} age-sex-race combinations")

    return survival_rate_df


def prepare_migration_rates(data, base_population, county_fips=38017):
    """Prepare migration rates for projection engine."""
    migration_df = data["migration"]

    # Get county migration data
    county_migration = migration_df[migration_df["county_fips"] == county_fips]

    if county_migration.empty:
        print(f"No migration data for county {county_fips}, using zero migration")
        migration_records = []
        for _, row in base_population.iterrows():
            migration_records.append(
                {
                    "age": row["age"],
                    "sex": row["sex"],
                    "race": row["race"],
                    "net_migration": 0.0,
                }
            )
        return pd.DataFrame(migration_records)

    # Calculate average annual net migration
    avg_net_migration = county_migration["net_migration"].mean()
    print(f"Average annual net migration for county: {avg_net_migration:+,.0f}")

    # Distribute migration by age pattern
    total_pop = base_population["population"].sum()

    migration_records = []
    for _, row in base_population.iterrows():
        age = row["age"]

        # Age-specific migration factor
        if 18 <= age <= 34:
            age_factor = 1.5
        elif 35 <= age <= 54:
            age_factor = 1.0
        elif age >= 65:
            age_factor = 0.8
        else:
            age_factor = 0.5

        pop_proportion = row["population"] / total_pop if total_pop > 0 else 0
        net_mig = avg_net_migration * pop_proportion * age_factor

        migration_records.append(
            {
                "age": row["age"],
                "sex": row["sex"],
                "race": row["race"],
                "net_migration": net_mig,
            }
        )

    migration_rate_df = pd.DataFrame(migration_records)

    # Normalize to match expected total
    current_total = migration_rate_df["net_migration"].sum()
    if current_total != 0:
        scale_factor = avg_net_migration / current_total
        migration_rate_df["net_migration"] *= scale_factor

    print(f"Migration rates prepared: {len(migration_rate_df)} cohorts")

    return migration_rate_df


def run_projection(base_pop, fertility_rates, survival_rates, migration_rates):
    """Run the 5-year projection."""
    print("\n" + "=" * 70)
    print("RUNNING 5-YEAR PROJECTION (2025-2030)")
    print("=" * 70)

    config = {
        "project": {
            "name": "Integration Test - Cass County",
            "base_year": 2025,
            "projection_horizon": 5,
        },
        "demographics": {
            "age_groups": {
                "min_age": 0,
                "max_age": 90,
            },
        },
        "rates": {
            "fertility": {
                "apply_to_ages": [15, 49],
                "sex_ratio_male": 0.51,
            },
            "mortality": {
                "improvement_factor": 0.0,
            },
        },
    }

    try:
        projection = CohortComponentProjection(
            base_population=base_pop,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=config,
        )

        results = projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        summary = projection.get_projection_summary()

        return results, summary, None

    except Exception as e:
        return None, None, str(e)


def validate_results(results):
    """Validate projection results."""
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    errors = []
    warnings = []

    # Check for negative populations
    negative = results[results["population"] < 0]
    if not negative.empty:
        errors.append(f"Found {len(negative)} cohorts with negative population")
    else:
        print("[PASS] No negative populations")

    # Calculate population by year
    pop_by_year = results.groupby("year")["population"].sum()

    # Check year-over-year changes
    all_changes_ok = True
    for i in range(1, len(pop_by_year)):
        prev = pop_by_year.iloc[i - 1]
        curr = pop_by_year.iloc[i]
        change = (curr - prev) / prev

        if abs(change) > 0.03:
            warnings.append(
                f"Year {pop_by_year.index[i]}: Population change {change:.1%} exceeds 3%"
            )
            all_changes_ok = False

    if all_changes_ok:
        print("[PASS] Year-over-year changes within +/- 3%")
    else:
        print("[WARN] Some year-over-year changes exceed 3%")

    # Check total population is reasonable
    start_pop = pop_by_year.iloc[0]

    if 150_000 < start_pop < 250_000:
        print(f"[PASS] Starting population {start_pop:,.0f} is reasonable for Cass County")
    else:
        warnings.append(f"Starting population {start_pop:,.0f} outside expected range")

    # Check births are reasonable
    females_15_44 = results[
        (results["year"] == 2025)
        & (results["sex"] == "Female")
        & (results["age"] >= 15)
        & (results["age"] <= 44)
    ]["population"].sum()

    births_2026 = results[(results["year"] == 2026) & (results["age"] == 0)]["population"].sum()

    if females_15_44 > 0:
        birth_rate = births_2026 / females_15_44
        if 0.015 < birth_rate < 0.05:
            print(f"[PASS] Birth rate {birth_rate:.3f} is reasonable (2-4% of females 15-44)")
        else:
            warnings.append(f"Birth rate {birth_rate:.3f} is outside expected range")

    return errors, warnings


def calculate_summary_statistics(results):
    """Calculate and display summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Population by year
    pop_by_year = results.groupby("year")["population"].sum()

    print("\nTotal Population by Year:")
    print("-" * 40)
    for year, pop in pop_by_year.items():
        if year > 2025:
            prev_pop = pop_by_year[year - 1]
            change = pop - prev_pop
            pct_change = (change / prev_pop) * 100
            print(f"  {year}: {pop:>12,.0f}  ({change:>+8,.0f}, {pct_change:>+.2f}%)")
        else:
            print(f"  {year}: {pop:>12,.0f}  (base year)")

    # Calculate component changes
    start_pop = pop_by_year.iloc[0]
    end_pop = pop_by_year.iloc[-1]

    # Births (all age 0 populations from 2026 onwards)
    total_births = results[(results["year"] > 2025) & (results["age"] == 0)]["population"].sum()

    # Estimate deaths using survival rates
    # Deaths = pop before survival - pop after survival
    # For simplicity, we approximate deaths as:
    # deaths ~ start_pop * (1 - avg_survival) * years
    total_deaths_estimate = start_pop * 0.01 * 5  # ~1% crude death rate

    # Net migration (rough estimate from change)
    natural_increase = total_births - total_deaths_estimate
    net_pop_change = end_pop - start_pop
    net_migration_estimate = net_pop_change - natural_increase

    print("\nComponent Summary (5-year totals, estimated):")
    print("-" * 40)
    print(f"  Starting population:    {start_pop:>12,.0f}")
    print(f"  Total births:           {total_births:>12,.0f}")
    print(f"  Total deaths (est):     {total_deaths_estimate:>12,.0f}")
    print(f"  Natural increase:       {natural_increase:>12,.0f}")
    print(f"  Net migration (est):    {net_migration_estimate:>12,.0f}")
    print(f"  Ending population:      {end_pop:>12,.0f}")
    print(f"  Net change:             {net_pop_change:>+12,.0f}")

    # Age structure comparison
    print("\nAge Structure (start vs end):")
    print("-" * 40)

    for label, year in [("Start (2025)", 2025), ("End (2030)", 2030)]:
        year_data = results[results["year"] == year]

        under_18 = year_data[year_data["age"] < 18]["population"].sum()
        age_18_64 = year_data[(year_data["age"] >= 18) & (year_data["age"] < 65)][
            "population"
        ].sum()
        age_65_plus = year_data[year_data["age"] >= 65]["population"].sum()
        total = year_data["population"].sum()

        print(f"\n  {label}:")
        print(f"    Under 18:   {under_18:>10,.0f} ({under_18 / total * 100:>5.1f}%)")
        print(f"    18-64:      {age_18_64:>10,.0f} ({age_18_64 / total * 100:>5.1f}%)")
        print(f"    65+:        {age_65_plus:>10,.0f} ({age_65_plus / total * 100:>5.1f}%)")

    return {
        "start_population": start_pop,
        "end_population": end_pop,
        "total_births": total_births,
        "deaths_estimate": total_deaths_estimate,
        "net_migration_estimate": net_migration_estimate,
    }


def save_results(results, output_path):
    """Save projection results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return output_path


def main():
    """Main entry point for integration test."""
    start_time = datetime.now(UTC)

    print(f"\nTest started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    try:
        data = load_processed_data()
    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found: {e}")
        print("Please ensure all processed data files exist.")
        sys.exit(1)

    # Build base population for Cass County
    try:
        base_pop = build_base_population(data, county_fips=38017)
    except Exception as e:
        print(f"\nERROR building base population: {e}")
        sys.exit(1)

    # Prepare rate data
    fertility_rates = prepare_fertility_rates(data)
    survival_rates = prepare_survival_rates(data)
    migration_rates = prepare_migration_rates(data, base_pop, county_fips=38017)

    # Run projection
    results, summary, error = run_projection(
        base_pop, fertility_rates, survival_rates, migration_rates
    )

    if error:
        print(f"\nERROR running projection: {error}")
        sys.exit(1)

    # Validate results
    errors, warnings = validate_results(results)

    if errors:
        print("\nValidation Errors:")
        for err in errors:
            print(f"  - {err}")

    if warnings:
        print("\nValidation Warnings:")
        for warn in warnings:
            print(f"  - {warn}")

    # Calculate summary statistics
    calculate_summary_statistics(results)

    # Save results
    output_path = PROJECT_ROOT / "data" / "processed" / "integration_test_results.csv"
    saved_path = save_results(results, output_path)

    # Final summary
    end_time = datetime.now(UTC)
    duration = (end_time - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)
    print(f"\nDuration: {duration:.1f} seconds")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Results file: {saved_path}")

    if errors:
        print("\nSTATUS: FAILED (see errors above)")
        sys.exit(1)
    else:
        print("\nSTATUS: PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
