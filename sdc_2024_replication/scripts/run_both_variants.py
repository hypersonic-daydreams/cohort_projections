#!/usr/bin/env python3
"""
SDC 2024 Replication: Run Both Methodology Variants and Compare

This script runs both the original data variant and the updated data variant
of the SDC 2024 population projection methodology, then produces a comparison
table showing results side-by-side with SDC official projections.

Variants:
1. Original Data Variant: Uses data from data/ directory, base year 2020
2. Updated Data Variant: Uses data from data_updated/ directory, base year 2024

Author: SDC 2024 Replication Project
Date: 2025-12-28
"""
# mypy: disable-error-code="assignment"

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Import projection engine components
from projection_engine import (
    AGE_GROUPS,
    PERIOD_MULTIPLIERS,
    ProjectionInputs,
    ProjectionOutputs,
    project_county,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# SDC Official Projections (for reference)
# =============================================================================

SDC_OFFICIAL_PROJECTIONS = {
    2020: 779_094,
    2025: 796_989,
    2030: 831_543,
    2035: 865_397,
    2040: 890_424,
    2045: 925_101,
    2050: 957_194,
}

# =============================================================================
# Data Loading Functions
# =============================================================================


def load_base_population(filepath: Path) -> pd.DataFrame:
    """Load base population from CSV file."""
    logger.info("Loading base population from: %s", filepath)
    df = pd.read_csv(filepath)

    # Rename columns to match expected format
    if "county_name" in df.columns:
        df = df.rename(columns={"county_name": "county"})

    # Standardize sex values to titlecase
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title()

    total_pop = df["population"].sum()
    logger.info("Loaded %d rows, total population: %s", len(df), f"{total_pop:,.0f}")
    return df


def load_survival_rates(filepath: Path) -> pd.DataFrame:
    """Load survival rates from CSV file."""
    logger.info("Loading survival rates from: %s", filepath)
    df = pd.read_csv(filepath)

    # Standardize sex values to titlecase
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title()

    logger.info("Loaded %d survival rate entries", len(df))
    return df


def load_fertility_rates(filepath: Path) -> pd.DataFrame:
    """Load fertility rates from CSV file.

    NOTE: The extracted fertility rates appear to be 5-year cumulative rates,
    but the projection engine expects annual rates (it multiplies by 5).
    We divide by 5 to convert to annual rates.
    """
    logger.info("Loading fertility rates from: %s", filepath)
    df = pd.read_csv(filepath)

    # Rename columns to match expected format
    if "county_name" in df.columns:
        df = df.rename(columns={"county_name": "county"})

    # Convert from 5-year cumulative to annual rates
    df["fertility_rate"] = df["fertility_rate"] / 5.0

    logger.info("Loaded %d fertility rate entries", len(df))
    return df


def load_migration_rates(filepath: Path) -> pd.DataFrame:
    """Load migration rates from CSV file."""
    logger.info("Loading migration rates from: %s", filepath)
    df = pd.read_csv(filepath)

    # Rename columns to match expected format
    if "county_name" in df.columns:
        df = df.rename(columns={"county_name": "county"})

    # Standardize sex values to titlecase
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title()

    logger.info("Loaded %d migration rate entries", len(df))
    return df


def load_adjustments(filepath: Path, projection_years: list[int]) -> pd.DataFrame | None:
    """Load manual adjustments from CSV file."""
    if not filepath.exists():
        logger.info("No adjustments file found at: %s", filepath)
        return None

    logger.info("Loading adjustments from: %s", filepath)
    df = pd.read_csv(filepath)

    # Rename columns to match expected format
    if "county_name" in df.columns:
        df = df.rename(columns={"county_name": "county"})

    # Standardize sex values to titlecase
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title()

    # Expand to all projection years if needed
    if "year" not in df.columns:
        logger.info("Expanding adjustments to all projection years")
        expanded_rows = []
        for year in projection_years:
            year_df = df.copy()
            year_df["year"] = year
            expanded_rows.append(year_df)
        df = pd.concat(expanded_rows, ignore_index=True)

    logger.info("Loaded %d adjustment entries", len(df))
    return df


def load_all_inputs(data_dir: Path, projection_years: list[int]) -> ProjectionInputs:
    """Load all projection inputs from the data directory."""
    base_pop = load_base_population(data_dir / "base_population_by_county.csv")
    survival = load_survival_rates(data_dir / "survival_rates_by_county.csv")
    fertility = load_fertility_rates(data_dir / "fertility_rates_by_county.csv")
    migration = load_migration_rates(data_dir / "migration_rates_by_county.csv")
    adjustments = load_adjustments(data_dir / "adjustment_factors_by_county.csv", projection_years)

    return ProjectionInputs(
        base_population=base_pop,
        survival_rates=survival,
        fertility_rates=fertility,
        migration_rates=migration,
        adjustments=adjustments,
    )


# =============================================================================
# Projection Runner (with configurable base year)
# =============================================================================


def calculate_state_totals_flexible(
    population: pd.DataFrame,
    births: pd.DataFrame,
    deaths: pd.DataFrame,
    migration: pd.DataFrame,
    all_years: list[int],
    projection_years: list[int],
) -> pd.DataFrame:
    """Calculate state-level totals from county projections (flexible years).

    Args:
        population: County-level population projections
        births: County-level births by period
        deaths: County-level deaths by period
        migration: County-level migration by period
        all_years: All years including base year
        projection_years: Projection years (without base year)

    Returns:
        DataFrame with state totals by year
    """
    results = []

    for year in all_years:
        year_pop = population[population["year"] == year]
        total_pop = year_pop["population"].sum()

        row = {
            "year": year,
            "population": total_pop,
            "male_population": year_pop[year_pop["sex"] == "Male"]["population"].sum(),
            "female_population": year_pop[year_pop["sex"] == "Female"]["population"].sum(),
        }

        # Add age distribution
        for age_group in AGE_GROUPS:
            row[f"pop_{age_group}"] = year_pop[year_pop["age_group"] == age_group][
                "population"
            ].sum()

        results.append(row)

    state_df = pd.DataFrame(results)

    # Add period components (births, deaths, migration)
    for year in projection_years:
        period_births = births[births["period_end"] == year]["births"].sum()
        period_deaths = deaths[deaths["period_end"] == year]["deaths"].sum()
        period_migration = migration[migration["period_end"] == year]["migration"].sum()

        idx = state_df[state_df["year"] == year].index
        if len(idx) > 0:
            idx = idx[0]
            state_df.loc[idx, "period_births"] = period_births
            state_df.loc[idx, "period_deaths"] = period_deaths
            state_df.loc[idx, "period_migration"] = period_migration
            state_df.loc[idx, "natural_change"] = period_births - period_deaths

    return state_df


def run_projections_with_base_year(
    inputs: ProjectionInputs,
    base_year: int,
    end_year: int = 2050,
    projection_years: list[int] | None = None,
) -> ProjectionOutputs:
    """Run projections with a configurable base year.

    Args:
        inputs: ProjectionInputs with all required data
        base_year: Starting year for projections
        end_year: Ending year for projections (default 2050)
        projection_years: Optional explicit list of projection years

    Returns:
        ProjectionOutputs with full results
    """
    # Generate projection years from base_year to end_year in 5-year steps
    # or use provided projection_years
    if projection_years is None:
        projection_years = list(range(base_year + 5, end_year + 1, 5))
    all_years = [base_year] + projection_years

    logger.info("=" * 60)
    logger.info("Starting Cohort-Component Projection")
    logger.info("=" * 60)
    logger.info("Base year: %d", base_year)
    logger.info("Projection years: %s", projection_years)
    logger.info("Counties: %d", len(inputs.counties))

    all_populations = []
    all_births = []
    all_deaths = []
    all_migrations = []

    # Initialize with base population
    base_pop = inputs.base_population.copy()
    base_pop["year"] = base_year
    all_populations.append(base_pop)

    current_population = base_pop.copy()

    # Project each period
    for i, to_year in enumerate(projection_years):
        from_year = all_years[i]
        logger.info("")
        logger.info("-" * 40)
        logger.info("Projecting period: %d -> %d", from_year, to_year)

        # Get period multiplier (use 0.6 as default for years not in original dict)
        period_multiplier = PERIOD_MULTIPLIERS.get(to_year, 0.6)
        logger.info("Period multiplier: %.1f", period_multiplier)

        period_populations = []
        period_births = []
        period_deaths = []
        period_migrations = []

        for county in inputs.counties:
            county_pop = current_population[current_population["county"] == county]

            projected, components = project_county(
                population=county_pop,
                inputs=inputs,
                county=county,
                from_year=from_year,
                to_year=to_year,
            )

            period_populations.append(projected)

            period_births.append(
                {
                    "county": county,
                    "period_start": from_year,
                    "period_end": to_year,
                    "births": components["births"],
                }
            )
            period_deaths.append(
                {
                    "county": county,
                    "period_start": from_year,
                    "period_end": to_year,
                    "deaths": components["deaths"],
                }
            )
            period_migrations.append(
                {
                    "county": county,
                    "period_start": from_year,
                    "period_end": to_year,
                    "migration": components["migration"],
                }
            )

        # Combine period results
        period_pop = pd.concat(period_populations, ignore_index=True)
        all_populations.append(period_pop)
        all_births.extend(period_births)
        all_deaths.extend(period_deaths)
        all_migrations.extend(period_migrations)

        # Update current population for next period
        current_population = period_pop.copy()

        # Log period summary
        period_total = period_pop["population"].sum()
        logger.info("Period %d total population: %s", to_year, f"{period_total:,.0f}")

    # Combine all results
    logger.info("")
    logger.info("=" * 60)
    logger.info("Projection Complete")
    logger.info("=" * 60)

    population_df = pd.concat(all_populations, ignore_index=True)
    births_df = pd.DataFrame(all_births)
    deaths_df = pd.DataFrame(all_deaths)
    migration_df = pd.DataFrame(all_migrations)

    # Calculate state totals using our flexible function
    state_totals = calculate_state_totals_flexible(
        population_df, births_df, deaths_df, migration_df, all_years, projection_years
    )

    # Log final summary
    for year in all_years:
        year_total = population_df[population_df["year"] == year]["population"].sum()
        logger.info("Year %d: %s", year, f"{year_total:,.0f}")

    return ProjectionOutputs(
        population=population_df,
        births=births_df,
        deaths=deaths_df,
        migration=migration_df,
        state_totals=state_totals,
    )


# =============================================================================
# Comparison Functions
# =============================================================================


def create_comparison_table(
    original_outputs: ProjectionOutputs,
    updated_outputs: ProjectionOutputs,
    original_base_year: int,
    updated_base_year: int,
) -> pd.DataFrame:
    """Create a comparison table with SDC official, original, and updated projections.

    Args:
        original_outputs: Results from original data variant
        updated_outputs: Results from updated data variant
        original_base_year: Base year for original variant
        updated_base_year: Base year for updated variant

    Returns:
        DataFrame with comparison results
    """
    comparison_rows = []

    # Get all unique years from SDC official and both outputs
    all_years = set(SDC_OFFICIAL_PROJECTIONS.keys())
    all_years.update(original_outputs.state_totals["year"].astype(int).tolist())
    all_years.update(updated_outputs.state_totals["year"].astype(int).tolist())
    all_years = sorted(all_years)

    for year in all_years:
        row = {"year": year}

        # SDC Official
        sdc_official = SDC_OFFICIAL_PROJECTIONS.get(year)
        row["sdc_official"] = sdc_official

        # Original data variant
        original_row = original_outputs.state_totals[original_outputs.state_totals["year"] == year]
        if not original_row.empty:
            row["original_data"] = round(original_row["population"].values[0])
        else:
            row["original_data"] = None

        # Updated data variant
        updated_row = updated_outputs.state_totals[updated_outputs.state_totals["year"] == year]
        if not updated_row.empty:
            row["updated_data"] = round(updated_row["population"].values[0])
        else:
            row["updated_data"] = None

        # Calculate difference between updated and original (where both exist)
        if row["original_data"] is not None and row["updated_data"] is not None:
            row["diff_updated_vs_original"] = row["updated_data"] - row["original_data"]
        else:
            row["diff_updated_vs_original"] = None

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def print_comparison_table(comparison: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 95)
    print("VARIANT COMPARISON: SDC Official vs Original Data vs Updated Data")
    print("=" * 95)

    print(
        "\n{:<8} {:>15} {:>15} {:>15} {:>20}".format(
            "Year", "SDC Official", "Original Data", "Updated Data", "Diff (Upd vs Orig)"
        )
    )
    print("-" * 95)

    for _, row in comparison.iterrows():
        year = int(row["year"])
        sdc = row["sdc_official"]
        orig = row["original_data"]
        upd = row["updated_data"]
        diff = row["diff_updated_vs_original"]

        sdc_str = f"{int(sdc):,}" if pd.notna(sdc) else "N/A"
        orig_str = f"{int(orig):,}" if pd.notna(orig) else "N/A"
        upd_str = f"{int(upd):,}" if pd.notna(upd) else "N/A"
        diff_str = f"{int(diff):+,}" if pd.notna(diff) else "N/A"

        print(f"{year:<8} {sdc_str:>15} {orig_str:>15} {upd_str:>15} {diff_str:>20}")

    print("-" * 95)

    # Summary statistics for common years
    common_rows = comparison[
        comparison["original_data"].notna() & comparison["updated_data"].notna()
    ]

    if not common_rows.empty:
        avg_diff = common_rows["diff_updated_vs_original"].mean()
        print(f"\nAverage difference (Updated vs Original): {avg_diff:+,.0f}")

        # Compare to SDC for 2050
        row_2050 = comparison[comparison["year"] == 2050]
        if not row_2050.empty:
            sdc_2050 = row_2050["sdc_official"].values[0]
            orig_2050 = row_2050["original_data"].values[0]
            upd_2050 = row_2050["updated_data"].values[0]

            if sdc_2050 is not None and orig_2050 is not None:
                print(f"\n2050 Comparison to SDC Official ({sdc_2050:,}):")
                print(f"  Original Data: {int(orig_2050):,} (diff: {int(orig_2050 - sdc_2050):+,})")
            if sdc_2050 is not None and upd_2050 is not None:
                print(f"  Updated Data:  {int(upd_2050):,} (diff: {int(upd_2050 - sdc_2050):+,})")

    print("=" * 95)


def save_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Save comparison results to CSV."""
    output_path = output_dir / "variant_comparison.csv"
    comparison.to_csv(output_path, index=False)
    logger.info("Saved comparison to: %s", output_path)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Run both variants and compare results."""
    # Determine paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    original_data_dir = base_dir / "data"
    updated_data_dir = base_dir / "data_updated"
    output_dir = base_dir / "output"

    # Configuration
    original_base_year = 2020
    updated_base_year = 2024
    end_year = 2050

    print("\n" + "=" * 95)
    print("SDC 2024 REPLICATION: RUNNING BOTH METHODOLOGY VARIANTS")
    print("=" * 95)

    print("\nConfiguration:")
    print(f"  Original Data Directory: {original_data_dir}")
    print(f"  Original Base Year: {original_base_year}")
    print(f"  Updated Data Directory: {updated_data_dir}")
    print(f"  Updated Base Year: {updated_base_year}")
    print(f"  End Year: {end_year}")
    print(f"  Output Directory: {output_dir}")

    # Verify directories exist
    if not original_data_dir.exists():
        raise FileNotFoundError(f"Original data directory not found: {original_data_dir}")
    if not updated_data_dir.exists():
        raise FileNotFoundError(f"Updated data directory not found: {updated_data_dir}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Run Original Data Variant
    # =========================================================================
    print("\n" + "=" * 60)
    print("VARIANT 1: ORIGINAL DATA (Base Year 2020)")
    print("=" * 60)

    original_projection_years = list(range(original_base_year + 5, end_year + 1, 5))
    original_inputs = load_all_inputs(original_data_dir, original_projection_years)

    print(f"\n  Counties: {len(original_inputs.counties)}")
    print(f"  Base population: {original_inputs.base_population['population'].sum():,.0f}")

    original_outputs = run_projections_with_base_year(
        inputs=original_inputs,
        base_year=original_base_year,
        end_year=end_year,
    )

    # =========================================================================
    # Run Updated Data Variant
    # =========================================================================
    print("\n" + "=" * 60)
    print("VARIANT 2: UPDATED DATA (Base Year 2024)")
    print("=" * 60)

    # Use standard SDC projection years for comparability
    # This means the first interval is 2024->2025 (1 year), then normal 5-year intervals
    updated_projection_years_for_adjustments = [2025, 2030, 2035, 2040, 2045, 2050]
    updated_inputs = load_all_inputs(updated_data_dir, updated_projection_years_for_adjustments)

    print(f"\n  Counties: {len(updated_inputs.counties)}")
    print(f"  Base population: {updated_inputs.base_population['population'].sum():,.0f}")
    print("  NOTE: Using standard SDC projection years for comparability")

    updated_outputs = run_projections_with_base_year(
        inputs=updated_inputs,
        base_year=updated_base_year,
        end_year=end_year,
        projection_years=updated_projection_years_for_adjustments,
    )

    # =========================================================================
    # Create and Save Comparison
    # =========================================================================
    comparison = create_comparison_table(
        original_outputs=original_outputs,
        updated_outputs=updated_outputs,
        original_base_year=original_base_year,
        updated_base_year=updated_base_year,
    )

    print_comparison_table(comparison)
    save_comparison(comparison, output_dir)

    # Also save individual variant state totals
    original_state_file = output_dir / "original_variant_state_totals.csv"
    original_outputs.state_totals.to_csv(original_state_file, index=False)
    logger.info("Saved original variant state totals to: %s", original_state_file)

    updated_state_file = output_dir / "updated_variant_state_totals.csv"
    updated_outputs.state_totals.to_csv(updated_state_file, index=False)
    logger.info("Saved updated variant state totals to: %s", updated_state_file)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - variant_comparison.csv")
    print("  - original_variant_state_totals.csv")
    print("  - updated_variant_state_totals.csv")


if __name__ == "__main__":
    main()
