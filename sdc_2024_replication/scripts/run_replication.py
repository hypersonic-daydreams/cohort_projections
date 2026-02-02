#!/usr/bin/env python3
"""
SDC 2024 Replication: Runner Script

DEPRECATED: This script is deprecated. Please use run_all_variants.py instead.

    python run_all_variants.py --variant original

This script:
1. Loads the extracted data from the SDC 2024 replication data directory
2. Runs the projection engine
3. Saves results to the output directory
4. Compares state totals against SDC official projections
5. Outputs a comparison table showing our replication vs SDC official

Author: SDC 2024 Replication Project
Date: 2025-12-28
"""

from __future__ import annotations

import warnings

warnings.warn(
    "run_replication.py is deprecated. Use run_all_variants.py --variant original instead.",
    DeprecationWarning,
    stacklevel=2,
)
print(
    "\n*** DEPRECATION WARNING ***\n"
    "This script is deprecated. Please use run_all_variants.py instead:\n"
    "    python run_all_variants.py --variant original\n"
)

import logging
from pathlib import Path

import pandas as pd

# Import the projection engine (same directory)
from projection_engine import (
    ProjectionInputs,
    ProjectionOutputs,
    print_summary,
    run_projections,
    save_results,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# SDC Official Projections (for validation)
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
# Data Loading Functions (adapted for actual file structure)
# =============================================================================


def load_base_population(filepath: Path) -> pd.DataFrame:
    """Load base population from CSV file.

    Adapts from actual file format (county_name, male/female)
    to expected format (county, Male/Female).
    """
    logger.info("Loading base population from: %s", filepath)
    df = pd.read_csv(filepath)

    # Rename columns to match expected format
    if "county_name" in df.columns:
        df = df.rename(columns={"county_name": "county"})

    # Standardize sex values to titlecase
    if "sex" in df.columns:
        df["sex"] = df["sex"].str.title()

    logger.info("Loaded %d rows, total population: %,.0f", len(df), df["population"].sum())
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

    Adapts from actual file format (county_name)
    to expected format (county).

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
    # The projection engine multiplies by PERIOD_LENGTH (5 years)
    # so we need to provide annual rates
    df["fertility_rate"] = df["fertility_rate"] / 5.0

    logger.info("Loaded %d fertility rate entries", len(df))
    logger.info(
        "Fertility rate range: %.4f to %.4f (converted to annual)",
        df["fertility_rate"].min(),
        df["fertility_rate"].max(),
    )
    return df


def load_migration_rates(filepath: Path) -> pd.DataFrame:
    """Load migration rates from CSV file.

    Adapts from actual file format (county_name, male/female)
    to expected format (county, Male/Female).
    """
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


def load_adjustments(filepath: Path) -> pd.DataFrame | None:
    """Load manual adjustments from CSV file.

    Note: The adjustment file doesn't have a 'year' column,
    so we apply the same adjustments to all projection years.
    """
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

    # The adjustment file doesn't have a year column - we need to expand
    # to all projection years (2025, 2030, 2035, 2040, 2045, 2050)
    if "year" not in df.columns:
        logger.info("Expanding adjustments to all projection years")
        projection_years = [2025, 2030, 2035, 2040, 2045, 2050]
        expanded_rows = []
        for year in projection_years:
            year_df = df.copy()
            year_df["year"] = year
            expanded_rows.append(year_df)
        df = pd.concat(expanded_rows, ignore_index=True)

    logger.info("Loaded %d adjustment entries", len(df))
    return df


def load_all_inputs(data_dir: Path) -> ProjectionInputs:
    """Load all projection inputs from the data directory.

    Uses the actual filenames in the data directory.
    """
    base_pop = load_base_population(data_dir / "base_population_by_county.csv")
    survival = load_survival_rates(data_dir / "survival_rates_by_county.csv")
    fertility = load_fertility_rates(data_dir / "fertility_rates_by_county.csv")
    migration = load_migration_rates(data_dir / "migration_rates_by_county.csv")
    adjustments = load_adjustments(data_dir / "adjustment_factors_by_county.csv")

    return ProjectionInputs(
        base_population=base_pop,
        survival_rates=survival,
        fertility_rates=fertility,
        migration_rates=migration,
        adjustments=adjustments,
    )


# =============================================================================
# Comparison Functions
# =============================================================================


def compare_with_sdc_official(outputs: ProjectionOutputs) -> pd.DataFrame:
    """Compare replication results with SDC official projections.

    Args:
        outputs: ProjectionOutputs from the replication run

    Returns:
        DataFrame with comparison results
    """
    comparison_rows = []

    for _, row in outputs.state_totals.iterrows():
        year = int(row["year"])
        replication_pop = row["population"]

        sdc_official = SDC_OFFICIAL_PROJECTIONS.get(year)
        if sdc_official is not None:
            difference = replication_pop - sdc_official
            pct_diff = (difference / sdc_official) * 100

            comparison_rows.append(
                {
                    "year": year,
                    "sdc_official": sdc_official,
                    "replication": round(replication_pop),
                    "difference": round(difference),
                    "pct_difference": round(pct_diff, 2),
                }
            )

    return pd.DataFrame(comparison_rows)


def print_comparison_table(comparison: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("SDC 2024 REPLICATION vs OFFICIAL SDC PROJECTIONS")
    print("=" * 80)

    print(
        "\n{:<8} {:>15} {:>15} {:>12} {:>12}".format(
            "Year", "SDC Official", "Replication", "Difference", "% Diff"
        )
    )
    print("-" * 80)

    for _, row in comparison.iterrows():
        print(
            "{:<8} {:>15,} {:>15,} {:>12,} {:>11.2f}%".format(
                int(row["year"]),
                int(row["sdc_official"]),
                int(row["replication"]),
                int(row["difference"]),
                row["pct_difference"],
            )
        )

    print("-" * 80)

    # Summary statistics
    avg_diff = comparison["difference"].mean()
    avg_pct_diff = comparison["pct_difference"].mean()
    max_abs_diff = comparison["difference"].abs().max()
    max_abs_pct = comparison["pct_difference"].abs().max()

    print("\nSummary Statistics:")
    print(f"  Average difference: {avg_diff:,.0f} ({avg_pct_diff:.2f}%)")
    print(f"  Maximum absolute difference: {max_abs_diff:,.0f}")
    print(f"  Maximum absolute % difference: {max_abs_pct:.2f}%")

    # Trend analysis
    final_diff = comparison[comparison["year"] == 2050]["difference"].values[0]
    if final_diff > 0:
        print("\n  Trend: Replication projects HIGHER than SDC by 2050")
    else:
        print("\n  Trend: Replication projects LOWER than SDC by 2050")

    print("=" * 80)


def save_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Save comparison results to CSV."""
    output_path = output_dir / "sdc_replication_comparison.csv"
    comparison.to_csv(output_path, index=False)
    logger.info("Saved comparison to: %s", output_path)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for running SDC 2024 replication."""
    # Determine paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    output_dir = script_dir.parent / "output"

    print("\n" + "=" * 80)
    print("SDC 2024 REPLICATION: COHORT-COMPONENT PROJECTION")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Verify data directory exists
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all inputs
    print("\n" + "-" * 40)
    print("Loading Input Data")
    print("-" * 40)
    inputs = load_all_inputs(data_dir)

    # Print data summary
    print(f"\n  Counties: {len(inputs.counties)}")
    print(f"  Base year population: {inputs.base_population['population'].sum():,.0f}")
    print(f"  Survival rate entries: {len(inputs.survival_rates)}")
    print(f"  Fertility rate entries: {len(inputs.fertility_rates)}")
    print(f"  Migration rate entries: {len(inputs.migration_rates)}")
    if inputs.adjustments is not None:
        print(f"  Adjustment entries: {len(inputs.adjustments)}")

    # Run projections
    print("\n" + "-" * 40)
    print("Running Projections")
    print("-" * 40)
    outputs = run_projections(inputs)

    # Print summary
    print_summary(outputs)

    # Save results
    save_results(outputs, output_dir)

    # Compare with SDC official
    comparison = compare_with_sdc_official(outputs)
    print_comparison_table(comparison)
    save_comparison(comparison, output_dir)

    print("\nReplication complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
