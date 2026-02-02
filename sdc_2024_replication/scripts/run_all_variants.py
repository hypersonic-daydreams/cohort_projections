#!/usr/bin/env python3
"""
Run SDC 2024 methodology variants and compare results.

This is the RECOMMENDED entry point for running SDC replication variants.

Variants:
1. Original: SDC methodology with original 2020 data
2. Updated: SDC methodology with 2024 Census + 2023 CDC data
3. Immigration Policy: Updated data with CBO-based migration adjustment

Usage:
    # Run all variants (default)
    python run_all_variants.py
    python run_all_variants.py --all

    # Run specific variant(s)
    python run_all_variants.py --variant original
    python run_all_variants.py --variant updated
    python run_all_variants.py --variant policy
    python run_all_variants.py --variant original --variant updated

    # List available variants
    python run_all_variants.py --list

Author: SDC 2024 Replication Project
Date: 2025-12-28
"""
# mypy: disable-error-code="arg-type,union-attr,assignment"

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from projection_engine import (  # noqa: E402
    PERIOD_MULTIPLIERS,
    ProjectionInputs,
    run_projections,
)

# Available variants with their descriptions
AVAILABLE_VARIANTS = {
    "original": "Original SDC methodology with 2020 data",
    "updated": "Updated methodology with 2024 Census + 2023 CDC data",
    "policy": "Immigration policy variant with CBO-based migration adjustment",
}

# Paths
BASE_DIR = SCRIPTS_DIR.parent  # sdc_2024_replication/
PROJECT_ROOT = BASE_DIR.parent  # cohort_projections/

ORIGINAL_DATA_DIR = BASE_DIR / "data"
UPDATED_DATA_DIR = BASE_DIR / "data_updated"
# Immigration policy rates are now in project-level processed data
POLICY_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "rates"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_projection_inputs(data_dir: Path) -> ProjectionInputs:
    """Load projection inputs from a data directory."""
    # Load CSVs
    base_pop = pd.read_csv(data_dir / "base_population_by_county.csv")
    fertility = pd.read_csv(data_dir / "fertility_rates_by_county.csv")
    survival = pd.read_csv(data_dir / "survival_rates_by_county.csv")
    migration = pd.read_csv(data_dir / "migration_rates_by_county.csv")

    # Rename columns to match expected format
    base_pop = base_pop.rename(columns={"county_name": "county"})
    fertility = fertility.rename(columns={"county_name": "county"})
    migration = migration.rename(columns={"county_name": "county"})

    # Load adjustments if available and properly formatted
    adjustments_path = data_dir / "adjustment_factors_by_county.csv"
    adjustments = None
    if adjustments_path.exists():
        adj_df = pd.read_csv(adjustments_path)
        adj_df = adj_df.rename(columns={"county_name": "county"})
        # Only use if it has the required 'year' column
        if "year" in adj_df.columns:
            adjustments = adj_df
        # else: skip adjustments (file doesn't have year column)

    return ProjectionInputs(
        base_population=base_pop,
        survival_rates=survival,
        fertility_rates=fertility,
        migration_rates=migration,
        adjustments=adjustments,
    )


def update_period_multipliers(new_multipliers: dict[int, float]) -> None:
    """Temporarily update the global period multipliers."""
    PERIOD_MULTIPLIERS.clear()
    PERIOD_MULTIPLIERS.update(new_multipliers)


def run_variant(
    name: str,
    data_dir: Path,
    period_multipliers: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Run projections for a variant and return state totals."""
    print(f"\n{'=' * 60}")
    print(f"Running variant: {name}")
    print(f"{'=' * 60}")
    print(f"Data directory: {data_dir}")

    # Store original multipliers
    original_multipliers = PERIOD_MULTIPLIERS.copy()

    try:
        # Update multipliers if specified
        if period_multipliers is not None:
            print("Using custom period multipliers:")
            for year, mult in sorted(period_multipliers.items()):
                print(f"  {year}: {mult:.4f}")
            update_period_multipliers(period_multipliers)

        # Load inputs
        print("Loading projection inputs...")
        inputs = load_projection_inputs(data_dir)
        print(f"  Counties: {len(inputs.counties)}")

        # Run projections
        print("Running projections...")
        outputs = run_projections(inputs)

        # Get state totals
        state_totals = outputs.state_totals.copy()
        state_totals["variant"] = name

        print("  State population by year:")
        for _, row in state_totals.iterrows():
            print(f"    {int(row['year'])}: {int(row['population']):,}")

        return state_totals

    finally:
        # Restore original multipliers
        update_period_multipliers(original_multipliers)


def load_sdc_official() -> pd.DataFrame:
    """Load SDC official projections for comparison."""
    # These are from the SDC 2024 report
    sdc_official = pd.DataFrame(
        {
            "year": [2020, 2025, 2030, 2035, 2040, 2045, 2050],
            "population": [779094, 796989, 831543, 865397, 890424, 925101, 957194],
            "variant": "SDC Official",
        }
    )
    return sdc_official


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SDC 2024 methodology variants and compare results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all variants (default)
    python run_all_variants.py
    python run_all_variants.py --all

    # Run specific variant(s)
    python run_all_variants.py --variant original
    python run_all_variants.py --variant updated
    python run_all_variants.py --variant policy
    python run_all_variants.py --variant original --variant updated

    # List available variants
    python run_all_variants.py --list
        """,
    )

    parser.add_argument(
        "--variant",
        "-v",
        action="append",
        choices=list(AVAILABLE_VARIANTS.keys()),
        dest="variants",
        help="Variant(s) to run. Can be specified multiple times.",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all available variants (default if no variants specified).",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available variants and exit.",
    )

    return parser.parse_args()


def main():
    """Run all variants and create comparison."""
    args = parse_args()

    # Handle --list flag
    if args.list:
        print("\nAvailable variants:")
        for name, description in AVAILABLE_VARIANTS.items():
            print(f"  {name:<12} - {description}")
        print("\nUse --variant <name> to run specific variant(s), or --all to run all.")
        return None

    # Determine which variants to run (default: all variants)
    variants_to_run = set(args.variants) if args.variants else set(AVAILABLE_VARIANTS.keys())

    print("=" * 60)
    print("SDC 2024 Methodology: Variant Comparison")
    print("=" * 60)
    print(f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Running variants: {', '.join(sorted(variants_to_run))}")

    # Check for policy-adjusted period multipliers
    policy_multipliers_path = POLICY_DATA_DIR / "period_multipliers.json"
    policy_multipliers = None
    if policy_multipliers_path.exists():
        with open(policy_multipliers_path) as f:
            pm_data = json.load(f)
            # Convert string keys to int
            policy_multipliers = {int(k): v for k, v in pm_data["policy_adjusted"].items()}
            if "policy" in variants_to_run:
                print("\nLoaded policy-adjusted period multipliers:")
                for year, mult in policy_multipliers.items():
                    print(f"  {year}: {mult:.4f}")

    # Run each variant
    results = []

    # 1. Original data variant
    if "original" in variants_to_run:
        if (
            ORIGINAL_DATA_DIR.exists()
            and (ORIGINAL_DATA_DIR / "base_population_by_county.csv").exists()
        ):
            original = run_variant("Original (2020 data)", ORIGINAL_DATA_DIR)
            results.append(original)
        else:
            print(f"\nSkipping original variant - data not found at {ORIGINAL_DATA_DIR}")

    # 2. Updated data variant
    if "updated" in variants_to_run:
        if (
            UPDATED_DATA_DIR.exists()
            and (UPDATED_DATA_DIR / "base_population_by_county.csv").exists()
        ):
            updated = run_variant("Updated (2024 data)", UPDATED_DATA_DIR)
            results.append(updated)
        else:
            print(f"\nSkipping updated variant - data not found at {UPDATED_DATA_DIR}")

    # 3. Immigration policy variant
    if "policy" in variants_to_run:
        if (
            POLICY_DATA_DIR.exists()
            and (POLICY_DATA_DIR / "base_population_by_county.csv").exists()
        ):
            # Use policy-adjusted multipliers if available
            policy = run_variant(
                "Immigration Policy",
                POLICY_DATA_DIR,
                period_multipliers=policy_multipliers,
            )
            results.append(policy)
        else:
            print(f"\nSkipping policy variant - data not found at {POLICY_DATA_DIR}")

    # Check if we have any results
    if not results:
        print("\nNo variants could be run. Check that data directories exist.")
        return None

    # Add SDC official for comparison
    sdc_official = load_sdc_official()
    results.append(sdc_official)

    # Combine all results
    combined = pd.concat(results, ignore_index=True)

    # Pivot for comparison
    comparison = combined.pivot(index="year", columns="variant", values="population")
    comparison = comparison.reset_index()

    # Calculate differences
    if "Updated (2024 data)" in comparison.columns and "Original (2020 data)" in comparison.columns:
        comparison["Updated vs Original"] = (
            comparison["Updated (2024 data)"] - comparison["Original (2020 data)"]
        )

    if "Immigration Policy" in comparison.columns and "Updated (2024 data)" in comparison.columns:
        comparison["Policy vs Updated"] = (
            comparison["Immigration Policy"] - comparison["Updated (2024 data)"]
        )

    if "Immigration Policy" in comparison.columns and "SDC Official" in comparison.columns:
        comparison["Policy vs SDC"] = comparison["Immigration Policy"] - comparison["SDC Official"]

    # Save comparison
    comparison.to_csv(OUTPUT_DIR / "three_variant_comparison.csv", index=False)
    print("\n\nSaved: three_variant_comparison.csv")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    # Format for display
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    print(comparison.to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Get 2050 values for summary
    row_2050 = comparison[comparison["year"] == 2050].iloc[0]

    if "SDC Official" in row_2050:
        print("\n2050 Projections:")
        print(f"  SDC Official:        {row_2050['SDC Official']:>12,.0f}")

    if "Original (2020 data)" in row_2050:
        print(f"  Original (2020):     {row_2050['Original (2020 data)']:>12,.0f}")

    if "Updated (2024 data)" in row_2050:
        print(f"  Updated (2024):      {row_2050['Updated (2024 data)']:>12,.0f}")

    if "Immigration Policy" in row_2050:
        print(f"  Immigration Policy:  {row_2050['Immigration Policy']:>12,.0f}")

    # Calculate percentage differences
    print("\nDifferences from SDC Official (2050):")
    if "Original (2020 data)" in row_2050 and "SDC Official" in row_2050:
        diff = row_2050["Original (2020 data)"] - row_2050["SDC Official"]
        pct = diff / row_2050["SDC Official"] * 100
        print(f"  Original:  {diff:>+12,.0f} ({pct:>+.1f}%)")

    if "Updated (2024 data)" in row_2050 and "SDC Official" in row_2050:
        diff = row_2050["Updated (2024 data)"] - row_2050["SDC Official"]
        pct = diff / row_2050["SDC Official"] * 100
        print(f"  Updated:   {diff:>+12,.0f} ({pct:>+.1f}%)")

    if "Immigration Policy" in row_2050 and "SDC Official" in row_2050:
        diff = row_2050["Immigration Policy"] - row_2050["SDC Official"]
        pct = diff / row_2050["SDC Official"] * 100
        print(f"  Policy:    {diff:>+12,.0f} ({pct:>+.1f}%)")

    # Immigration policy impact
    if "Immigration Policy" in row_2050 and "Updated (2024 data)" in row_2050:
        diff = row_2050["Immigration Policy"] - row_2050["Updated (2024 data)"]
        pct = diff / row_2050["Updated (2024 data)"] * 100
        print("\nImmigration Policy Impact (vs Updated):")
        print(f"  2050: {diff:>+12,.0f} ({pct:>+.1f}%)")

    return comparison


if __name__ == "__main__":
    main()
