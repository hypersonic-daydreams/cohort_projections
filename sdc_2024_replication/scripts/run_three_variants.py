#!/usr/bin/env python3
"""
SDC 2024 Replication: Run All Three Methodology Variants and Compare

This script runs all three variants of the SDC 2024 population projection methodology:
1. Original Data Variant: Uses data from data/ directory, base year 2020
2. Updated Data Variant: Uses data from data_updated/ directory, base year 2024
3. Immigration Policy Variant: Uses data from data_immigration_policy/rates/, base year 2024

Author: SDC 2024 Replication Project
Date: 2025-12-28
"""
# mypy: disable-error-code="assignment"

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# Import from run_both_variants
from run_both_variants import (
    SDC_OFFICIAL_PROJECTIONS,
    load_all_inputs,
    run_projections_with_base_year,
)

# Configure logging - reduce verbosity
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_three_way_comparison(
    original_outputs,
    updated_outputs,
    policy_outputs,
) -> pd.DataFrame:
    """Create a comparison table with all three variants plus SDC official."""
    comparison_rows = []

    # Get all unique years
    all_years = set(SDC_OFFICIAL_PROJECTIONS.keys())
    all_years.update(original_outputs.state_totals["year"].astype(int).tolist())
    all_years.update(updated_outputs.state_totals["year"].astype(int).tolist())
    all_years.update(policy_outputs.state_totals["year"].astype(int).tolist())
    all_years = sorted(all_years)

    for year in all_years:
        row = {"year": year}

        # SDC Official
        row["sdc_official"] = SDC_OFFICIAL_PROJECTIONS.get(year)

        # Original data variant
        original_row = original_outputs.state_totals[original_outputs.state_totals["year"] == year]
        row["original_2020"] = (
            round(original_row["population"].values[0]) if not original_row.empty else None
        )

        # Updated data variant
        updated_row = updated_outputs.state_totals[updated_outputs.state_totals["year"] == year]
        row["updated_2024"] = (
            round(updated_row["population"].values[0]) if not updated_row.empty else None
        )

        # Immigration policy variant
        policy_row = policy_outputs.state_totals[policy_outputs.state_totals["year"] == year]
        row["immigration_policy"] = (
            round(policy_row["population"].values[0]) if not policy_row.empty else None
        )

        comparison_rows.append(row)

    df = pd.DataFrame(comparison_rows)

    # Add difference columns
    df["policy_vs_updated"] = df["immigration_policy"] - df["updated_2024"]
    df["policy_vs_sdc"] = df["immigration_policy"] - df["sdc_official"]

    return df


def print_three_way_comparison(comparison: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print("THREE-VARIANT COMPARISON: SDC Official vs Original vs Updated vs Immigration Policy")
    print("=" * 110)

    print(
        "\n{:<6} {:>12} {:>14} {:>14} {:>16} {:>14} {:>14}".format(
            "Year",
            "SDC Official",
            "Original 2020",
            "Updated 2024",
            "Immig. Policy",
            "Policy-Updated",
            "Policy-SDC",
        )
    )
    print("-" * 110)

    for _, row in comparison.iterrows():
        year = int(row["year"])
        sdc = row["sdc_official"]
        orig = row["original_2020"]
        upd = row["updated_2024"]
        pol = row["immigration_policy"]
        pol_upd = row["policy_vs_updated"]
        pol_sdc = row["policy_vs_sdc"]

        sdc_str = f"{int(sdc):,}" if pd.notna(sdc) else ""
        orig_str = f"{int(orig):,}" if pd.notna(orig) else ""
        upd_str = f"{int(upd):,}" if pd.notna(upd) else ""
        pol_str = f"{int(pol):,}" if pd.notna(pol) else ""
        pol_upd_str = f"{int(pol_upd):+,}" if pd.notna(pol_upd) else ""
        pol_sdc_str = f"{int(pol_sdc):+,}" if pd.notna(pol_sdc) else ""

        print(
            f"{year:<6} {sdc_str:>12} {orig_str:>14} {upd_str:>14} "
            f"{pol_str:>16} {pol_upd_str:>14} {pol_sdc_str:>14}"
        )

    print("-" * 110)

    # Summary for 2050
    row_2050 = comparison[comparison["year"] == 2050].iloc[0]
    print("\n2050 SUMMARY:")
    print(f"  SDC Official:        {int(row_2050['sdc_official']):>12,}")
    if pd.notna(row_2050["original_2020"]):
        print(f"  Original (2020):     {int(row_2050['original_2020']):>12,}")
    if pd.notna(row_2050["updated_2024"]):
        print(f"  Updated (2024):      {int(row_2050['updated_2024']):>12,}")
    if pd.notna(row_2050["immigration_policy"]):
        print(f"  Immigration Policy:  {int(row_2050['immigration_policy']):>12,}")

    print("\n  Immigration Policy Impact (vs Updated 2024):")
    if pd.notna(row_2050["policy_vs_updated"]):
        diff = row_2050["policy_vs_updated"]
        pct = diff / row_2050["updated_2024"] * 100 if row_2050["updated_2024"] else 0
        print(f"    2050: {int(diff):+,} ({pct:+.1f}%)")

    print("=" * 110)


def main() -> None:
    """Run all three variants and compare results."""
    # Determine paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent  # sdc_2024_replication/
    project_root = base_dir.parent  # cohort_projections/

    original_data_dir = base_dir / "data"
    updated_data_dir = base_dir / "data_updated"
    # Immigration policy rates are now in project-level processed data
    policy_data_dir = project_root / "data" / "processed" / "immigration" / "rates"
    output_dir = base_dir / "output"

    # Configuration
    original_base_year = 2020
    updated_base_year = 2024
    end_year = 2050

    print("\n" + "=" * 110)
    print("SDC 2024 REPLICATION: RUNNING ALL THREE METHODOLOGY VARIANTS")
    print("=" * 110)

    print("\nConfiguration:")
    print(f"  Original Data Directory: {original_data_dir}")
    print(f"  Updated Data Directory:  {updated_data_dir}")
    print(f"  Policy Data Directory:   {policy_data_dir}")
    print(f"  End Year: {end_year}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Run Original Data Variant
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running VARIANT 1: Original Data (Base Year 2020)...")

    original_projection_years = list(range(original_base_year + 5, end_year + 1, 5))
    original_inputs = load_all_inputs(original_data_dir, original_projection_years)

    print(f"  Counties: {len(original_inputs.counties)}")
    print(f"  Base population: {original_inputs.base_population['population'].sum():,.0f}")

    original_outputs = run_projections_with_base_year(
        inputs=original_inputs,
        base_year=original_base_year,
        end_year=end_year,
    )
    print(
        f"  2050 projection: {original_outputs.state_totals[original_outputs.state_totals['year']==2050]['population'].values[0]:,.0f}"
    )

    # =========================================================================
    # Run Updated Data Variant
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running VARIANT 2: Updated Data (Base Year 2024)...")

    updated_projection_years = [2025, 2030, 2035, 2040, 2045, 2050]
    updated_inputs = load_all_inputs(updated_data_dir, updated_projection_years)

    print(f"  Counties: {len(updated_inputs.counties)}")
    print(f"  Base population: {updated_inputs.base_population['population'].sum():,.0f}")

    updated_outputs = run_projections_with_base_year(
        inputs=updated_inputs,
        base_year=updated_base_year,
        end_year=end_year,
        projection_years=updated_projection_years,
    )
    print(
        f"  2050 projection: {updated_outputs.state_totals[updated_outputs.state_totals['year']==2050]['population'].values[0]:,.0f}"
    )

    # =========================================================================
    # Run Immigration Policy Variant
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running VARIANT 3: Immigration Policy (Base Year 2024)...")

    policy_inputs = load_all_inputs(policy_data_dir, updated_projection_years)

    print(f"  Counties: {len(policy_inputs.counties)}")
    print(f"  Base population: {policy_inputs.base_population['population'].sum():,.0f}")

    policy_outputs = run_projections_with_base_year(
        inputs=policy_inputs,
        base_year=updated_base_year,
        end_year=end_year,
        projection_years=updated_projection_years,
    )
    print(
        f"  2050 projection: {policy_outputs.state_totals[policy_outputs.state_totals['year']==2050]['population'].values[0]:,.0f}"
    )

    # =========================================================================
    # Create and Save Comparison
    # =========================================================================
    comparison = create_three_way_comparison(
        original_outputs=original_outputs,
        updated_outputs=updated_outputs,
        policy_outputs=policy_outputs,
    )

    print_three_way_comparison(comparison)

    # Save results
    output_path = output_dir / "three_variant_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"\nSaved comparison to: {output_path}")

    # Save original variant state totals
    original_state_file = output_dir / "original_variant_state_totals.csv"
    original_outputs.state_totals.to_csv(original_state_file, index=False)
    print(f"Saved original variant state totals to: {original_state_file}")

    # Save updated variant state totals
    updated_state_file = output_dir / "updated_variant_state_totals.csv"
    updated_outputs.state_totals.to_csv(updated_state_file, index=False)
    print(f"Saved updated variant state totals to: {updated_state_file}")

    # Save policy variant state totals
    policy_state_file = output_dir / "policy_variant_state_totals.csv"
    policy_outputs.state_totals.to_csv(policy_state_file, index=False)
    print(f"Saved policy variant state totals to: {policy_state_file}")

    print("\n" + "=" * 60)
    print("THREE-VARIANT COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
