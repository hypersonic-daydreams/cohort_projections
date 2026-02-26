#!/usr/bin/env python3
"""
Extract County-Level Net Migration from Census PEP Components of Change

Phase 1 of ADR-035 implementation: Extract, harmonize, and validate county-level
net migration data from Census Population Estimates Program (PEP) components of
change files spanning 2000-2024.

Purpose:
- Extract NETMIG, INTERNATIONALMIG, DOMESTICMIG for all ND counties
- Harmonize across three vintage files (2000-2009, 2010-2019, 2020-2024)
- Validate hierarchical consistency (county sums = state totals)
- Create clean, long-format dataset for subsequent analysis

Author: Generated for ADR-035 Phase 1
Date: 2026-02-03
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Constants
ND_FIPS = "38"  # North Dakota state FIPS code
PEP_BASE = Path.home() / "workspace" / "shared-data" / "census" / "popest" / "parquet"
OUTPUT_DIR = Path("data/processed")
STATE_FILE = Path("data/processed/immigration/state_migration_components_2000_2024.csv")

# Vintage file mappings
VINTAGES = {
    "2000-2009": {
        "file": PEP_BASE / "2000-2009/county/co-est2009-alldata.parquet",
        "years": list(range(2000, 2010)),
        "description": "Postcensal estimates (2000-2009)",
    },
    "2010-2019": {
        "file": PEP_BASE / "2010-2019/county/co-est2019-alldata.parquet",
        "years": list(range(2010, 2020)),
        "description": "Postcensal estimates (2010-2019)",
    },
    "2020-2024": {
        "file": PEP_BASE / "2020-2024/county/co-est2024-alldata.parquet",
        "years": list(range(2020, 2025)),
        "description": "Postcensal estimates (2020-2024)",
    },
}


def load_state_totals() -> pd.DataFrame:
    """
    Load state-level migration totals for validation.

    Returns:
        DataFrame with columns: year, state, NETMIG, INTERNATIONALMIG, DOMESTICMIG
    """
    print(f"Loading state-level totals from: {STATE_FILE}")

    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State-level PEP file not found: {STATE_FILE}\n"
            "This file is required for validation. Please ensure it exists."
        )

    df = pd.read_csv(STATE_FILE)

    # Filter for North Dakota only
    df = df[df["NAME"] == "North Dakota"].copy()

    # Select relevant columns (already has 'year' column)
    df = df[["year", "NETMIG", "INTERNATIONALMIG", "DOMESTICMIG"]].copy()

    # Convert year to integer
    df["year"] = df["year"].astype(int)

    print(f"  Loaded {len(df)} years of state-level data ({df['year'].min()}-{df['year'].max()})")

    return df


def extract_vintage(vintage_name: str, vintage_info: dict) -> pd.DataFrame:
    """
    Extract and reshape migration data from a single vintage file.

    Args:
        vintage_name: Name identifier for the vintage (e.g., "2000-2009")
        vintage_info: Dictionary containing file path, years, description

    Returns:
        Long-format DataFrame with columns:
        - year, state_fips, county_fips, state_name, county_name
        - netmig, intl_mig, domestic_mig
        - vintage (for provenance tracking)
    """
    file_path = vintage_info["file"]
    years = vintage_info["years"]

    print(f"\n{'=' * 70}")
    print(f"Processing vintage: {vintage_name}")
    print(f"  Description: {vintage_info['description']}")
    print(f"  File: {file_path}")
    print(f"  Years: {min(years)}-{max(years)}")

    if not file_path.exists():
        raise FileNotFoundError(f"Vintage file not found: {file_path}")

    # Load full vintage file
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Filter for North Dakota counties only (COUNTY != '000' excludes state summary)
    df_nd = df[(df["STATE"] == ND_FIPS) & (df["COUNTY"] != "000")].copy()
    print(f"  Filtered to {len(df_nd)} ND counties")

    if len(df_nd) == 0:
        raise ValueError(f"No North Dakota counties found in {file_path}")

    # Store geographic identifiers
    geo_cols = ["STATE", "COUNTY", "STNAME", "CTYNAME"]
    geo_df = df_nd[geo_cols].copy()

    # Extract migration columns for each year
    records = []

    for year in years:
        year_str = str(year)

        # Column names vary slightly by vintage
        netmig_col = f"NETMIG{year_str}"
        intl_col = f"INTERNATIONALMIG{year_str}"
        dom_col = f"DOMESTICMIG{year_str}"

        # Check if columns exist
        if netmig_col not in df_nd.columns:
            print(f"  WARNING: {netmig_col} not found in {vintage_name}, skipping year {year}")
            continue

        # Extract data for this year
        year_data = geo_df.copy()
        year_data["year"] = year
        year_data["netmig"] = pd.to_numeric(df_nd[netmig_col], errors="coerce")
        year_data["intl_mig"] = (
            pd.to_numeric(df_nd[intl_col], errors="coerce") if intl_col in df_nd.columns else np.nan
        )
        year_data["domestic_mig"] = (
            pd.to_numeric(df_nd[dom_col], errors="coerce") if dom_col in df_nd.columns else np.nan
        )
        year_data["vintage"] = vintage_name

        records.append(year_data)

    # Combine all years for this vintage
    result = pd.concat(records, ignore_index=True)

    print(f"  Extracted {len(result):,} county-year observations")
    print(f"  Missing NETMIG: {result['netmig'].isna().sum()}")
    print(f"  Missing INTLMIG: {result['intl_mig'].isna().sum()}")
    print(f"  Missing DOMESTICMIG: {result['domestic_mig'].isna().sum()}")

    return result


def validate_hierarchical_consistency(
    county_df: pd.DataFrame, state_df: pd.DataFrame, tolerance: float = 0.01
) -> tuple[bool, pd.DataFrame]:
    """
    Validate that county-level net migration sums to state totals within tolerance.

    Args:
        county_df: County-level data with 'year' and 'netmig' columns
        state_df: State-level data with 'year' and 'NETMIG' columns
        tolerance: Acceptable relative error (default 1%)

    Returns:
        Tuple of (all_valid: bool, comparison_df: DataFrame)
    """
    print(f"\n{'=' * 70}")
    print("VALIDATION: Hierarchical Consistency Check")
    print(f"  Tolerance: {tolerance * 100:.1f}%")

    # Sum county-level data by year
    county_sums = county_df.groupby("year")["netmig"].sum().reset_index()
    county_sums = county_sums.rename(columns={"netmig": "county_sum"})

    # Merge with state totals
    comparison = state_df[["year", "NETMIG"]].merge(county_sums, on="year", how="outer")
    comparison = comparison.rename(columns={"NETMIG": "state_total"})

    # Calculate differences
    comparison["difference"] = comparison["county_sum"] - comparison["state_total"]
    comparison["abs_diff"] = comparison["difference"].abs()
    comparison["pct_diff"] = (comparison["difference"] / comparison["state_total"].abs()) * 100
    comparison["abs_pct_diff"] = comparison["pct_diff"].abs()

    # Flag validation failures
    comparison["valid"] = comparison["abs_pct_diff"] <= (tolerance * 100)

    # Summary statistics
    print(f"\n  Years covered: {comparison['year'].min()}-{comparison['year'].max()}")
    print(f"  Total years: {len(comparison)}")
    print(f"  Years with county data: {comparison['county_sum'].notna().sum()}")
    print(f"  Years with state data: {comparison['state_total'].notna().sum()}")
    print(f"  Years passing validation: {comparison['valid'].sum()}")
    print(f"  Years failing validation: {(~comparison['valid']).sum()}")

    # Detailed failure report
    failures = comparison[~comparison["valid"]]
    if len(failures) > 0:
        print(f"\n  VALIDATION FAILURES ({len(failures)} years):")
        for _, row in failures.iterrows():
            print(
                f"    Year {int(row['year'])}: "
                f"County sum = {row['county_sum']:,.0f}, "
                f"State total = {row['state_total']:,.0f}, "
                f"Diff = {row['difference']:,.0f} ({row['pct_diff']:.2f}%)"
            )

    # Overall statistics
    valid_rows = comparison[comparison["valid"]]
    if len(valid_rows) > 0:
        print("\n  Statistics for passing years:")
        print(f"    Mean absolute difference: {valid_rows['abs_diff'].mean():.1f}")
        print(f"    Max absolute difference: {valid_rows['abs_diff'].max():.1f}")
        print(f"    Mean absolute % diff: {valid_rows['abs_pct_diff'].mean():.3f}%")
        print(f"    Max absolute % diff: {valid_rows['abs_pct_diff'].max():.3f}%")

    all_valid = bool(comparison["valid"].all())

    return all_valid, comparison


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the extracted data.

    Args:
        df: Harmonized county-year DataFrame

    Returns:
        DataFrame with summary statistics by period/vintage
    """
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")

    stats_records = []

    # Overall statistics
    stats_records.append(
        {
            "period": "Full Period",
            "years": f"{df['year'].min()}-{df['year'].max()}",
            "n_counties": df["county_fips"].nunique(),
            "n_years": df["year"].nunique(),
            "n_observations": len(df),
            "mean_netmig": df["netmig"].mean(),
            "median_netmig": df["netmig"].median(),
            "total_netmig": df["netmig"].sum(),
            "min_netmig": df["netmig"].min(),
            "max_netmig": df["netmig"].max(),
            "pct_negative": (df["netmig"] < 0).mean() * 100,
        }
    )

    # By vintage
    for vintage in df["vintage"].unique():
        vdf = df[df["vintage"] == vintage]
        stats_records.append(
            {
                "period": vintage,
                "years": f"{vdf['year'].min()}-{vdf['year'].max()}",
                "n_counties": vdf["county_fips"].nunique(),
                "n_years": vdf["year"].nunique(),
                "n_observations": len(vdf),
                "mean_netmig": vdf["netmig"].mean(),
                "median_netmig": vdf["netmig"].median(),
                "total_netmig": vdf["netmig"].sum(),
                "min_netmig": vdf["netmig"].min(),
                "max_netmig": vdf["netmig"].max(),
                "pct_negative": (vdf["netmig"] < 0).mean() * 100,
            }
        )

    # By regime (as defined in ADR-035)
    regimes = [
        ("Pre-Bakken", 2000, 2010),
        ("Bakken Boom", 2011, 2015),
        ("Bust + COVID", 2016, 2021),
        ("Recovery", 2022, 2024),
    ]

    for regime_name, start_year, end_year in regimes:
        rdf = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
        if len(rdf) > 0:
            stats_records.append(
                {
                    "period": regime_name,
                    "years": f"{start_year}-{end_year}",
                    "n_counties": rdf["county_fips"].nunique(),
                    "n_years": rdf["year"].nunique(),
                    "n_observations": len(rdf),
                    "mean_netmig": rdf["netmig"].mean(),
                    "median_netmig": rdf["netmig"].median(),
                    "total_netmig": rdf["netmig"].sum(),
                    "min_netmig": rdf["netmig"].min(),
                    "max_netmig": rdf["netmig"].max(),
                    "pct_negative": (rdf["netmig"] < 0).mean() * 100,
                }
            )

    stats_df = pd.DataFrame(stats_records)

    # Print summary
    print(f"\n{stats_df.to_string(index=False)}")

    return stats_df


def main():
    """Main extraction and validation pipeline."""

    print("=" * 70)
    print("CENSUS PEP COUNTY-LEVEL NET MIGRATION EXTRACTION")
    print("ADR-035 Phase 1: Data Extraction and Validation")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load state-level totals for validation
    state_df = load_state_totals()

    # Step 2: Extract data from each vintage
    vintage_dfs = []
    for vintage_name, vintage_info in VINTAGES.items():
        try:
            vdf = extract_vintage(vintage_name, vintage_info)
            vintage_dfs.append(vdf)
        except Exception as e:
            print(f"\n  ERROR processing {vintage_name}: {e}")
            raise

    # Step 3: Combine all vintages
    print(f"\n{'=' * 70}")
    print("HARMONIZATION: Combining vintages")

    combined_df = pd.concat(vintage_dfs, ignore_index=True)
    print(f"  Total observations: {len(combined_df):,}")
    print(f"  Years: {combined_df['year'].min()}-{combined_df['year'].max()}")
    print(f"  Counties: {combined_df['COUNTY'].nunique()}")

    # Rename columns for consistency
    combined_df = combined_df.rename(
        columns={
            "STATE": "state_fips",
            "COUNTY": "county_fips",
            "STNAME": "state_name",
            "CTYNAME": "county_name",
        }
    )

    # Create full GEOID (state + county)
    combined_df["geoid"] = combined_df["state_fips"] + combined_df["county_fips"]

    # Sort by year and county
    combined_df = combined_df.sort_values(["year", "geoid"]).reset_index(drop=True)

    # Step 4: Validate hierarchical consistency
    all_valid, validation_df = validate_hierarchical_consistency(combined_df, state_df)

    if not all_valid:
        print("\n  WARNING: Some years failed hierarchical validation!")
        print("  Proceeding with data extraction, but investigate failures.")
    else:
        print("\n  ✓ All years passed hierarchical validation!")

    # Step 5: Generate summary statistics
    stats_df = generate_summary_statistics(combined_df)

    # Step 6: Save outputs
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUTS")

    # Main harmonized dataset
    output_file = OUTPUT_DIR / "pep_county_components_2000_2024.parquet"
    combined_df.to_parquet(output_file, index=False, compression="gzip")
    print(f"  ✓ Saved harmonized data: {output_file}")
    print(f"    Rows: {len(combined_df):,}")
    print(f"    Columns: {len(combined_df.columns)}")
    print(f"    Size: {output_file.stat().st_size / 1024:.1f} KB")

    # Also save as CSV for easy inspection
    csv_file = OUTPUT_DIR / "pep_county_components_2000_2024.csv"
    combined_df.to_csv(csv_file, index=False)
    print(f"  ✓ Saved CSV version: {csv_file}")

    # Validation results
    validation_file = OUTPUT_DIR / "pep_county_validation_results.csv"
    validation_df.to_csv(validation_file, index=False)
    print(f"  ✓ Saved validation results: {validation_file}")

    # Summary statistics
    stats_file = OUTPUT_DIR / "pep_county_summary_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"  ✓ Saved summary statistics: {stats_file}")

    # Step 7: Final summary
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nHierarchical Validation: {'PASS ✓' if all_valid else 'FAIL ✗'}")
    print(f"Total County-Year Observations: {len(combined_df):,}")
    print(
        f"Time Coverage: {combined_df['year'].min()}-{combined_df['year'].max()} ({combined_df['year'].nunique()} years)"
    )
    print(f"Counties: {combined_df['county_fips'].nunique()}")
    print(f"Mean Net Migration: {combined_df['netmig'].mean():.1f} per county-year")
    print(f"Total Net Migration (2000-2024): {combined_df['netmig'].sum():,.0f}")
    print("\nOutput files:")
    print(f"  - {output_file}")
    print(f"  - {csv_file}")
    print(f"  - {validation_file}")
    print(f"  - {stats_file}")
    print("\nNext steps: Phase 2 - Regime Analysis")
    print("  See: docs/governance/adrs/035-migration-data-source-census-pep.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"ERROR: {e}")
        print(f"{'=' * 70}")
        sys.exit(1)
