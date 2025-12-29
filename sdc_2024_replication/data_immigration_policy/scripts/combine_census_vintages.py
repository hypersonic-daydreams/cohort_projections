#!/usr/bin/env python3
"""
Combine Census Bureau population components of change data from multiple vintages
into a single time series (2000-2024).

Data Sources:
- NST-EST2009-ALLDATA.csv: Vintage 2009 (2000-2009)
- NST-EST2020-ALLDATA.csv: Vintage 2020 (2010-2020)
- NST-EST2024-ALLDATA.csv: Vintage 2024 (2020-2024)

Key Variables:
- INTERNATIONALMIG: International migration (net)
- DOMESTICMIG: Domestic (internal) migration (net)
- STATE: State FIPS codes
- NAME: State name

Note on Overlapping Years:
- 2020 appears in both NST-EST2020 and NST-EST2024
- We use the later vintage (NST-EST2024) for 2020 as it has revised figures

Output:
- state_migration_components_2000_2024.csv: Long-format time series
"""

from pathlib import Path

import pandas as pd


def load_vintage_data(filepath: Path) -> pd.DataFrame:
    """Load a Census vintage file and return standardized dataframe."""
    df = pd.read_csv(filepath)
    return df


def extract_migration_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Extract column names matching a prefix pattern (e.g., INTERNATIONALMIG)."""
    return [col for col in df.columns if col.startswith(prefix) and col[-4:].isdigit()]


def reshape_to_long(
    df: pd.DataFrame, id_cols: list[str], value_cols: list[str], var_name: str
) -> pd.DataFrame:
    """Reshape wide data to long format, extracting year from column names."""
    # Filter to relevant columns
    df_subset = df[id_cols + value_cols].copy()

    # Melt to long format
    df_long = df_subset.melt(
        id_vars=id_cols, value_vars=value_cols, var_name="variable", value_name=var_name
    )

    # Extract year from variable name (last 4 characters)
    df_long["year"] = df_long["variable"].str[-4:].astype(int)
    df_long = df_long.drop(columns=["variable"])

    return df_long


def process_vintage(filepath: Path, years_to_keep: list[int] | None = None) -> pd.DataFrame:
    """
    Process a single vintage file and return long-format migration data.

    Args:
        filepath: Path to the vintage CSV file
        years_to_keep: Optional list of years to include (for handling overlaps)

    Returns:
        DataFrame with columns: SUMLEV, REGION, DIVISION, STATE, NAME, year,
                                INTERNATIONALMIG, DOMESTICMIG
    """
    df = load_vintage_data(filepath)

    # Standard ID columns
    id_cols = ["SUMLEV", "REGION", "DIVISION", "STATE", "NAME"]

    # Get international migration columns
    intl_cols = extract_migration_columns(df, "INTERNATIONALMIG")
    intl_long = reshape_to_long(df, id_cols, intl_cols, "INTERNATIONALMIG")

    # Get domestic migration columns
    dom_cols = extract_migration_columns(df, "DOMESTICMIG")
    dom_long = reshape_to_long(df, id_cols, dom_cols, "DOMESTICMIG")

    # Merge international and domestic
    result = intl_long.merge(dom_long, on=id_cols + ["year"], how="outer")

    # Filter years if specified
    if years_to_keep is not None:
        result = result[result["year"].isin(years_to_keep)]

    return result


def combine_vintages(source_dir: Path, output_path: Path) -> pd.DataFrame:
    """
    Combine all vintage files into a single time series.

    Strategy:
    - 2000-2009: From NST-EST2009-ALLDATA (Vintage 2009)
    - 2010-2019: From NST-EST2020-ALLDATA (Vintage 2020)
    - 2020-2024: From NST-EST2024-ALLDATA (Vintage 2024)

    Note: 2020 data comes from Vintage 2024 (more recent revision)
    """
    vintages = []

    # Vintage 2009: 2000-2009
    v2009_path = source_dir / "NST-EST2009-ALLDATA.csv"
    if v2009_path.exists():
        print(f"Processing {v2009_path.name}...")
        v2009 = process_vintage(v2009_path, years_to_keep=list(range(2000, 2010)))
        v2009["vintage"] = 2009
        vintages.append(v2009)
        print(f"  Years: {sorted(v2009['year'].unique())}")
        print(f"  Records: {len(v2009)}")
    else:
        print(f"WARNING: {v2009_path} not found")

    # Vintage 2020: 2010-2019 (exclude 2020, will use Vintage 2024 for that)
    v2020_path = source_dir / "NST-EST2020-ALLDATA.csv"
    if v2020_path.exists():
        print(f"Processing {v2020_path.name}...")
        v2020 = process_vintage(v2020_path, years_to_keep=list(range(2010, 2020)))
        v2020["vintage"] = 2020
        vintages.append(v2020)
        print(f"  Years: {sorted(v2020['year'].unique())}")
        print(f"  Records: {len(v2020)}")
    else:
        print(f"WARNING: {v2020_path} not found")

    # Vintage 2024: 2020-2024
    v2024_path = source_dir / "NST-EST2024-ALLDATA.csv"
    if v2024_path.exists():
        print(f"Processing {v2024_path.name}...")
        v2024 = process_vintage(v2024_path, years_to_keep=list(range(2020, 2025)))
        v2024["vintage"] = 2024
        vintages.append(v2024)
        print(f"  Years: {sorted(v2024['year'].unique())}")
        print(f"  Records: {len(v2024)}")
    else:
        print(f"WARNING: {v2024_path} not found")

    # Combine all vintages
    if not vintages:
        raise ValueError("No vintage files found!")

    combined = pd.concat(vintages, ignore_index=True)

    # Sort by state and year
    combined = combined.sort_values(["STATE", "year"]).reset_index(drop=True)

    # Calculate net migration
    combined["NETMIG"] = combined["INTERNATIONALMIG"] + combined["DOMESTICMIG"]

    # Save to output
    combined.to_csv(output_path, index=False)
    print(f"\nSaved combined data to: {output_path}")

    return combined


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics by decade and state."""
    # Filter to state-level data only (SUMLEV 40 or 040)
    # Handle both string and integer types
    states = df[(df["SUMLEV"] == 40) | (df["SUMLEV"] == "040") | (df["SUMLEV"] == "40")].copy()

    # Create decade column
    states["decade"] = (states["year"] // 10) * 10

    # Summarize by state and decade
    summary = (
        states.groupby(["STATE", "NAME", "decade"])
        .agg({"INTERNATIONALMIG": "sum", "DOMESTICMIG": "sum", "NETMIG": "sum", "year": "count"})
        .reset_index()
    )

    summary = summary.rename(columns={"year": "years_count"})

    return summary


def main():
    """Main entry point."""
    # Define paths - Use project-level data directories
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent  # cohort_projections/

    # Raw data source (input)
    source_dir = project_root / "data" / "raw" / "immigration"

    # Processed data output
    output_dir = project_root / "data" / "processed" / "immigration"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine vintages
    output_path = output_dir / "state_migration_components_2000_2024.csv"
    combined = combine_vintages(source_dir, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal records: {len(combined)}")
    print(f"Year range: {combined['year'].min()}-{combined['year'].max()}")
    print(f"Unique states/regions: {combined['STATE'].nunique()}")

    # Check for any gaps
    all_years = set(range(2000, 2025))
    present_years = set(combined["year"].unique())
    missing = all_years - present_years
    if missing:
        print(f"WARNING: Missing years: {sorted(missing)}")
    else:
        print("All years 2000-2024 present")

    # Show sample for North Dakota (STATE=38)
    print("\n" + "-" * 60)
    print("Sample: North Dakota (FIPS 38)")
    print("-" * 60)
    nd = combined[(combined["STATE"] == "38") | (combined["STATE"] == 38)]
    if len(nd) > 0:
        print(
            nd[["year", "INTERNATIONALMIG", "DOMESTICMIG", "NETMIG", "vintage"]].to_string(
                index=False
            )
        )

    # Show national totals
    print("\n" + "-" * 60)
    print("National Migration Totals by Year")
    print("-" * 60)
    national = combined[(combined["STATE"] == "00") | (combined["STATE"] == 0)]
    if len(national) > 0:
        print(
            national[["year", "INTERNATIONALMIG", "DOMESTICMIG", "NETMIG", "vintage"]].to_string(
                index=False
            )
        )

    # Generate decade summary
    summary_path = output_dir / "state_migration_decade_summary.csv"
    summary = generate_summary_statistics(combined)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved decade summary to: {summary_path}")

    return combined


if __name__ == "__main__":
    main()
