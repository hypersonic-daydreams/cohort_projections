# mypy: ignore-errors
"""
Process DHS Yearbook of Immigration Statistics LPR data.

This script processes:
1. Table 3: LPR by region and country of birth (FY 2014-2023)
2. Table 4: LPR by state (FY 2014-2023)
3. LPRSuppTable 1: LPR by state and country of birth (FY 2023)

Outputs a cleaned parquet file with focus on North Dakota analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase with underscores."""
    df.columns = [
        str(c).lower().replace(" ", "_").replace("-", "_").replace("/", "_") for c in df.columns
    ]
    return df


def process_table3_country_of_birth(xl: pd.ExcelFile) -> pd.DataFrame:
    """Process Table 3: LPR by region and country of birth over time."""
    df = pd.read_excel(xl, sheet_name="Table 3", header=5)

    # Clean column names
    df.columns = ["region_country_of_birth"] + list(range(2014, 2024))

    # Remove header rows (REGION, COUNTRY markers)
    df = df[~df["region_country_of_birth"].isin(["REGION", "COUNTRY", np.nan])]
    df = df.dropna(subset=["region_country_of_birth"])

    # Melt to long format
    df_long = df.melt(
        id_vars=["region_country_of_birth"], var_name="fiscal_year", value_name="lpr_count"
    )

    # Clean values
    df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")

    # Add type indicator
    regions = [
        "Total",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
        "Unknown",
    ]
    df_long["is_region"] = df_long["region_country_of_birth"].isin(regions)

    return df_long


def process_table4_state_time_series(xl: pd.ExcelFile) -> pd.DataFrame:
    """Process Table 4: LPR by state over time (FY 2014-2023)."""
    df = pd.read_excel(xl, sheet_name="Table 4", header=5)

    # Clean column names
    df.columns = ["state_or_territory"] + list(range(2014, 2024))

    # Remove non-state rows
    df = df[~df["state_or_territory"].isin([np.nan, "Total", "Unknown", "Other"])]
    df = df.dropna(subset=["state_or_territory"])

    # Clean up special entries
    exclude_patterns = [
        "Note",
        "Source",
        "Return",
        "Table",
        "Armed Services",
        "Territories",
        "Virgin Islands",
    ]
    for pattern in exclude_patterns:
        df = df[~df["state_or_territory"].str.contains(pattern, case=False, na=False)]

    # Melt to long format
    df_long = df.melt(
        id_vars=["state_or_territory"], var_name="fiscal_year", value_name="lpr_count"
    )

    df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")

    return df_long


def process_supptable1_state_country(xl: pd.ExcelFile) -> pd.DataFrame:
    """Process LPRSuppTable 1: LPR by state and country of birth (FY 2023)."""
    df = pd.read_excel(xl, sheet_name="LPRSuppTable 1", header=5)

    # First column is country/region
    df = df.rename(columns={df.columns[0]: "region_country_of_birth"})

    # Remove marker rows
    df = df[~df["region_country_of_birth"].isin(["REGION", "COUNTRY", np.nan])]
    df = df.dropna(subset=["region_country_of_birth"])

    # Exclude notes/source rows
    exclude_patterns = ["Note", "Source", "Return", "Table", "D "]
    for pattern in exclude_patterns:
        df = df[
            ~df["region_country_of_birth"].astype(str).str.contains(pattern, case=False, na=False)
        ]

    # Get state columns (exclude Total and Unknown)
    state_cols = [
        c
        for c in df.columns[1:]
        if c
        not in [
            "Total",
            "Unknown",
            "U.S. Armed Services Posts",
            "U.S. Territories1",
            "Guam",
            "Puerto Rico",
        ]
    ]

    # Melt to long format with state as column
    df_long = df.melt(
        id_vars=["region_country_of_birth"],
        value_vars=state_cols,
        var_name="state",
        value_name="lpr_count",
    )

    df_long["fiscal_year"] = 2023
    df_long["lpr_count"] = pd.to_numeric(df_long["lpr_count"], errors="coerce")

    # Add type indicator
    regions = [
        "Total",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
        "Unknown",
    ]
    df_long["is_region"] = df_long["region_country_of_birth"].isin(regions)

    return df_long


def compute_nd_share_analysis(df_state_country: pd.DataFrame) -> pd.DataFrame:
    """Compute North Dakota's share of LPRs by country of origin."""
    # Get ND data
    nd_data = df_state_country[df_state_country["state"] == "North Dakota"].copy()
    nd_data = nd_data.rename(columns={"lpr_count": "nd_lpr_count"})

    # Get total US by country (sum across all states)
    us_totals = df_state_country.groupby("region_country_of_birth")["lpr_count"].sum().reset_index()
    us_totals = us_totals.rename(columns={"lpr_count": "us_total_lpr_count"})

    # Merge
    nd_share = nd_data.merge(us_totals, on="region_country_of_birth")

    # Calculate share
    nd_share["nd_share_pct"] = (
        nd_share["nd_lpr_count"] / nd_share["us_total_lpr_count"] * 100
    ).round(3)

    # Sort by ND count
    nd_share = nd_share.sort_values("nd_lpr_count", ascending=False)

    return nd_share[
        [
            "region_country_of_birth",
            "is_region",
            "nd_lpr_count",
            "us_total_lpr_count",
            "nd_share_pct",
            "fiscal_year",
        ]
    ]


def main():
    # Project root directory
    project_root = Path("/home/nigel/cohort_projections")

    # Raw data source (input)
    source_path = project_root / "data" / "raw" / "immigration" / "dhs_yearbook"

    # Processed data output
    output_path = project_root / "data" / "processed" / "immigration" / "analysis"
    output_path.mkdir(parents=True, exist_ok=True)

    yearbook_file = source_path / "yearbook_lpr_2023_all_tables.xlsx"

    print("Processing DHS Yearbook LPR data...")
    xl = pd.ExcelFile(yearbook_file)

    # Process each table
    print("\n1. Processing Table 3: LPR by region/country of birth (FY 2014-2023)")
    df_country_time = process_table3_country_of_birth(xl)
    print(f"   Rows: {len(df_country_time):,}")

    print("\n2. Processing Table 4: LPR by state (FY 2014-2023)")
    df_state_time = process_table4_state_time_series(xl)
    print(f"   Rows: {len(df_state_time):,}")

    print("\n3. Processing LPRSuppTable 1: LPR by state and country (FY 2023)")
    df_state_country = process_supptable1_state_country(xl)
    print(f"   Rows: {len(df_state_country):,}")

    # Create combined output
    print("\n4. Creating combined dataset...")

    # Save individual datasets
    df_country_time.to_parquet(output_path / "dhs_lpr_by_country_time.parquet", index=False)
    df_state_time.to_parquet(output_path / "dhs_lpr_by_state_time.parquet", index=False)
    df_state_country.to_parquet(output_path / "dhs_lpr_by_state_country.parquet", index=False)

    # Compute ND analysis
    print("\n5. Computing North Dakota share analysis...")
    nd_share = compute_nd_share_analysis(df_state_country)
    nd_share.to_parquet(output_path / "dhs_lpr_nd_share_by_country.parquet", index=False)

    # Create summary statistics
    print("\n=== SUMMARY STATISTICS ===")

    # ND time series
    nd_time = df_state_time[df_state_time["state_or_territory"] == "North Dakota"]
    print("\nNorth Dakota LPR Admissions by Fiscal Year:")
    print(nd_time[["fiscal_year", "lpr_count"]].to_string(index=False))

    # ND top countries of origin (FY 2023)
    print("\nTop 20 Countries of Origin for ND LPRs (FY 2023):")
    nd_top = nd_share[~nd_share["is_region"]].head(20)
    print(
        nd_top[
            ["region_country_of_birth", "nd_lpr_count", "us_total_lpr_count", "nd_share_pct"]
        ].to_string(index=False)
    )

    # ND by region (FY 2023)
    print("\nND LPRs by Region of Birth (FY 2023):")
    nd_regions = nd_share[nd_share["is_region"]]
    nd_regions = nd_regions[nd_regions["region_country_of_birth"] != "Total"]
    print(
        nd_regions[["region_country_of_birth", "nd_lpr_count", "nd_share_pct"]].to_string(
            index=False
        )
    )

    print("\n=== FILES SAVED ===")
    print(f"  - {output_path / 'dhs_lpr_by_country_time.parquet'}")
    print(f"  - {output_path / 'dhs_lpr_by_state_time.parquet'}")
    print(f"  - {output_path / 'dhs_lpr_by_state_country.parquet'}")
    print(f"  - {output_path / 'dhs_lpr_nd_share_by_country.parquet'}")


if __name__ == "__main__":
    main()
