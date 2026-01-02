#!/usr/bin/env python3
"""
Process IRS county-to-county migration data for North Dakota counties.

This script reads IRS migration inflow and outflow files and produces
a clean dataset of net migration for North Dakota counties.

Output: data/raw/migration/nd_migration_processed.csv
"""

from pathlib import Path

import pandas as pd

# Configuration
ND_STATE_FIPS = 38
BASE_DIR = Path(__file__).resolve().parents[1]  # scripts -> cohort_projections
MIGRATION_DIR = BASE_DIR / "data" / "raw" / "migration"

# File mappings: filename suffix to migration year
FILE_YEARS = {
    "1819": 2019,  # Tax year 2018-2019 represents migration in 2019
    "1920": 2020,
    "2021": 2021,
    "2122": 2022,
}

# North Dakota county FIPS codes and names
ND_COUNTIES = {
    1: "Adams",
    3: "Barnes",
    5: "Benson",
    7: "Billings",
    9: "Bottineau",
    11: "Bowman",
    13: "Burke",
    15: "Burleigh",
    17: "Cass",
    19: "Cavalier",
    21: "Dickey",
    23: "Divide",
    25: "Dunn",
    27: "Eddy",
    29: "Emmons",
    31: "Foster",
    33: "Golden Valley",
    35: "Grand Forks",
    37: "Grant",
    39: "Griggs",
    41: "Hettinger",
    43: "Kidder",
    45: "LaMoure",
    47: "Logan",
    49: "McHenry",
    51: "McIntosh",
    53: "McKenzie",
    55: "McLean",
    57: "Mercer",
    59: "Morton",
    61: "Mountrail",
    63: "Nelson",
    65: "Oliver",
    67: "Pembina",
    69: "Pierce",
    71: "Ramsey",
    73: "Ransom",
    75: "Renville",
    77: "Richland",
    79: "Rolette",
    81: "Sargent",
    83: "Sheridan",
    85: "Sioux",
    87: "Slope",
    89: "Stark",
    91: "Steele",
    93: "Stutsman",
    95: "Towner",
    97: "Traill",
    99: "Walsh",
    101: "Ward",
    103: "Wells",
    105: "Williams",
}


def process_inflows(file_path: Path, year: int) -> pd.DataFrame:
    """
    Process an inflow file to extract North Dakota inflows.

    Inflow files have y2_statefips as the destination state.
    We need rows where y2_statefips=38 (ND is destination).
    """
    df = pd.read_csv(
        file_path,
        encoding="latin-1",
        dtype={
            "y2_statefips": int,
            "y2_countyfips": int,
            "y1_statefips": int,
            "y1_countyfips": int,
            "n1": int,
            "n2": int,
            "agi": int,
        },
    )

    # Filter to ND destinations only
    nd_inflows = df[df["y2_statefips"] == ND_STATE_FIPS].copy()

    results = []

    for county_fips in ND_COUNTIES:
        county_data = nd_inflows[nd_inflows["y2_countyfips"] == county_fips]

        # Total inflow (US + Foreign)
        total_row = county_data[
            (county_data["y1_statefips"] == 96) & (county_data["y1_countyfips"] == 0)
        ]

        # US only inflow (domestic)
        us_row = county_data[
            (county_data["y1_statefips"] == 97) & (county_data["y1_countyfips"] == 0)
        ]

        total_n2 = total_row["n2"].values[0] if len(total_row) > 0 else 0
        us_n2 = us_row["n2"].values[0] if len(us_row) > 0 else 0

        # Handle suppressed values (-1)
        if total_n2 == -1:
            total_n2 = 0
        if us_n2 == -1:
            us_n2 = 0

        results.append(
            {
                "county_fips": int(f"{ND_STATE_FIPS}{county_fips:03d}"),
                "county_fips_3": county_fips,
                "year": year,
                "inflow_n2": total_n2,
                "inflow_domestic": us_n2,
            }
        )

    return pd.DataFrame(results)


def process_outflows(file_path: Path, year: int) -> pd.DataFrame:
    """
    Process an outflow file to extract North Dakota outflows.

    Outflow files have y1_statefips as the origin state.
    We need rows where y1_statefips=38 (ND is origin).
    """
    df = pd.read_csv(
        file_path,
        encoding="latin-1",
        dtype={
            "y1_statefips": int,
            "y1_countyfips": int,
            "y2_statefips": int,
            "y2_countyfips": int,
            "n1": int,
            "n2": int,
            "agi": int,
        },
    )

    # Filter to ND origins only
    nd_outflows = df[df["y1_statefips"] == ND_STATE_FIPS].copy()

    results = []

    for county_fips in ND_COUNTIES:
        county_data = nd_outflows[nd_outflows["y1_countyfips"] == county_fips]

        # Total outflow (US + Foreign)
        total_row = county_data[
            (county_data["y2_statefips"] == 96) & (county_data["y2_countyfips"] == 0)
        ]

        # US only outflow (domestic)
        us_row = county_data[
            (county_data["y2_statefips"] == 97) & (county_data["y2_countyfips"] == 0)
        ]

        total_n2 = total_row["n2"].values[0] if len(total_row) > 0 else 0
        us_n2 = us_row["n2"].values[0] if len(us_row) > 0 else 0

        # Handle suppressed values (-1)
        if total_n2 == -1:
            total_n2 = 0
        if us_n2 == -1:
            us_n2 = 0

        results.append(
            {
                "county_fips_3": county_fips,
                "year": year,
                "outflow_n2": total_n2,
                "outflow_domestic": us_n2,
            }
        )

    return pd.DataFrame(results)


def main():
    """Main processing function."""
    print("=" * 60)
    print("Processing IRS Migration Data for North Dakota")
    print("=" * 60)

    all_inflows = []
    all_outflows = []

    # Process each year
    for suffix, year in FILE_YEARS.items():
        inflow_file = MIGRATION_DIR / f"countyinflow{suffix}.csv"
        outflow_file = MIGRATION_DIR / f"countyoutflow{suffix}.csv"

        if inflow_file.exists() and outflow_file.exists():
            print(f"\nProcessing year {year} (files: {suffix})...")

            inflow_df = process_inflows(inflow_file, year)
            outflow_df = process_outflows(outflow_file, year)

            all_inflows.append(inflow_df)
            all_outflows.append(outflow_df)

            print(f"  - Processed {len(inflow_df)} county inflow records")
            print(f"  - Processed {len(outflow_df)} county outflow records")
        else:
            print(f"\nWarning: Files for {year} not found, skipping...")

    # Combine all years
    inflows_combined = pd.concat(all_inflows, ignore_index=True)
    outflows_combined = pd.concat(all_outflows, ignore_index=True)

    # Merge inflows and outflows
    merged = pd.merge(
        inflows_combined, outflows_combined, on=["county_fips_3", "year"], how="outer"
    )

    # Add county names
    merged["county_name"] = merged["county_fips_3"].map(ND_COUNTIES)

    # Calculate net migration
    merged["net_migration"] = merged["inflow_n2"] - merged["outflow_n2"]

    # Select and order final columns
    final_df = (
        merged[
            [
                "county_fips",
                "county_name",
                "year",
                "inflow_n2",
                "outflow_n2",
                "net_migration",
                "inflow_domestic",
                "outflow_domestic",
            ]
        ]
        .sort_values(["year", "county_fips"])
        .reset_index(drop=True)
    )

    # Save output
    output_path = MIGRATION_DIR / "nd_migration_processed.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Output saved to: {output_path}")
    print(f"Total records: {len(final_df)}")
    print(f"Years covered: {sorted(final_df['year'].unique())}")
    print(f"Counties covered: {final_df['county_fips'].nunique()}")

    # Generate summary statistics
    print(f"\n{'=' * 60}")
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # State-level totals by year
    print("\n1. TOTAL ND MIGRATION BY YEAR")
    print("-" * 40)
    yearly_totals = (
        final_df.groupby("year")
        .agg({"inflow_n2": "sum", "outflow_n2": "sum", "net_migration": "sum"})
        .reset_index()
    )
    yearly_totals.columns = pd.Index(["Year", "Total Inflows", "Total Outflows", "Net Migration"])
    print(yearly_totals.to_string(index=False))

    # Top 5 counties by positive net migration (all years combined)
    print("\n2. TOP 5 COUNTIES BY NET MIGRATION (POSITIVE - ALL YEARS)")
    print("-" * 40)
    county_totals = (
        final_df.groupby(["county_fips", "county_name"])
        .agg({"net_migration": "sum", "inflow_n2": "sum", "outflow_n2": "sum"})
        .reset_index()
    )

    top_positive = county_totals.nlargest(5, "net_migration")
    for _, row in top_positive.iterrows():
        print(
            f"  {row['county_name']:20s} ({row['county_fips']}): "
            f"+{row['net_migration']:,} (in: {row['inflow_n2']:,}, out: {row['outflow_n2']:,})"
        )

    # Top 5 counties by negative net migration (all years combined)
    print("\n3. TOP 5 COUNTIES BY NET MIGRATION (NEGATIVE - ALL YEARS)")
    print("-" * 40)
    top_negative = county_totals.nsmallest(5, "net_migration")
    for _, row in top_negative.iterrows():
        print(
            f"  {row['county_name']:20s} ({row['county_fips']}): "
            f"{row['net_migration']:,} (in: {row['inflow_n2']:,}, out: {row['outflow_n2']:,})"
        )

    # State-level trend
    print("\n4. STATE-LEVEL NET MIGRATION TREND")
    print("-" * 40)
    for _, row in yearly_totals.iterrows():
        bar_len = int(abs(row["Net Migration"]) / 200)  # Scale for display
        bar_char = "+" if row["Net Migration"] >= 0 else "-"
        bar = bar_char * min(bar_len, 50)
        sign = "+" if row["Net Migration"] >= 0 else ""
        print(f"  {int(row['Year'])}: {sign}{row['Net Migration']:,} {bar}")

    # Data quality summary
    print("\n5. DATA QUALITY NOTES")
    print("-" * 40)
    zero_inflows = (final_df["inflow_n2"] == 0).sum()
    zero_outflows = (final_df["outflow_n2"] == 0).sum()
    print(f"  - Records with zero inflows: {zero_inflows}")
    print(f"  - Records with zero outflows: {zero_outflows}")
    print("  - (Zero values may indicate suppressed data)")

    return final_df


if __name__ == "__main__":
    df = main()
