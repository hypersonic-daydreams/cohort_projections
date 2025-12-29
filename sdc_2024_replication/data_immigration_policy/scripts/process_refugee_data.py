# mypy: ignore-errors
"""
Process WRAPS Refugee Arrivals by State and Nationality data.

This script reads the downloaded Excel files from the Refugee Processing Center
and aggregates them to create a state-year-nationality level dataset.

Data sources:
- FY 2002-2011: ORR/PRM Academic Dataset (Dreher et al. 2020) - Stata file with city-level data
  Downloaded from refugeeresettlementdata.com
- FY 2012-2020: Excel files (.xls/.xlsx) with state-nationality breakdowns from RPC/WRAPS
- FY 2021-2024: PDF files (not directly parseable - need manual extraction or alternative source)

Note: The academic dataset (FY 2002-2011) may have slightly different totals than official WRAPS
data due to different data collection methodologies and inclusion of Amerasians/SIVs.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="xlrd")

# State FIPS to state name mapping
STATE_FIPS_TO_NAME = {
    "01": "Alabama",
    "02": "Alaska",
    "04": "Arizona",
    "05": "Arkansas",
    "06": "California",
    "08": "Colorado",
    "09": "Connecticut",
    "10": "Delaware",
    "11": "District of Columbia",
    "12": "Florida",
    "13": "Georgia",
    "15": "Hawaii",
    "16": "Idaho",
    "17": "Illinois",
    "18": "Indiana",
    "19": "Iowa",
    "20": "Kansas",
    "21": "Kentucky",
    "22": "Louisiana",
    "23": "Maine",
    "24": "Maryland",
    "25": "Massachusetts",
    "26": "Michigan",
    "27": "Minnesota",
    "28": "Mississippi",
    "29": "Missouri",
    "30": "Montana",
    "31": "Nebraska",
    "32": "Nevada",
    "33": "New Hampshire",
    "34": "New Jersey",
    "35": "New Mexico",
    "36": "New York",
    "37": "North Carolina",
    "38": "North Dakota",
    "39": "Ohio",
    "40": "Oklahoma",
    "41": "Oregon",
    "42": "Pennsylvania",
    "44": "Rhode Island",
    "45": "South Carolina",
    "46": "South Dakota",
    "47": "Tennessee",
    "48": "Texas",
    "49": "Utah",
    "50": "Vermont",
    "51": "Virginia",
    "53": "Washington",
    "54": "West Virginia",
    "55": "Wisconsin",
    "56": "Wyoming",
    "60": "American Samoa",
    "66": "Guam",
    "69": "Northern Mariana Islands",
    "72": "Puerto Rico",
    "78": "Virgin Islands",
}

# Nationality name standardization mapping (applied to lowercase versions)
NATIONALITY_STANDARDIZATION = {
    "ussr": "USSR",
    "dem. rep. congo": "Democratic Republic of the Congo",
    "dem. rep. of the congo": "Democratic Republic of the Congo",
    "drc": "Democratic Republic of the Congo",
    "zaire": "Democratic Republic of the Congo",
    "unknown": "Unknown",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
}


def is_numeric(val) -> bool:
    """Check if value is numeric (can be converted to int)."""
    if pd.isna(val):
        return False
    try:
        int(val)
        return True
    except (ValueError, TypeError):
        return False


def clean_nationality_name(nat: str) -> str:
    """Standardize nationality names."""
    if pd.isna(nat):
        return nat
    nat_lower = nat.lower().strip()
    if nat_lower in NATIONALITY_STANDARDIZATION:
        return NATIONALITY_STANDARDIZATION[nat_lower]
    # Title case for other nationalities
    return nat.title()


def parse_state_nationality_file(filepath: Path, fiscal_year: int) -> pd.DataFrame:
    """
    Parse a WRAPS arrivals by state and nationality Excel file.

    The files have a hierarchical structure:
    - State name in column 0 with total in column 2/3
    - Nationality breakdowns in subsequent rows with state=NaN

    Args:
        filepath: Path to Excel file
        fiscal_year: Fiscal year for this data

    Returns:
        DataFrame with columns: state, nationality, fiscal_year, arrivals
    """
    # Read file without headers
    if filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath, header=None)
    else:  # .xls
        df = pd.read_excel(filepath, header=None)

    # Find the data start - look for first state "Alabama"
    data_start = None
    for i in range(len(df)):
        if df.iloc[i, 0] == "Alabama":
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Could not find data start in {filepath}")

    # Determine which column has numeric data (FY data vs cumulative)
    # Column 2 usually has FY data, column 3 has cumulative
    # They should have the same values for a single-year file
    fy_col = 2  # Default

    # Parse the data
    records = []
    current_state = None

    for i in range(data_start, len(df)):
        row = df.iloc[i]

        # Check if this is a state row (has state name in column 0)
        if pd.notna(row[0]) and row[0] not in ["Grand Total", "Total"]:
            current_state = str(row[0]).strip()
            # State row has the total for all nationalities
            # Use column 2 if numeric, otherwise column 3
            total = row[fy_col] if is_numeric(row[fy_col]) else row[3]
            if is_numeric(total):
                records.append(
                    {
                        "state": current_state,
                        "nationality": "Total",
                        "fiscal_year": fiscal_year,
                        "arrivals": int(total),
                    }
                )
        elif pd.isna(row[0]) and current_state is not None:
            # Nationality breakdown row
            nationality = row[1]
            if pd.notna(nationality) and str(nationality).strip() not in ["Nationality", ""]:
                arrivals = row[fy_col] if is_numeric(row[fy_col]) else row[3]
                if is_numeric(arrivals):
                    records.append(
                        {
                            "state": current_state,
                            "nationality": clean_nationality_name(str(nationality).strip()),
                            "fiscal_year": fiscal_year,
                            "arrivals": int(arrivals),
                        }
                    )
        elif row[0] in ["Grand Total", "Total"]:
            # End of state data
            break

    return pd.DataFrame(records)


def process_academic_dataset(
    source_dir: Path, start_year: int = 2002, end_year: int = 2011
) -> pd.DataFrame:
    """
    Process the ORR/PRM academic dataset (Dreher et al. 2020) for historical data.

    This dataset covers 1975-2018 with city-level geocoded refugee resettlement data.
    We aggregate it to state-nationality-year level to match the WRAPS format.

    Args:
        source_dir: Directory containing the orr_prm_1975_2018_v1.dta file
        start_year: First fiscal year to include (default 2002)
        end_year: Last fiscal year to include (default 2011)

    Returns:
        DataFrame with columns: state, nationality, fiscal_year, arrivals
    """
    dta_file = source_dir / "orr_prm_1975_2018_v1.dta"
    if not dta_file.exists():
        print(f"  Warning: Academic dataset not found at {dta_file}")
        return pd.DataFrame()

    print(f"Processing academic dataset for FY {start_year}-{end_year}...")

    # Read the Stata file
    df = pd.read_stata(dta_file)

    # Filter to desired years
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # Map state FIPS to state names
    df["state"] = df["state_fips"].map(STATE_FIPS_TO_NAME)

    # Filter out unmapped records (some have '.' as FIPS)
    df = df[df["state"].notna()]

    # Aggregate to state-nationality-year level
    state_nat_year = (
        df.groupby(["year", "state", "citizenship_stable"])["refugees"].sum().reset_index()
    )
    state_nat_year.columns = ["fiscal_year", "state", "nationality", "arrivals"]

    # Also create state totals
    state_year = df.groupby(["year", "state"])["refugees"].sum().reset_index()
    state_year.columns = ["fiscal_year", "state", "arrivals"]
    state_year["nationality"] = "Total"

    # Combine nationality detail with totals
    combined = pd.concat(
        [state_nat_year, state_year[["fiscal_year", "state", "nationality", "arrivals"]]],
        ignore_index=True,
    )

    # Clean nationality names
    combined["nationality"] = combined["nationality"].apply(clean_nationality_name)

    # Convert arrivals to int
    combined["arrivals"] = combined["arrivals"].fillna(0).astype(int)

    print(f"  Found {len(combined)} records from academic dataset")
    print(f"  Years: {sorted(combined['fiscal_year'].unique())}")
    print(f"  States: {combined['state'].nunique()}")

    return combined


def process_all_wraps_files(source_dir: Path) -> pd.DataFrame:
    """
    Process all WRAPS Excel files in the source directory.

    Args:
        source_dir: Directory containing downloaded WRAPS files

    Returns:
        Combined DataFrame with all fiscal years
    """
    all_data = []

    # Process FY 2012-2020 Excel files
    for year in range(2012, 2021):
        xlsx_file = source_dir / f"FY_{year}_Arrivals_by_State_and_Nationality.xlsx"
        xls_file = source_dir / f"FY_{year}_Arrivals_by_State_and_Nationality.xls"

        filepath = xlsx_file if xlsx_file.exists() else xls_file

        if filepath.exists():
            print(f"Processing {filepath.name}...")
            try:
                df = parse_state_nationality_file(filepath, year)
                all_data.append(df)
                print(f"  Found {len(df)} records, {df['state'].nunique()} states")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Warning: No file found for FY {year}")

    if not all_data:
        raise ValueError("No data files were successfully processed")

    # Combine all years
    combined = pd.concat(all_data, ignore_index=True)

    return combined


def calculate_nd_share(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate North Dakota's share of total refugee arrivals by nationality.

    Args:
        df: DataFrame with state, nationality, fiscal_year, arrivals

    Returns:
        DataFrame with nd_share column added
    """
    # Filter to non-Total rows only
    detail = df[df["nationality"] != "Total"].copy()

    # Calculate national totals by year and nationality
    national_totals = (
        detail.groupby(["fiscal_year", "nationality"])["arrivals"]
        .sum()
        .reset_index()
        .rename(columns={"arrivals": "national_total"})
    )

    # Get North Dakota data
    nd_data = detail[detail["state"] == "North Dakota"].copy()

    # Merge and calculate share
    nd_with_share = nd_data.merge(national_totals, on=["fiscal_year", "nationality"], how="left")
    nd_with_share["nd_share"] = nd_with_share["arrivals"] / nd_with_share["national_total"]

    # For the main dataset, add national totals
    result = df.merge(national_totals, on=["fiscal_year", "nationality"], how="left")

    # Calculate share for all states
    result["state_share_of_nationality"] = np.where(
        result["national_total"] > 0, result["arrivals"] / result["national_total"], 0
    )

    return result


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase with underscores."""
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    return df


def main():
    """Main processing function."""
    # Set up paths - Use project-level data directories
    project_root = Path(__file__).parent.parent.parent.parent  # cohort_projections/

    # Input: raw refugee data
    source_dir = project_root / "data" / "raw" / "immigration" / "refugee_data"

    # Output: analysis goes to project-level processed directory
    output_dir = project_root / "data" / "processed" / "immigration" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Processing Refugee Arrivals Data")
    print("=" * 60)

    all_data = []

    # Process FY 2002-2011 from academic dataset
    print("\n" + "-" * 60)
    print("Part 1: Historical data from ORR/PRM Academic Dataset")
    print("-" * 60)
    academic_df = process_academic_dataset(source_dir, start_year=2002, end_year=2011)
    if not academic_df.empty:
        academic_df["data_source"] = "ORR/PRM Academic Dataset (Dreher et al. 2020)"
        all_data.append(academic_df)

    # Process FY 2012-2020 from WRAPS Excel files
    print("\n" + "-" * 60)
    print("Part 2: WRAPS Excel files (FY 2012-2020)")
    print("-" * 60)
    wraps_df = process_all_wraps_files(source_dir)
    if not wraps_df.empty:
        wraps_df["data_source"] = "WRAPS/RPC Excel files"
        all_data.append(wraps_df)

    if not all_data:
        raise ValueError("No data files were successfully processed")

    # Combine all data sources
    df = pd.concat(all_data, ignore_index=True)

    print(f"\nCombined data: {len(df)} records")
    print(f"States: {df['state'].nunique()}")
    print(f"Fiscal years: {sorted(df['fiscal_year'].unique())}")
    print(f"Nationalities: {df['nationality'].nunique()}")

    # Calculate shares
    print("\nCalculating state shares of arrivals by nationality...")
    df = calculate_nd_share(df)

    # Clean column names
    df = clean_column_names(df)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    # Total arrivals by year
    yearly_totals = df[df["nationality"] == "Total"].groupby("fiscal_year")["arrivals"].sum()
    print("\nTotal arrivals by fiscal year:")
    for year, total in yearly_totals.items():
        print(f"  FY {year}: {total:,}")

    # North Dakota totals
    nd_totals = (
        df[(df["state"] == "North Dakota") & (df["nationality"] == "Total")]
        .groupby("fiscal_year")["arrivals"]
        .sum()
    )
    print("\nNorth Dakota arrivals by fiscal year:")
    for year, total in nd_totals.items():
        nd_share = total / yearly_totals[year] * 100 if year in yearly_totals else 0
        print(f"  FY {year}: {total:,} ({nd_share:.2f}% of national)")

    # Top nationalities to North Dakota
    min_year = df["fiscal_year"].min()
    max_year = df["fiscal_year"].max()
    nd_by_nationality = (
        df[(df["state"] == "North Dakota") & (df["nationality"] != "Total")]
        .groupby("nationality")["arrivals"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"\nTop 10 nationalities resettled in North Dakota (FY {min_year}-{max_year}):")
    for nationality, count in nd_by_nationality.items():
        print(f"  {nationality}: {count:,}")

    # Save to parquet
    output_file = output_dir / "refugee_arrivals_by_state_nationality.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Also save a CSV for inspection
    csv_file = output_dir / "refugee_arrivals_by_state_nationality.csv"
    df.to_csv(csv_file, index=False)
    print(f"Also saved CSV: {csv_file}")

    return df


if __name__ == "__main__":
    main()
