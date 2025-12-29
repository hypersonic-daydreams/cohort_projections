# mypy: ignore-errors
"""
Process Historical DHS Naturalization Data (FY 2000-2023)

This script processes naturalization data from the DHS Yearbook of Immigration Statistics.
The data structure varies across years:
- FY 2000-2001: Table 52 contains naturalizations by state
- FY 2002-2004: Table 34 contains naturalizations by state
- FY 2005-2023: Table 22 contains naturalizations by state

Supplemental Table 1 contains detailed state x country of birth data.

Output files:
- dhs_naturalizations_by_state_historical.parquet - All state-level data FY 2000-2023
- dhs_naturalizations_by_state_country_historical.parquet - Detailed state x country data
"""

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# Paths - Use project-level data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/

# Input: raw DHS historical data
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "immigration" / "dhs_refugees_naturalization"
HISTORICAL_DIR = SOURCE_DIR / "historical_downloads"

# Output: analysis goes to project-level processed directory
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"


def clean_column_name(name) -> str:
    """Convert column name to lowercase with underscores."""
    if pd.isna(name):
        return "unknown"
    name = str(name).strip().lower()
    name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
    name = name.replace("__", "_")
    return name


def find_table22_file(year_dir: Path, year: int) -> tuple:
    """
    Find the Table 22 (or equivalent) file for naturalizations by state.
    Returns (file_path, table_number, header_row).
    """
    # Different table numbers across years - these are the STATE-LEVEL MULTI-YEAR tables
    table_mapping = {
        # FY 2000: Table 51 is naturalizations by state (multi-year)
        2000: ("Table51.xls", 51, 3),
        # FY 2001: Table 50 is naturalizations by state (multi-year)
        2001: ("Table50.xls", 50, 3),
        # FY 2002: Table 36 is naturalizations by state (multi-year)
        2002: ("Table36.xls", 36, 3),
        # FY 2003: Table 33D is naturalizations by state (multi-year)
        2003: ("Table33D.xls", 33, 3),
        # FY 2004: Table 33 is naturalizations by state (multi-year)
        2004: ("Table33.xls", 33, 3),
        # FY 2005+: Table 22 is naturalizations by state
    }

    if year in table_mapping:
        filename, table_num, header = table_mapping[year]
        file_path = year_dir / filename
        if file_path.exists():
            return file_path, table_num, header

    # For FY 2005+, look for Table 22 with various naming patterns
    patterns = [
        f"fy{year}_table22.xlsx",
        f"fy{year}_table22.xls",
        "table22.xlsx",
        "table22.xls",
        "Table22.xls",
        "Table22.xlsx",
        "Table22D.xls",
        "Table 22.xls",
        f"{year}_table22.xls",
    ]

    # Check for subdirectory (some years have nested folders)
    subdirs = list(year_dir.glob("YRBK*")) + list(year_dir.glob("Zipped*"))
    search_dirs = [year_dir] + subdirs

    for search_dir in search_dirs:
        for pattern in patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                return matches[0], 22, 3  # Header row 3 for these files

    return None, None, None


def find_supp_table1_file(year_dir: Path, year: int) -> tuple:
    """
    Find the Supplemental Table 1 file for naturalizations by state and country.
    Returns (file_path, header_row).
    """
    patterns = [
        f"fy{year}_natzsuptable1d.xlsx",
        f"fy{year}_natzsuptable1d.xls",
        f"{year}_natzsuptable1d.xls",
        "natzsuptable1d.xls",
        "natzsuptable1d.xlsx",
        "NatzSupTable1D.xls",
        "NatzSupTable1D.xlsx",
        f"NatzSupTable1DFY{str(year)[2:]}.xls",
        "NatzSupTable1.xls",
        f"NatzSupTable1fy{str(year)[2:]}D.xls",
        f"NatzSupTable1 fy{str(year)[2:]}D.xls",
        "NatzSupTable01.xls",
    ]

    subdirs = list(year_dir.glob("YRBK*")) + list(year_dir.glob("Zipped*"))
    search_dirs = [year_dir] + subdirs

    for search_dir in search_dirs:
        for pattern in patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                return matches[0], 5

    return None, None


def process_table22(file_path: Path, year: int, header_row: int = 5) -> pd.DataFrame:
    """
    Process a Table 22 (or equivalent) file to extract state-level naturalizations.
    """
    try:
        # Try different engines for old .xls files
        if file_path.suffix == ".xls":
            df = pd.read_excel(file_path, header=header_row, engine="xlrd")
        else:
            df = pd.read_excel(file_path, header=header_row)

        df = df.dropna(how="all")
        df.columns = [clean_column_name(c) for c in df.columns]

        # First column should be state
        cols = list(df.columns)
        cols[0] = "state"
        df.columns = cols

        # Find year columns (numeric columns that look like years)
        year_cols = []
        for c in df.columns:
            col_str = str(c).replace(".0", "").strip()
            # Check if column name looks like a year (1986-2030)
            if col_str.startswith(("20", "19")) or c != "state":
                try:
                    y = int(float(col_str))
                    if 1980 <= y <= 2030:
                        year_cols.append(c)
                except (ValueError, TypeError):
                    pass

        if not year_cols:
            print(f"  Warning: No year columns found in {file_path.name}")
            print(f"  Columns: {list(df.columns)[:10]}")
            return pd.DataFrame()

        # Filter to relevant columns
        df = df[["state"] + year_cols]

        # Clean state names
        df = df[df["state"].notna()]
        df = df[
            ~df["state"].str.contains(
                "Region|Table|Source|Note|total|unknown|nan", case=False, na=True
            )
        ]

        # Melt to long format
        df_long = df.melt(id_vars=["state"], var_name="fiscal_year", value_name="naturalizations")

        # Clean fiscal year
        df_long["fiscal_year"] = df_long["fiscal_year"].astype(str).str.replace(".0", "")
        df_long["fiscal_year"] = pd.to_numeric(df_long["fiscal_year"], errors="coerce")
        df_long = df_long.dropna(subset=["fiscal_year"])
        df_long["fiscal_year"] = df_long["fiscal_year"].astype(int)

        # Clean naturalizations
        df_long["naturalizations"] = pd.to_numeric(df_long["naturalizations"], errors="coerce")

        # Clean state names
        df_long["state"] = df_long["state"].str.strip()
        df_long = df_long[df_long["state"].notna()]

        return df_long

    except Exception as e:
        print(f"  Error processing {file_path}: {e}")
        return pd.DataFrame()


def process_supp_table1(file_path: Path, year: int, header_row: int = 5) -> pd.DataFrame:
    """
    Process Supplemental Table 1 file to extract state x country naturalizations.
    """
    try:
        if file_path.suffix == ".xls":
            df = pd.read_excel(file_path, header=header_row, engine="xlrd")
        else:
            df = pd.read_excel(file_path, header=header_row)

        df = df.dropna(how="all")
        df.columns = [clean_column_name(c) for c in df.columns]

        # First column should be country of birth
        cols = list(df.columns)
        cols[0] = "country_of_birth"
        df.columns = cols

        # Filter out header rows and notes
        df = df[df["country_of_birth"].notna()]
        df = df[
            ~df["country_of_birth"].str.contains("Region|Table|Source|Note", case=False, na=True)
        ]

        df["fiscal_year"] = year

        return df

    except Exception as e:
        print(f"  Error processing supp table {file_path}: {e}")
        return pd.DataFrame()


def process_modern_files(year: int) -> tuple:
    """
    Process FY 2021-2023 files which are in the main directory.
    Returns (state_df, supp_df).
    """
    state_data = pd.DataFrame()
    supp_data = pd.DataFrame()

    # Map years to their file locations
    if year == 2021:
        file_path = (
            SOURCE_DIR / "naturalization_2021" / "Zipped for website" / "fy2021_table22.xlsx"
        )
    elif year == 2022:
        file_path = SOURCE_DIR / "naturalizations_fy2022.xlsx"
    elif year == 2023:
        file_path = SOURCE_DIR / "naturalizations_fy2023.xlsx"
    else:
        return state_data, supp_data

    # Process Table 22
    try:
        if year in [2022, 2023]:
            df = pd.read_excel(file_path, sheet_name="Table 22", header=5)
        else:
            df = pd.read_excel(file_path, header=5)

        df = df.dropna(how="all")
        df.columns = [clean_column_name(c) for c in df.columns]

        cols = list(df.columns)
        cols[0] = "state"
        df.columns = cols

        year_cols = [c for c in df.columns if c.startswith("20")]
        df = df[["state"] + year_cols]

        df_long = df.melt(id_vars=["state"], var_name="fiscal_year", value_name="naturalizations")
        df_long["fiscal_year"] = df_long["fiscal_year"].str.replace(".0", "").astype(int)
        df_long["state"] = df_long["state"].str.strip()
        df_long = df_long[df_long["state"].notna()]

        state_data = df_long
        print(f"  Processed FY{year} Table 22: {len(df_long)} records")

    except Exception as e:
        print(f"  Error processing FY{year} Table 22: {e}")

    # Process Supplemental Table 1 for 2022/2023
    if year in [2022, 2023]:
        try:
            supp_df = pd.read_excel(file_path, sheet_name="NATZSuppTable1", header=5)
            supp_df = supp_df.dropna(how="all")
            supp_df.columns = [clean_column_name(c) for c in supp_df.columns]

            cols = list(supp_df.columns)
            cols[0] = "country_of_birth"
            supp_df.columns = cols

            supp_df = supp_df[supp_df["country_of_birth"].notna()]
            supp_df["fiscal_year"] = year

            supp_data = supp_df
            print(f"  Processed FY{year} Supp Table 1: {len(supp_df)} records")

        except Exception as e:
            print(f"  Error processing FY{year} Supp Table 1: {e}")

    return state_data, supp_data


def main():
    print("=" * 70)
    print("Processing Historical DHS Naturalization Data (FY 2000-2023)")
    print("=" * 70)

    # Create analysis directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_state_data = []
    all_supp_data = []

    # Process historical years (FY 2000-2020)
    print("\n--- Processing Historical Years (FY 2000-2020) ---")

    for year in range(2000, 2021):
        year_dir = HISTORICAL_DIR / f"fy{year}"

        if not year_dir.exists():
            print(f"FY{year}: Directory not found")
            continue

        print(f"FY{year}:")

        # Find and process Table 22 (or equivalent)
        file_path, table_num, header = find_table22_file(year_dir, year)

        if file_path and file_path.exists():
            df = process_table22(file_path, year, header)
            if not df.empty:
                all_state_data.append(df)
                years_found = df["fiscal_year"].unique()
                print(f"  Table {table_num}: {len(df)} records, years: {sorted(years_found)}")
        else:
            print("  Warning: No Table 22 equivalent found")

        # Find and process Supplemental Table 1
        supp_path, supp_header = find_supp_table1_file(year_dir, year)

        if supp_path and supp_path.exists():
            supp_df = process_supp_table1(supp_path, year, supp_header)
            if not supp_df.empty:
                all_supp_data.append(supp_df)
                print(f"  Supp Table 1: {len(supp_df)} records")

    # Process modern years (FY 2021-2023)
    print("\n--- Processing Modern Years (FY 2021-2023) ---")

    for year in [2021, 2022, 2023]:
        print(f"FY{year}:")
        state_df, supp_df = process_modern_files(year)
        if not state_df.empty:
            all_state_data.append(state_df)
        if not supp_df.empty:
            all_supp_data.append(supp_df)

    # Combine all data
    print("\n--- Combining and Saving Results ---")

    if all_state_data:
        combined_state = pd.concat(all_state_data, ignore_index=True)

        # Remove duplicates (keep the most detailed/recent data)
        combined_state = combined_state.drop_duplicates(
            subset=["state", "fiscal_year"], keep="last"
        )

        # Filter to valid states (remove Total, headers, etc.)
        combined_state = combined_state[
            ~combined_state["state"]
            .str.lower()
            .isin(["total", "unknown", "region", "all", "other"])
        ]

        # Filter to valid fiscal years (1986-2030) to remove false positives
        combined_state = combined_state[
            (combined_state["fiscal_year"] >= 1986) & (combined_state["fiscal_year"] <= 2030)
        ]

        # Sort
        combined_state = combined_state.sort_values(["state", "fiscal_year"])

        # Save
        state_output = ANALYSIS_DIR / "dhs_naturalizations_by_state_historical.parquet"
        combined_state.to_parquet(state_output, index=False)

        print(f"\nState-level data saved to: {state_output}")
        print(f"  Total records: {len(combined_state):,}")
        print(f"  States: {combined_state['state'].nunique()}")
        print(f"  Fiscal years: {sorted(combined_state['fiscal_year'].unique())}")

        # Show sample for North Dakota
        nd_data = combined_state[combined_state["state"] == "North Dakota"].sort_values(
            "fiscal_year"
        )
        if not nd_data.empty:
            print("\n  North Dakota naturalizations:")
            for _, row in nd_data.iterrows():
                nat = row["naturalizations"]
                if pd.notna(nat):
                    print(f"    FY{row['fiscal_year']}: {int(nat):,}")

    if all_supp_data:
        combined_supp = pd.concat(all_supp_data, ignore_index=True)

        # Remove duplicates
        combined_supp = combined_supp.drop_duplicates()

        # Convert all columns except country_of_birth and fiscal_year to numeric
        for col in combined_supp.columns:
            if col not in ["country_of_birth", "fiscal_year"]:
                combined_supp[col] = pd.to_numeric(combined_supp[col], errors="coerce")

        # Save
        supp_output = ANALYSIS_DIR / "dhs_naturalizations_by_state_country_historical.parquet"
        combined_supp.to_parquet(supp_output, index=False)

        print(f"\nState x Country data saved to: {supp_output}")
        print(f"  Total records: {len(combined_supp):,}")
        print(f"  Fiscal years: {sorted(combined_supp['fiscal_year'].unique())}")

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
