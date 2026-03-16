#!/usr/bin/env python3
"""
Ingest Census Bureau stcoreview (State/County Review) Vintage 2025 data.

Created: 2026-02-18
ADR: 035 (Census PEP Components of Change), 055 (Group Quarters Separation)
Author: nhaarstad

Purpose
-------
Parse the pre-release stcoreview Excel file from the Census Bureau's Population
Estimates Program into a tidy long-format parquet for downstream pipeline use.
The stcoreview file is the authoritative source for county-level population
estimates, components of change (births, deaths, domestic/international migration),
and household/group-quarters population splits for North Dakota Vintage 2025
(2020-2025). Without this ingestion step, the project cannot compute residual
migration rates (ADR-035) or separate group quarters from household population
(ADR-055).

Method
------
1. Read the "in" sheet from the stcoreview Excel file, preserving State and
   County columns as strings for FIPS code construction.
2. Build 5-digit FIPS codes (state + county) and flag the state-total row
   (county code 000).
3. Parse each data column name using a regex pattern that extracts the variable
   name (e.g., Respop, GQpop, Births), optional age group suffix (0017, 1864,
   65up), and period (census, base, or 4-digit year).
4. Unpivot all parsed data columns into long format with one row per
   geoid x variable x age_group x period combination.
5. Convert '.' sentinel values (Census missing indicator) to None.
6. Add an integer year column derived from the period field where possible.
7. Write the result to a gzip-compressed parquet file.

Key design decisions
--------------------
- **Long format over wide**: The source file has ~300+ columns in wide format.
  Converting to long format (geoid, variable, age_group, period, value) makes
  downstream filtering and joining straightforward without requiring column name
  parsing at every use site.
- **All variables preserved**: Rather than extracting only Respop or GQpop, all
  variables (births, deaths, migration, residual, etc.) are retained so a single
  parquet file serves multiple downstream consumers (migration pipeline, GQ
  separation, validation scripts).
- **String FIPS codes**: State and county codes are stored as zero-padded strings
  (e.g., "38", "015") to match the project-wide FIPS convention and prevent
  integer truncation of leading zeros.

Validation results (2026-02-18)
-------------------------------
- 54 geographic units parsed (53 counties + 1 state total)
- 12 variables extracted: Respop, HHpop, GQpop, Births, Deaths, Dommig,
  Dommigrate, Intlmig, Residual, Popturning18, Popturning65, Natrake
- 4 age groups: total, 0-17, 18-64, 65+
- 8 periods: census, base, 2020-2025

Inputs
------
- data/raw/population/stcoreview_v2025_ND.xlsx
    Census Bureau PEP stcoreview pre-release file for North Dakota,
    Vintage 2025. Contains "in" sheet with wide-format county rows and
    ~300+ columns. Received 2026-02 from Census pre-release distribution.

Output
------
- data/raw/population/stcoreview_v2025_nd_parsed.parquet
    Long-format parquet, gzip compressed. Columns: geoid, state_fips,
    county_fips, county_name, is_state_total, variable, age_group, period,
    value, year.

Usage
-----
    python scripts/data/ingest_stcoreview.py
    python scripts/data/ingest_stcoreview.py --input /path/to/file.xlsx
    python scripts/data/ingest_stcoreview.py --output data/raw/population/custom_output.parquet
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "population" / "stcoreview_v2025_ND.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "population" / "stcoreview_v2025_nd_parsed.parquet"

# North Dakota state FIPS
ND_STATE_FIPS = "38"

# Sheet name in the Excel file
SHEET_NAME = "in"

# Age group label mapping (source suffix -> readable label)
AGE_GROUP_MAP = {
    "0017": "0-17",
    "1864": "18-64",
    "65up": "65+",
}

# Column name pattern: {Variable}[_{AgeGroup}]_{YearOrPeriod}
# Examples: Respop_2025, Respop_0017_2025, Dommigrate_65up_2020, Respop_census
COL_PATTERN = re.compile(
    r"^(?P<variable>[A-Za-z]+\d*)"
    r"(?:_(?P<age_group>0017|1864|65up))?"
    r"_(?P<period>census|base|\d{4})$"
)


def parse_column_name(col: str) -> dict | None:
    """Parse a stcoreview column name into its components.

    Returns a dict with keys: variable, age_group (or None), period.
    Returns None if the column doesn't match the expected pattern.
    """
    m = COL_PATTERN.match(col)
    if not m:
        return None
    return {
        "variable": m.group("variable"),
        "age_group": AGE_GROUP_MAP.get(m.group("age_group")),
        "period": m.group("period"),
    }


def build_fips(state: int | str, county: int | str) -> str:
    """Build a 5-digit FIPS code from state and county codes."""
    return f"{int(state):02d}{int(county):03d}"


def ingest_stcoreview(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read and parse the stcoreview Excel file.

    Args:
        input_path: Path to the stcoreview xlsx file.
        output_path: Path to write the parsed parquet file.

    Returns:
        The parsed DataFrame in long format.
    """
    print(f"Reading: {input_path}")
    raw = pd.read_excel(input_path, sheet_name=SHEET_NAME, dtype={"State": str, "County": str})
    print(f"  Shape: {raw.shape} ({raw.shape[0]} rows x {raw.shape[1]} columns)")

    # Build 5-digit FIPS codes
    raw["geoid"] = raw.apply(lambda r: build_fips(r["State"], r["County"]), axis=1)
    raw["state_fips"] = raw["State"].apply(lambda s: f"{int(s):02d}")
    raw["county_fips"] = raw["County"].apply(lambda c: f"{int(c):03d}")

    # Identify the state row (county == 0 means state total)
    raw["is_state_total"] = raw["County"].astype(int) == 0

    # Parse data columns into long format
    data_cols = [c for c in raw.columns if c not in ["State", "County", "Name",
                                                       "geoid", "state_fips",
                                                       "county_fips", "is_state_total"]]

    records = []
    for col in data_cols:
        parsed = parse_column_name(col)
        if parsed is None:
            print(f"  WARNING: Could not parse column '{col}', skipping.")
            continue

        variable = parsed["variable"]
        age_group = parsed["age_group"]  # None for total
        period = parsed["period"]

        for _, row in raw.iterrows():
            val = row[col]
            # Handle '.' missing values
            if isinstance(val, str) and val.strip() == ".":
                val = None
            else:
                try:
                    val = float(val) if val is not None else None
                except (ValueError, TypeError):
                    val = None

            records.append({
                "geoid": row["geoid"],
                "state_fips": row["state_fips"],
                "county_fips": row["county_fips"],
                "county_name": row["Name"],
                "is_state_total": row["is_state_total"],
                "variable": variable,
                "age_group": age_group if age_group else "total",
                "period": period,
                "value": val,
            })

    df = pd.DataFrame(records)

    # Convert period to integer year where possible
    df["year"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")

    print(f"  Parsed {len(df)} records")
    print(f"  Variables: {sorted(df['variable'].unique())}")
    print(f"  Age groups: {sorted(df['age_group'].unique())}")
    print(f"  Periods: {sorted(df['period'].unique())}")
    print(f"  Counties: {df[~df['is_state_total']]['geoid'].nunique()}")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="gzip")
    print(f"\nSaved parsed data to: {output_path}")
    print(f"  Records: {len(df)}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Census Bureau stcoreview Vintage 2025 data for North Dakota.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to stcoreview xlsx file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to output parquet file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ingest_stcoreview(args.input, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
