#!/usr/bin/env python3
"""
Ingest Census Bureau stcoreview (State/County Review) Vintage 2025 data.

Parses the Excel file from the Census Bureau's Population Estimates Program
and outputs a tidy parquet file for downstream use. The stcoreview file contains
county-level population estimates and components of change for North Dakota,
Vintage 2025 (covering 2020-2025).

Column naming convention in the source file:
    {Variable}_{Year}          -- total, e.g., Respop_2025, Births_2024
    {Variable}_{AgeGroup}_{Year} -- by age group, e.g., Respop_0017_2023
    {Variable}_census          -- Census 2020 count (often '.' for missing)
    {Variable}_base            -- Census 2020 base population

Variables include: Respop, HHpop, GQpop, Births, Deaths, Dommig, Dommigrate,
Intlmig, Residual, Popturning18, Popturning65, Natrake

Age groups: 0017 (0-17), 1864 (18-64), 65up (65+)

Usage:
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
