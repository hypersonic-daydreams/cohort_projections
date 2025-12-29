#!/usr/bin/env python3
# mypy: ignore-errors
"""
Download Census Bureau ACS Table B05006: Place of Birth for Foreign-Born Population
Downloads data for all states by year using the Census API group() function.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR
YEARS = list(range(2009, 2024))  # 2009-2023 (5-year ACS started in 2009)
API_BASE = "https://api.census.gov/data/{year}/acs/acs5"


def fetch_variable_labels(year: int) -> dict:
    """Fetch variable labels for the B05006 table."""
    url = f"{API_BASE.format(year=year)}/groups/B05006.json"

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Extract variable labels
        labels = {}
        for var_id, var_info in data.get("variables", {}).items():
            if var_id.startswith("B05006_"):
                labels[var_id] = var_info.get("label", "")

        return labels

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching variable labels for {year}: {e}")
        return {}


def fetch_year_data(year: int) -> pd.DataFrame | None:
    """Fetch B05006 data for a specific year using group()."""

    url = f"{API_BASE.format(year=year)}"
    params = {"get": "group(B05006)", "for": "state:*"}

    print("  Fetching all B05006 data...")

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        # First row is header
        headers = data[0]
        rows = data[1:]

        df = pd.DataFrame(rows, columns=headers)

        # Drop annotation columns (EA, MA suffixes) - they're mostly null
        drop_cols = [c for c in df.columns if c.endswith(("EA", "MA"))]
        df = df.drop(columns=drop_cols, errors="ignore")

        print(f"  Got {len(df)} rows, {len(df.columns)} columns")
        return df

    except requests.exceptions.RequestException as e:
        print(f"  Error fetching year {year}: {e}")
        return None


def main():
    """Download B05006 data for multiple years."""

    all_years_data = []
    all_labels = {}

    for year in YEARS:
        print(f"\nFetching data for {year}...")

        df = fetch_year_data(year)

        if df is not None:
            df["year"] = year
            all_years_data.append(df)

            # Save individual year file
            year_file = OUTPUT_DIR / f"b05006_states_{year}.csv"
            df.to_csv(year_file, index=False)
            print(f"  Saved {year_file}")

            # Get variable labels (just once)
            if not all_labels:
                all_labels = fetch_variable_labels(year)

        time.sleep(1)  # Be nice to the API between years

    if all_years_data:
        # Combine all years
        combined_df = pd.concat(all_years_data, ignore_index=True)
        combined_file = OUTPUT_DIR / "b05006_states_all_years.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\nSaved combined file: {combined_file}")
        print(f"Total rows: {len(combined_df)}")

        # Save variable labels
        if all_labels:
            labels_file = OUTPUT_DIR / "b05006_variable_labels.json"
            with open(labels_file, "w") as f:
                json.dump(all_labels, f, indent=2)
            print(f"Saved variable labels: {labels_file}")

    return all_years_data


if __name__ == "__main__":
    main()
