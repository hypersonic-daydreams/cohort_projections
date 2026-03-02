#!/usr/bin/env python3
"""
Fetch ACS housing unit and household size data for North Dakota places.

Created: 2026-03-01
ADR: 060 (Housing-Unit Method for Place Projections)
Author: Claude Code / N. Haarstad

Purpose
-------
Download American Community Survey (ACS) 5-year estimate data for housing
units (Table B25001) and average household size (Table B25010) at the place
level for North Dakota.  These data feed the housing-unit method (HU method),
a complementary short-term cross-check for the share-trending place
projections.

Method
------
1. Query the Census ACS 5-year API for each available vintage (2009 through
   2023, representing the 2005-2009 through 2019-2023 estimate periods).
2. Fetch variables B25001_001E (total housing units) and B25010_001E
   (average household size) for all places in North Dakota (state FIPS 38).
3. Combine vintages into a single long-format CSV and save to
   ``data/raw/housing/nd_place_housing_units.csv``.

If the Census API is unreachable the script will log errors per vintage
and write whatever data it successfully retrieved.

Inputs
------
- Census ACS 5-year API (online)

Outputs
-------
- data/raw/housing/nd_place_housing_units.csv
    Columns: place_fips, place_name, year, housing_units, avg_hh_size
    One row per place per ACS vintage year.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests  # type: ignore[import-untyped]

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

logger = setup_logger(__name__, log_level="INFO")

STATE_FIPS = "38"
ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"

# ACS 5-year vintages to fetch.  The ``year`` key is the end-year of the
# 5-year period (e.g. 2023 = 2019-2023 estimates).
ACS_VINTAGES = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Variables of interest
VARIABLES = {
    "B25001_001E": "housing_units",       # Total housing units
    "B25010_001E": "avg_hh_size",         # Average household size
}

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def fetch_acs_housing_vintage(year: int) -> pd.DataFrame | None:
    """
    Fetch housing data for a single ACS 5-year vintage.

    Args:
        year: ACS end-year (e.g. 2023 for 2019-2023 estimates).

    Returns:
        DataFrame with place_fips, place_name, year, housing_units, avg_hh_size
        or None if the API call fails.
    """
    url = ACS_BASE_URL.format(year=year)
    var_list = ["NAME"] + list(VARIABLES.keys())
    params = {
        "get": ",".join(var_list),
        "for": "place:*",
        "in": f"state:{STATE_FIPS}",
    }

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"ACS {year} attempt {attempt + 1}/{MAX_RETRIES}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError) as exc:
            logger.warning(f"ACS {year} attempt {attempt + 1} failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"ACS {year}: all retries exhausted")
                return None

    df = pd.DataFrame(data[1:], columns=data[0])

    # Build output columns
    result = pd.DataFrame({
        "place_fips": df["state"] + df["place"],
        "place_name": df["NAME"],
        "year": year,
    })
    for api_var, col_name in VARIABLES.items():
        values = pd.to_numeric(df[api_var], errors="coerce")
        # Census uses -666666666 (and similar) as "not available" sentinel
        values = values.where(values >= 0)
        result[col_name] = values

    logger.info(f"ACS {year}: fetched {len(result)} places")
    return result


def fetch_all_vintages(vintages: list[int] | None = None) -> pd.DataFrame:
    """
    Fetch housing data across all requested ACS vintages.

    Args:
        vintages: List of ACS end-years.  Defaults to ``ACS_VINTAGES``.

    Returns:
        Combined DataFrame across all vintages that returned data.
    """
    if vintages is None:
        vintages = ACS_VINTAGES

    frames: list[pd.DataFrame] = []
    for year in vintages:
        logger.info(f"Fetching ACS {year} housing data...")
        df = fetch_acs_housing_vintage(year)
        if df is not None:
            frames.append(df)
        # Rate-limit to stay within Census API norms
        time.sleep(1)

    if not frames:
        logger.error("No ACS housing data was retrieved from any vintage")
        return pd.DataFrame(columns=["place_fips", "place_name", "year", "housing_units", "avg_hh_size"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["place_fips", "year"]).reset_index(drop=True)
    logger.info(
        f"Combined housing data: {len(combined)} rows, "
        f"{combined['place_fips'].nunique()} places, "
        f"{combined['year'].nunique()} vintages"
    )
    return combined


def main() -> int:
    """Main entry point: fetch and save housing unit data."""
    logger.info("=" * 70)
    logger.info("Fetching Census ACS Housing Unit Data (ADR-060)")
    logger.info("=" * 70)

    combined = fetch_all_vintages()

    if combined.empty:
        logger.error("No data retrieved; nothing to write.")
        return 1

    output_dir = project_root / "data" / "raw" / "housing"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nd_place_housing_units.csv"
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved housing data to {output_path}")

    # Summary
    logger.info(f"Places: {combined['place_fips'].nunique()}")
    logger.info(f"Vintages: {sorted(combined['year'].unique())}")
    for year in sorted(combined["year"].unique()):
        year_data = combined[combined["year"] == year]
        logger.info(
            f"  {year}: {len(year_data)} places, "
            f"total HU={year_data['housing_units'].sum():,.0f}, "
            f"mean PPH={year_data['avg_hh_size'].mean():.2f}"
        )

    logger.info("Housing data fetch complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
