#!/usr/bin/env python3
"""
Example: Fetching Census Data for North Dakota Cohort Projections

This script demonstrates how to use the CensusDataFetcher class to retrieve
demographic data from the Census Bureau's Population Estimates Program (PEP)
and American Community Survey (ACS) APIs.

Usage:
    python examples/fetch_census_data.py

Requirements:
    - Internet connection for API access
    - Census API key (optional, set in environment variable CENSUS_API_KEY)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.fetch.census_api import CensusDataFetcher  # noqa: E402


def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)

    # Initialize fetcher with default cache directory
    fetcher = CensusDataFetcher()

    # Fetch state-level PEP data
    print("\nFetching state-level PEP data...")
    state_df = fetcher.fetch_pep_state_data(vintage=2024)
    print(f"Retrieved {len(state_df)} records")
    print(f"Columns: {state_df.columns.tolist()}")

    # Show sample data
    print("\nSample data (first 5 rows):")
    print(state_df.head())


def example_county_data():
    """Example 2: Fetching county-level data."""
    print("\n" + "=" * 70)
    print("Example 2: County-Level Data")
    print("=" * 70)

    fetcher = CensusDataFetcher()

    # Fetch county-level PEP data
    print("\nFetching county-level PEP data for all ND counties...")
    county_df = fetcher.fetch_pep_county_data(vintage=2024)
    print(f"Retrieved {len(county_df)} records")
    print(f"Number of counties: {county_df['county'].nunique()}")

    # Show unique counties
    print("\nSample counties:")
    print(county_df["county"].unique()[:10])


def example_acs_places():
    """Example 3: Fetching ACS place data."""
    print("\n" + "=" * 70)
    print("Example 3: ACS Place Data")
    print("=" * 70)

    fetcher = CensusDataFetcher()

    # Fetch ACS place data
    print("\nFetching ACS 5-year estimates for ND places...")
    places_df = fetcher.fetch_acs_place_data(year=2023, dataset="acs5")
    print(f"Retrieved {len(places_df)} places")

    # Show incorporated vs CDPs
    incorporated = places_df[~places_df["is_cdp"]]
    cdps = places_df[places_df["is_cdp"]]

    print(f"\nIncorporated places: {len(incorporated)}")
    print(f"Census-Designated Places: {len(cdps)}")

    # Show largest places
    print("\nTop 10 places by population:")
    top10 = places_df.nlargest(10, "B01001_001E")[["NAME", "B01001_001E", "place_type"]]
    for _idx, row in top10.iterrows():
        print(f"  {row['NAME']}: {row['B01001_001E']:,} ({row['place_type']})")


def example_fetch_all():
    """Example 4: Fetch all data in one call."""
    print("\n" + "=" * 70)
    print("Example 4: Fetch All Data")
    print("=" * 70)

    fetcher = CensusDataFetcher()

    # Fetch all PEP data
    print("\nFetching all PEP data (state and county)...")
    pep_data = fetcher.fetch_all_pep_data(vintage=2024, use_file_method=True)

    print("\nResults:")
    print(f"  State-level: {len(pep_data['state'])} records")
    print(f"  County-level: {len(pep_data['county'])} records")

    # Fetch all ACS data
    print("\nFetching all ACS data...")
    acs_data = fetcher.fetch_all_acs_data(year=2023, dataset="acs5")
    print(f"  Place-level: {len(acs_data)} places")


def example_caching():
    """Example 5: Using cached data."""
    print("\n" + "=" * 70)
    print("Example 5: Caching and Cache Retrieval")
    print("=" * 70)

    fetcher = CensusDataFetcher()

    # List cached files
    print("\nCached files:")
    cached = fetcher.list_cached_files()
    for source, files in cached.items():
        print(f"  {source}: {len(files)} files")
        for file in files[:3]:  # Show first 3
            print(f"    - {file.name}")

    # Try to load cached data
    print("\nAttempting to load cached state data...")
    cached_df = fetcher.get_cached_data("pep", "state", 2024)

    if cached_df is not None:
        print(f"Successfully loaded {len(cached_df)} records from cache")
    else:
        print("No cached data found - would fetch from API")


def example_custom_cache_dir():
    """Example 6: Using a custom cache directory."""
    print("\n" + "=" * 70)
    print("Example 6: Custom Cache Directory")
    print("=" * 70)

    # Use custom cache directory
    custom_cache = Path("/tmp/census_cache")
    fetcher = CensusDataFetcher(cache_dir=custom_cache)

    print(f"\nUsing cache directory: {fetcher.cache_dir}")
    print(f"Cache exists: {fetcher.cache_dir.exists()}")


def example_with_api_key():
    """Example 7: Using Census API key for higher rate limits."""
    print("\n" + "=" * 70)
    print("Example 7: Using API Key")
    print("=" * 70)

    # Get API key from environment variable
    api_key = os.environ.get("CENSUS_API_KEY")

    if api_key:
        print(f"\nUsing Census API key: {api_key[:8]}...")
        fetcher = CensusDataFetcher(api_key=api_key)
        print("API key configured - higher rate limits available")
    else:
        print("\nNo API key found in environment variable CENSUS_API_KEY")
        print("Using public API endpoint (lower rate limits)")
        fetcher = CensusDataFetcher()

    print(f"Max retries: {fetcher.max_retries}")
    print(f"Retry delay: {fetcher.retry_delay} seconds")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Census Data Fetcher Examples")
    print("North Dakota Cohort Projections Project")
    print("=" * 70)

    examples = [
        ("Basic Usage", example_basic_usage),
        ("County Data", example_county_data),
        ("ACS Places", example_acs_places),
        ("Fetch All", example_fetch_all),
        ("Caching", example_caching),
        ("Custom Cache Dir", example_custom_cache_dir),
        ("API Key", example_with_api_key),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRun specific example by uncommenting the function call below,")
    print("or run all examples by uncommenting the loop.")

    # Option 1: Run a specific example
    # example_basic_usage()
    # example_county_data()
    # example_acs_places()
    # example_fetch_all()
    # example_caching()
    # example_custom_cache_dir()
    # example_with_api_key()

    # Option 2: Run all examples (commented out by default)
    # for name, func in examples:
    #     try:
    #         func()
    #     except Exception as e:
    #         print(f"\nError in {name}: {e}")
    #         import traceback
    #         traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
