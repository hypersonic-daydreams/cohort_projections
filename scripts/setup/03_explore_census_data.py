#!/usr/bin/env python3
"""
Explore Census data available in BigQuery public datasets.

This script queries known Census tables directly without using INFORMATION_SCHEMA,
which requires additional permissions.

Usage:
    python scripts/setup/03_explore_census_data.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.utils import get_bigquery_client, get_logger_from_config  # noqa: E402

logger = get_logger_from_config(__name__)


def test_connection():
    """Test BigQuery connection."""
    logger.info("=" * 80)
    logger.info("Testing BigQuery Connection")
    logger.info("=" * 80)

    try:
        bq = get_bigquery_client()
        logger.info("✓ BigQuery client initialized successfully")
        logger.info(f"  Project ID: {bq.project_id}")
        logger.info(f"  Dataset ID: {bq.dataset_id}")
        logger.info(f"  Location: {bq.location}")
        return bq
    except Exception as e:
        logger.error(f"✗ Failed to connect to BigQuery: {e}")
        return None


def explore_census_population_data(bq):
    """Explore Census population tables by direct query."""
    logger.info("\n" + "=" * 80)
    logger.info("Exploring Census Population Data")
    logger.info("=" * 80)

    # Try querying population by ZIP 2010
    try:
        logger.info("\n--- Testing: census_bureau_usa.population_by_zip_2010 ---")

        sql = """
        SELECT *
        FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010`
        WHERE state_code = '38'  -- North Dakota
        LIMIT 10
        """

        df = bq.query(sql)
        logger.info(f"✓ Successfully queried! Found {len(df)} rows")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info("\nSample data:")
        print(df.head(3).to_string())

        return df
    except Exception as e:
        logger.error(f"✗ Failed to query population table: {e}")
        return None


def explore_acs_demographics(bq):
    """Explore American Community Survey data."""
    logger.info("\n" + "=" * 80)
    logger.info("Exploring ACS Demographic Data")
    logger.info("=" * 80)

    try:
        logger.info("\n--- Testing: census_bureau_acs ---")

        # Try blockgroup_2018_5yr (has demographic details)
        sql = """
        SELECT
            geo_id,
            total_pop,
            male_pop,
            female_pop,
            median_age,
            white_pop,
            black_pop
        FROM `bigquery-public-data.census_bureau_acs.blockgroup_2018_5yr`
        WHERE state_code = '38'  -- North Dakota
        LIMIT 10
        """

        df = bq.query(sql)
        logger.info(f"✓ Successfully queried ACS data! Found {len(df)} rows")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info("\nSample data:")
        print(df.head(3).to_string())

        return df
    except Exception as e:
        logger.warning(f"Could not query ACS blockgroup: {e}")

        # Try county_2018_5yr instead
        try:
            logger.info("\n--- Trying: census_bureau_acs.county_2018_5yr ---")

            sql = """
            SELECT *
            FROM `bigquery-public-data.census_bureau_acs.county_2018_5yr`
            WHERE state_fips_code = '38'  -- North Dakota
            LIMIT 5
            """

            df = bq.query(sql)
            logger.info(f"✓ Successfully queried ACS county data! Found {len(df)} rows")
            logger.info(f"  Total columns: {len(df.columns)}")
            logger.info(f"  Sample columns: {df.columns.tolist()[:10]}")

            return df
        except Exception as e2:
            logger.error(f"✗ Failed to query ACS county data: {e2}")
            return None


def explore_sdoh_data(bq):
    """Explore Social Determinants of Health (SDOH) data if available."""
    logger.info("\n" + "=" * 80)
    logger.info("Exploring SDOH Data")
    logger.info("=" * 80)

    try:
        logger.info("\n--- Testing: sdoh_cdc_wonder_natality ---")

        sql = """
        SELECT *
        FROM `bigquery-public-data.sdoh_cdc_wonder_natality.county_natality`
        WHERE state = 'ND'
        LIMIT 10
        """

        df = bq.query(sql)
        logger.info(f"✓ Successfully queried SDOH natality data! Found {len(df)} rows")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info("\nSample data:")
        print(df.head(3).to_string())

        return df
    except Exception as e:
        logger.warning(f"SDOH natality data not accessible: {e}")
        return None


def check_available_public_datasets(bq):
    """Check what public datasets are available by trying known ones."""
    logger.info("\n" + "=" * 80)
    logger.info("Checking Known Public Datasets")
    logger.info("=" * 80)

    datasets_to_try = [
        "bigquery-public-data.census_bureau_usa",
        "bigquery-public-data.census_bureau_acs",
        "bigquery-public-data.census_bureau_tiger",
        "bigquery-public-data.sdoh_cdc_wonder_natality",
    ]

    available = []
    for dataset in datasets_to_try:
        try:
            # Try to get dataset metadata
            test_query = f"SELECT 1 FROM `{dataset}.__TABLES__` LIMIT 1"
            bq.query(test_query)
            available.append(dataset)
            logger.info(f"  ✓ {dataset}")
        except Exception:
            logger.info(f"  ✗ {dataset} - not accessible")

    return available


def check_north_dakota_data_availability(bq):
    """Check what North Dakota-specific data is available."""
    logger.info("\n" + "=" * 80)
    logger.info("North Dakota Data Availability Check")
    logger.info("=" * 80)

    queries = {
        "Population by ZIP (2010)": """
            SELECT COUNT(*) as count
            FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010`
            WHERE state_code = '38'
        """,
        "Population by ZIP (2020)": """
            SELECT COUNT(*) as count
            FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2020`
            WHERE state_code = '38'
        """,
        "ACS County Data (2018)": """
            SELECT COUNT(*) as count
            FROM `bigquery-public-data.census_bureau_acs.county_2018_5yr`
            WHERE state_fips_code = '38'
        """,
    }

    results = {}
    for name, sql in queries.items():
        try:
            df = bq.query(sql)
            count = df["count"].iloc[0]
            results[name] = count
            logger.info(f"  ✓ {name}: {count:,} records")
        except Exception:
            results[name] = None
            logger.warning(f"  ✗ {name}: Not accessible")

    return results


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("BigQuery Census Data Exploration")
    print("=" * 80 + "\n")

    # Test connection
    bq = test_connection()
    if bq is None:
        logger.error("\n❌ Cannot proceed without valid BigQuery connection")
        return 1

    # Check available datasets
    check_available_public_datasets(bq)

    # Check ND data availability
    check_north_dakota_data_availability(bq)

    # Explore population data
    pop_df = explore_census_population_data(bq)

    # Explore ACS data
    acs_df = explore_acs_demographics(bq)

    # Try SDOH data
    sdoh_df = explore_sdoh_data(bq)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Exploration Summary")
    logger.info("=" * 80)

    if pop_df is not None:
        logger.info("✓ Census population data: ACCESSIBLE")
    if acs_df is not None:
        logger.info("✓ ACS demographic data: ACCESSIBLE")
    if sdoh_df is not None:
        logger.info("✓ SDOH natality data: ACCESSIBLE")

    logger.info("\n" + "=" * 80)
    logger.info("Next Steps")
    logger.info("=" * 80)
    logger.info("\nBigQuery provides:")
    logger.info("  ✓ Base population data (Census)")
    logger.info("  ✓ Demographic breakdowns (ACS)")
    logger.info("  ? Natality/fertility data (SDOH - check if accessible)")
    logger.info("\nStill need to obtain separately:")
    logger.info("  • SEER fertility rates (age-specific by race)")
    logger.info("  • SEER/CDC life tables (survival rates)")
    logger.info("  • IRS migration flows (county-to-county)")
    logger.info("\nRecommendation:")
    logger.info("  1. Use BigQuery for base population data")
    logger.info("  2. Download SEER/CDC data for demographic rates")
    logger.info("  3. Start implementing fertility_rates.py processor")

    bq.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
