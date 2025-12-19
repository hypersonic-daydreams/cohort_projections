#!/usr/bin/env python3
"""
Test BigQuery connection and explore available Census/demographic datasets.

This script verifies that BigQuery credentials are set up correctly and
explores what demographic data is available in BigQuery public datasets.

Usage:
    python scripts/setup/02_test_bigquery_connection.py

Prerequisites:
    1. Service account key saved to: ~/.config/gcloud/cohort-projections-key.json
    2. BigQuery dependencies installed: pip install -r requirements.txt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.utils import get_bigquery_client, get_logger_from_config

logger = get_logger_from_config(__name__)


def test_connection():
    """Test BigQuery connection and credentials."""
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
    except FileNotFoundError as e:
        logger.error("✗ Credentials file not found")
        logger.error(f"  {e}")
        logger.error("\nPlease create a service account key and save it to:")
        logger.error("  ~/.config/gcloud/cohort-projections-key.json")
        return None
    except Exception as e:
        logger.error(f"✗ Failed to connect to BigQuery: {e}")
        return None


def explore_census_datasets(bq):
    """Explore available Census-related datasets."""
    logger.info("\n" + "=" * 80)
    logger.info("Exploring Census & Demographic Datasets")
    logger.info("=" * 80)

    try:
        datasets = bq.list_public_datasets(filter_census=True)
        logger.info(f"\nFound {len(datasets)} Census-related datasets:")

        for idx, row in datasets.head(20).iterrows():
            print(f"  - {row['dataset_id']}")

        return datasets
    except Exception as e:
        logger.error(f"✗ Failed to list datasets: {e}")
        return None


def explore_census_bureau_usa(bq):
    """Explore the main Census Bureau USA dataset."""
    logger.info("\n" + "=" * 80)
    logger.info("Exploring: bigquery-public-data.census_bureau_usa")
    logger.info("=" * 80)

    dataset = "bigquery-public-data.census_bureau_usa"

    try:
        tables = bq.list_tables(dataset)
        logger.info(f"\nFound {len(tables)} tables:")

        for idx, row in tables.iterrows():
            print(f"  - {row['table_name']}")
            if 'size_mb' in row and row['size_mb']:
                print(f"    Size: {row['size_mb']:.2f} MB, Rows: {row['row_count']:,}")

        return tables
    except Exception as e:
        logger.error(f"✗ Failed to list tables: {e}")
        return None


def preview_population_data(bq):
    """Preview population-related tables."""
    logger.info("\n" + "=" * 80)
    logger.info("Previewing Population Data Tables")
    logger.info("=" * 80)

    # Try some common population tables
    potential_tables = [
        "bigquery-public-data.census_bureau_usa.population_by_zip_2010",
        "bigquery-public-data.census_bureau_usa.population_by_zip_2020",
        "bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr",
    ]

    for table_ref in potential_tables:
        try:
            logger.info(f"\n--- {table_ref.split('.')[-1]} ---")

            # Get schema
            schema = bq.get_table_schema(table_ref)
            logger.info(f"Columns ({len(schema)}):")
            for idx, col in schema.head(10).iterrows():
                print(f"  - {col['column_name']}: {col['data_type']}")

            # Preview data
            preview = bq.preview_table(table_ref, limit=3)
            logger.info(f"\nPreview ({len(preview)} rows):")
            print(preview.to_string(max_rows=3, max_cols=8))

        except Exception as e:
            logger.warning(f"Could not access {table_ref}: {e}")
            continue


def check_acs_datasets(bq):
    """Check American Community Survey datasets."""
    logger.info("\n" + "=" * 80)
    logger.info("Checking American Community Survey (ACS) Datasets")
    logger.info("=" * 80)

    try:
        # List ACS datasets
        sql = """
        SELECT
            schema_name as dataset_id,
            catalog_name as project_id
        FROM `bigquery-public-data.INFORMATION_SCHEMA.SCHEMATA`
        WHERE schema_name LIKE '%acs%'
        ORDER BY schema_name
        """

        datasets = bq.query(sql)
        logger.info(f"\nFound {len(datasets)} ACS-related datasets:")

        for idx, row in datasets.iterrows():
            print(f"  - {row['dataset_id']}")

        # Explore census_bureau_acs dataset
        if len(datasets) > 0:
            logger.info("\n--- Tables in census_bureau_acs ---")
            tables = bq.list_tables("bigquery-public-data.census_bureau_acs")
            for idx, row in tables.head(15).iterrows():
                print(f"  - {row['table_name']}")

    except Exception as e:
        logger.error(f"✗ Failed to check ACS datasets: {e}")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("BigQuery Connection & Data Exploration Test")
    print("=" * 80 + "\n")

    # Step 1: Test connection
    bq = test_connection()
    if bq is None:
        logger.error("\n❌ Cannot proceed without valid BigQuery connection")
        return 1

    # Step 2: Explore Census datasets
    datasets = explore_census_datasets(bq)

    # Step 3: Explore Census Bureau USA dataset
    if datasets is not None:
        tables = explore_census_bureau_usa(bq)

    # Step 4: Preview population data
    if tables is not None:
        preview_population_data(bq)

    # Step 5: Check ACS datasets
    check_acs_datasets(bq)

    # Success
    logger.info("\n" + "=" * 80)
    logger.info("✓ BigQuery exploration completed successfully!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Review available datasets and tables above")
    logger.info("  2. Identify which tables contain demographic rates (fertility, mortality, migration)")
    logger.info("  3. Start implementing data fetchers for those tables")

    bq.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
