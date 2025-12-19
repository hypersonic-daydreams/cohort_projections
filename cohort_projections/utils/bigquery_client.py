"""
Google BigQuery client utilities for demographic data access.

This module provides a wrapper around the Google Cloud BigQuery client,
configured for accessing Census and demographic data from BigQuery public datasets.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

from .logger import get_logger_from_config
from .config_loader import load_projection_config

logger = get_logger_from_config(__name__)


class BigQueryClient:
    """
    BigQuery client for demographic data access.

    Handles authentication, connection management, and provides
    convenience methods for querying Census and demographic data.

    Attributes:
        client (bigquery.Client): Authenticated BigQuery client
        project_id (str): GCP project ID
        dataset_id (str): Default dataset ID for queries
        location (str): BigQuery location (e.g., 'US')

    Example:
        >>> bq = BigQueryClient()
        >>> df = bq.query("SELECT * FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010` LIMIT 10")
        >>> print(df.head())
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BigQuery client with credentials.

        Args:
            credentials_path: Path to service account JSON key file.
                If None, reads from config or environment variable.
            project_id: GCP project ID. If None, reads from config.
            config: Configuration dictionary. If None, loads from config file.

        Raises:
            ValueError: If credentials or project ID cannot be determined
            FileNotFoundError: If credentials file doesn't exist
        """
        # Load configuration
        if config is None:
            config = load_projection_config()

        self.config = config.get('bigquery', {})

        # Determine project ID
        self.project_id = project_id or self.config.get('project_id')
        if not self.project_id:
            raise ValueError(
                "BigQuery project_id must be provided either as argument or in config file"
            )

        # Determine credentials path
        if credentials_path is None:
            credentials_path = self.config.get('credentials_path')

        if not credentials_path:
            raise ValueError(
                "BigQuery credentials_path must be provided either as argument or in config file"
            )

        # Expand user path (e.g., ~/)
        credentials_path = Path(credentials_path).expanduser()

        if not credentials_path.exists():
            raise FileNotFoundError(
                f"BigQuery credentials file not found: {credentials_path}\n"
                f"Please create service account key and save to: {credentials_path}"
            )

        # Authenticate and create client
        logger.info(f"Authenticating with BigQuery using credentials: {credentials_path}")

        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                str(credentials_path)
            )
            self.client = bigquery.Client(
                credentials=self.credentials,
                project=self.project_id
            )
            logger.info(f"BigQuery client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise

        # Set attributes from config
        self.dataset_id = self.config.get('dataset_id', 'demographic_data')
        self.location = self.config.get('location', 'US')
        self.use_public_data = self.config.get('use_public_data', True)
        self.cache_queries = self.config.get('cache_queries', True)

    def query(
        self,
        sql: str,
        to_dataframe: bool = True,
        use_cache: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results.

        Args:
            sql: SQL query string
            to_dataframe: If True, return results as pandas DataFrame
            use_cache: If True, use BigQuery cache. If None, uses config default.

        Returns:
            Query results as DataFrame if to_dataframe=True, else QueryJob

        Raises:
            GoogleAPIError: If query fails

        Example:
            >>> sql = "SELECT * FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010` LIMIT 10"
            >>> df = bq.query(sql)
        """
        if use_cache is None:
            use_cache = self.cache_queries

        logger.debug(f"Executing query (use_cache={use_cache}):\n{sql[:200]}...")

        try:
            job_config = bigquery.QueryJobConfig(use_query_cache=use_cache)
            query_job = self.client.query(sql, job_config=job_config)

            if to_dataframe:
                df = query_job.to_dataframe()
                logger.info(f"Query returned {len(df)} rows, {len(df.columns)} columns")
                return df
            else:
                return query_job

        except GoogleAPIError as e:
            logger.error(f"BigQuery query failed: {e}")
            raise

    def list_public_datasets(self, filter_census: bool = True) -> pd.DataFrame:
        """
        List available public datasets in BigQuery.

        Args:
            filter_census: If True, only show Census-related datasets

        Returns:
            DataFrame with dataset information

        Example:
            >>> datasets = bq.list_public_datasets()
            >>> print(datasets[['dataset_id', 'description']])
        """
        logger.info("Listing BigQuery public datasets...")

        sql = """
        SELECT
            schema_name as dataset_id,
            catalog_name as project_id
        FROM `bigquery-public-data.INFORMATION_SCHEMA.SCHEMATA`
        """

        if filter_census:
            sql += " WHERE schema_name LIKE '%census%' OR schema_name LIKE '%population%'"

        sql += " ORDER BY schema_name"

        return self.query(sql)

    def list_tables(self, dataset: str = "bigquery-public-data.census_bureau_usa") -> pd.DataFrame:
        """
        List tables in a dataset.

        Args:
            dataset: Full dataset name (project.dataset)

        Returns:
            DataFrame with table information

        Example:
            >>> tables = bq.list_tables("bigquery-public-data.census_bureau_usa")
            >>> print(tables['table_name'].tolist())
        """
        logger.info(f"Listing tables in dataset: {dataset}")

        project, dataset_id = dataset.split('.')

        sql = f"""
        SELECT
            table_name,
            table_type,
            ROUND(size_bytes / 1024 / 1024, 2) as size_mb,
            row_count
        FROM `{project}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
        ORDER BY table_name
        """

        return self.query(sql)

    def get_table_schema(self, table_ref: str) -> pd.DataFrame:
        """
        Get schema information for a table.

        Args:
            table_ref: Full table reference (project.dataset.table)

        Returns:
            DataFrame with column information

        Example:
            >>> schema = bq.get_table_schema("bigquery-public-data.census_bureau_usa.population_by_zip_2010")
            >>> print(schema[['column_name', 'data_type', 'description']])
        """
        logger.info(f"Getting schema for table: {table_ref}")

        parts = table_ref.split('.')
        if len(parts) != 3:
            raise ValueError(f"Table reference must be in format 'project.dataset.table', got: {table_ref}")

        project, dataset, table = parts

        sql = f"""
        SELECT
            column_name,
            data_type,
            is_nullable,
            description
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table}'
        ORDER BY ordinal_position
        """

        return self.query(sql)

    def preview_table(self, table_ref: str, limit: int = 10) -> pd.DataFrame:
        """
        Preview first N rows of a table.

        Args:
            table_ref: Full table reference (project.dataset.table)
            limit: Number of rows to return

        Returns:
            DataFrame with preview data

        Example:
            >>> preview = bq.preview_table("bigquery-public-data.census_bureau_usa.population_by_zip_2010", 5)
        """
        logger.info(f"Previewing table: {table_ref} (limit={limit})")

        sql = f"SELECT * FROM `{table_ref}` LIMIT {limit}"
        return self.query(sql)

    def create_dataset(self, dataset_id: Optional[str] = None) -> None:
        """
        Create a BigQuery dataset for storing processed demographic data.

        Args:
            dataset_id: Dataset ID to create. If None, uses config default.

        Raises:
            GoogleAPIError: If dataset creation fails
        """
        if dataset_id is None:
            dataset_id = self.dataset_id

        dataset_ref = f"{self.project_id}.{dataset_id}"

        try:
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_ref} already exists")
        except Exception:
            logger.info(f"Creating dataset: {dataset_ref}")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = self.location
            self.client.create_dataset(dataset)
            logger.info(f"Dataset {dataset_ref} created successfully")

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        dataset_id: Optional[str] = None,
        write_disposition: str = "WRITE_TRUNCATE"
    ) -> None:
        """
        Upload a pandas DataFrame to BigQuery.

        Args:
            df: DataFrame to upload
            table_name: Name of table to create/update
            dataset_id: Dataset ID. If None, uses config default.
            write_disposition: Write mode - WRITE_TRUNCATE (replace), WRITE_APPEND, WRITE_EMPTY

        Raises:
            GoogleAPIError: If upload fails

        Example:
            >>> df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
            >>> bq.upload_dataframe(df, 'my_table')
        """
        if dataset_id is None:
            dataset_id = self.dataset_id

        table_ref = f"{self.project_id}.{dataset_id}.{table_name}"

        logger.info(f"Uploading {len(df)} rows to {table_ref}")

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition
        )

        try:
            job = self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            job.result()  # Wait for job to complete
            logger.info(f"Successfully uploaded {len(df)} rows to {table_ref}")
        except GoogleAPIError as e:
            logger.error(f"Failed to upload DataFrame to BigQuery: {e}")
            raise

    def close(self) -> None:
        """Close the BigQuery client connection."""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("BigQuery client connection closed")


def get_bigquery_client(config: Optional[Dict[str, Any]] = None) -> BigQueryClient:
    """
    Get a configured BigQuery client instance.

    Convenience function for quick client access.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured BigQueryClient instance

    Example:
        >>> bq = get_bigquery_client()
        >>> df = bq.query("SELECT * FROM ...")
    """
    return BigQueryClient(config=config)
