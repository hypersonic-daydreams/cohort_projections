"""
Shared pytest fixtures for utils module tests.

Provides fixtures for configuration, mock clients, and sample data.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def temp_config_dir(tmp_path: Path) -> Path:
    """
    Create a temporary config directory with sample config files.

    Parameters
    ----------
    tmp_path : Path
        Pytest built-in fixture providing a temporary directory

    Returns
    -------
    Path
        Path to the created config directory
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_projection_config() -> dict[str, Any]:
    """
    Sample projection configuration dictionary.

    Provides a complete configuration structure for testing ConfigLoader
    and related utilities.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with projection, demographics, bigquery,
        and logging sections
    """
    return {
        "projection": {
            "base_year": 2020,
            "end_year": 2050,
            "components": ["fertility", "mortality", "migration"],
        },
        "demographics": {
            "age_groups": {"type": "single_year", "max_age": 90},
            "sex_categories": ["Male", "Female"],
            "race_categories": ["White", "Black", "Hispanic", "Asian", "Other"],
        },
        "bigquery": {
            "project_id": "test-project",
            "credentials_path": "~/.gcp/test-creds.json",
            "dataset_id": "demographic_data",
            "location": "US",
        },
        "logging": {"level": "INFO", "log_to_file": False},
    }


@pytest.fixture
def sample_projection_config_file(
    temp_config_dir: Path, sample_projection_config: dict[str, Any]
) -> Path:
    """
    Create a sample projection_config.yaml file.

    Parameters
    ----------
    temp_config_dir : Path
        Temporary config directory fixture
    sample_projection_config : dict[str, Any]
        Sample configuration dictionary fixture

    Returns
    -------
    Path
        Path to the created YAML config file
    """
    import yaml

    config_path = temp_config_dir / "projection_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_projection_config, f)
    return config_path


@pytest.fixture
def mock_bigquery_client() -> MagicMock:
    """
    Mock BigQuery client for testing.

    Provides a MagicMock configured to return sample query results
    without requiring actual BigQuery credentials.

    Returns
    -------
    MagicMock
        Mock client with query() method returning sample DataFrame
    """
    mock_client = MagicMock()

    # Mock query result
    mock_query_job = MagicMock()
    mock_query_job.to_dataframe.return_value = pd.DataFrame(
        {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    )
    mock_client.query.return_value = mock_query_job

    return mock_client


@pytest.fixture
def mock_credentials() -> MagicMock:
    """
    Mock GCP service account credentials.

    Returns
    -------
    MagicMock
        Mock credentials with project_id attribute set to "test-project"
    """
    mock_creds = MagicMock()
    mock_creds.project_id = "test-project"
    return mock_creds


@pytest.fixture
def sample_population_df() -> pd.DataFrame:
    """
    Sample population DataFrame for demographic utils testing.

    Synthetic Data Characteristics
    ------------------------------
    - Ages: 0-90 (91 single-year age groups)
    - Sexes: Male, Female (n=182 total rows)
    - Population distribution: Peaks around age 30-50
      - Ages 0-19: Declining from 1000 to ~800
      - Ages 20-49: Rising from 800 to ~950
      - Ages 50-90: Declining from 950 to minimum 50
    - Minimum population: 50 per cohort

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: age, sex, population
    """
    data = []
    for sex in ["Male", "Female"]:
        for age in range(91):
            # Create age distribution that peaks around 30-40
            if age < 20:
                pop = 1000 - age * 10
            elif age < 50:
                pop = 800 + (age - 20) * 5
            else:
                pop = 950 - (age - 50) * 15
            pop = max(pop, 50)  # Minimum population
            data.append({"age": age, "sex": sex, "population": pop})
    return pd.DataFrame(data)


@pytest.fixture
def mock_db_connection() -> MagicMock:
    """
    Mock psycopg2 database connection.

    Provides a mock database connection for testing reproducibility
    logging without requiring an actual PostgreSQL database.

    Returns
    -------
    MagicMock
        Mock connection with cursor() method returning a mock cursor
        configured to return (1,) as script ID from fetchone()
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)  # Script ID
    return mock_conn
