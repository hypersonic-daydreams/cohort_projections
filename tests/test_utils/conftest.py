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
    """Create a temporary config directory with sample config files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_projection_config() -> dict[str, Any]:
    """Sample projection configuration dictionary."""
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
    """Create a sample projection_config.yaml file."""
    import yaml

    config_path = temp_config_dir / "projection_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_projection_config, f)
    return config_path


@pytest.fixture
def mock_bigquery_client() -> MagicMock:
    """Mock BigQuery client for testing."""
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
    """Mock GCP service account credentials."""
    mock_creds = MagicMock()
    mock_creds.project_id = "test-project"
    return mock_creds


@pytest.fixture
def sample_population_df() -> pd.DataFrame:
    """Sample population DataFrame for demographic utils testing."""
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
    """Mock psycopg2 database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)  # Script ID
    return mock_conn
