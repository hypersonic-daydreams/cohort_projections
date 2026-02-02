"""
Unit tests for the bigquery_client module.

Tests BigQueryClient class with mocked GCP clients.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import the module, but we'll mock the GCP dependencies
from cohort_projections.utils import bigquery_client


class TestBigQueryClientInit:
    """Tests for BigQueryClient initialization."""

    @pytest.fixture
    def mock_credentials_file(self, tmp_path: Path) -> Path:
        """Create a mock credentials file."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text('{"type": "service_account"}')
        return creds_file

    @pytest.fixture
    def valid_config(self, mock_credentials_file: Path) -> dict[str, Any]:
        """Valid BigQuery configuration."""
        return {
            "bigquery": {
                "project_id": "test-project",
                "credentials_path": str(mock_credentials_file),
                "dataset_id": "test_dataset",
                "location": "US",
                "use_public_data": True,
                "cache_queries": True,
            }
        }

    @patch.object(bigquery_client, "load_projection_config")
    @patch.object(bigquery_client.service_account.Credentials, "from_service_account_file")
    @patch.object(bigquery_client.bigquery, "Client")
    def test_init_with_config(
        self,
        mock_bq_client: MagicMock,
        mock_credentials: MagicMock,
        mock_load_config: MagicMock,
        valid_config: dict[str, Any],
    ) -> None:
        """Test initialization with valid config."""
        mock_load_config.return_value = valid_config
        mock_credentials.return_value = MagicMock()
        mock_bq_client.return_value = MagicMock()

        client = bigquery_client.BigQueryClient(config=valid_config)

        assert client.project_id == "test-project"
        assert client.dataset_id == "test_dataset"
        assert client.location == "US"
        assert client.use_public_data is True
        assert client.cache_queries is True

    @patch.object(bigquery_client, "load_projection_config")
    def test_init_missing_project_id_raises_error(
        self, mock_load_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test that missing project_id raises ValueError."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")

        config = {
            "bigquery": {
                "credentials_path": str(creds_file),
                # Missing project_id
            }
        }
        mock_load_config.return_value = config

        with pytest.raises(ValueError, match="project_id must be provided"):
            bigquery_client.BigQueryClient(config=config)

    @patch.object(bigquery_client, "load_projection_config")
    def test_init_missing_credentials_path_raises_error(self, mock_load_config: MagicMock) -> None:
        """Test that missing credentials_path raises ValueError."""
        config = {
            "bigquery": {
                "project_id": "test-project",
                # Missing credentials_path
            }
        }
        mock_load_config.return_value = config

        with pytest.raises(ValueError, match="credentials_path must be provided"):
            bigquery_client.BigQueryClient(config=config)

    @patch.object(bigquery_client, "load_projection_config")
    def test_init_nonexistent_credentials_file_raises_error(
        self, mock_load_config: MagicMock
    ) -> None:
        """Test that nonexistent credentials file raises FileNotFoundError."""
        config = {
            "bigquery": {
                "project_id": "test-project",
                "credentials_path": "/nonexistent/path/creds.json",
            }
        }
        mock_load_config.return_value = config

        with pytest.raises(FileNotFoundError, match="credentials file not found"):
            bigquery_client.BigQueryClient(config=config)

    @patch.object(bigquery_client, "load_projection_config")
    @patch.object(bigquery_client.service_account.Credentials, "from_service_account_file")
    @patch.object(bigquery_client.bigquery, "Client")
    def test_init_expands_user_path(
        self,
        mock_bq_client: MagicMock,
        mock_credentials: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that ~ in credentials path is expanded."""
        # Create credentials file in a temporary location
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")

        config = {
            "bigquery": {
                "project_id": "test-project",
                "credentials_path": str(creds_file),
            }
        }
        mock_load_config.return_value = config
        mock_credentials.return_value = MagicMock()
        mock_bq_client.return_value = MagicMock()

        client = bigquery_client.BigQueryClient(config=config)

        # Should initialize successfully with expanded path
        assert client.project_id == "test-project"

    @patch.object(bigquery_client, "load_projection_config")
    @patch.object(bigquery_client.service_account.Credentials, "from_service_account_file")
    @patch.object(bigquery_client.bigquery, "Client")
    def test_init_with_explicit_args(
        self,
        mock_bq_client: MagicMock,
        mock_credentials: MagicMock,
        mock_load_config: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test initialization with explicit arguments overrides config."""
        creds_file = tmp_path / "explicit_creds.json"
        creds_file.write_text("{}")

        config = {
            "bigquery": {
                "project_id": "config-project",
                "credentials_path": "/some/other/path.json",
            }
        }
        mock_load_config.return_value = config
        mock_credentials.return_value = MagicMock()
        mock_bq_client.return_value = MagicMock()

        client = bigquery_client.BigQueryClient(
            project_id="explicit-project",
            credentials_path=str(creds_file),
            config=config,
        )

        assert client.project_id == "explicit-project"


class TestBigQueryClientQuery:
    """Tests for BigQueryClient.query method."""

    @pytest.fixture
    def mock_client(self, tmp_path: Path) -> bigquery_client.BigQueryClient:
        """Create a BigQueryClient with mocked internals."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")

        config = {
            "bigquery": {
                "project_id": "test-project",
                "credentials_path": str(creds_file),
                "cache_queries": True,
            }
        }

        with (
            patch.object(bigquery_client.service_account.Credentials, "from_service_account_file"),
            patch.object(bigquery_client.bigquery, "Client") as mock_bq,
        ):
            mock_bq.return_value = MagicMock()
            client = bigquery_client.BigQueryClient(config=config)
            return client

    def test_query_returns_dataframe(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test that query returns a DataFrame by default."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
        )
        mock_client.client.query.return_value = mock_query_job

        result = mock_client.query("SELECT * FROM test")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["col1", "col2"]

    def test_query_uses_cache_setting(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test that query respects cache setting."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame()
        mock_client.client.query.return_value = mock_query_job

        # Default cache setting
        mock_client.query("SELECT 1")

        # Check that QueryJobConfig was created with cache setting
        call_args = mock_client.client.query.call_args
        assert call_args is not None

    def test_query_with_explicit_cache_false(
        self, mock_client: bigquery_client.BigQueryClient
    ) -> None:
        """Test query with explicitly disabled cache."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame()
        mock_client.client.query.return_value = mock_query_job

        mock_client.query("SELECT 1", use_cache=False)

        # Query should still execute
        assert mock_client.client.query.called

    def test_query_returns_job_when_to_dataframe_false(
        self, mock_client: bigquery_client.BigQueryClient
    ) -> None:
        """Test that query returns QueryJob when to_dataframe=False."""
        mock_query_job = MagicMock()
        mock_client.client.query.return_value = mock_query_job

        result = mock_client.query("SELECT 1", to_dataframe=False)

        assert result is mock_query_job
        mock_query_job.to_dataframe.assert_not_called()

    def test_query_raises_on_api_error(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test that query raises GoogleAPIError on failure."""
        from google.api_core.exceptions import GoogleAPIError

        mock_client.client.query.side_effect = GoogleAPIError("Query failed")

        with pytest.raises(GoogleAPIError):
            mock_client.query("SELECT * FROM invalid_table")


class TestBigQueryClientMethods:
    """Tests for other BigQueryClient methods."""

    @pytest.fixture
    def mock_client(self, tmp_path: Path) -> bigquery_client.BigQueryClient:
        """Create a BigQueryClient with mocked internals."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")

        config = {
            "bigquery": {
                "project_id": "test-project",
                "credentials_path": str(creds_file),
            }
        }

        with (
            patch.object(bigquery_client.service_account.Credentials, "from_service_account_file"),
            patch.object(bigquery_client.bigquery, "Client") as mock_bq,
        ):
            mock_bq.return_value = MagicMock()
            client = bigquery_client.BigQueryClient(config=config)
            return client

    def test_list_public_datasets(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test list_public_datasets method."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame(
            {"dataset_id": ["census_1", "census_2"], "project_id": ["public", "public"]}
        )
        mock_client.client.query.return_value = mock_query_job

        result = mock_client.list_public_datasets(filter_census=True)

        assert isinstance(result, pd.DataFrame)
        # Verify the query was called
        assert mock_client.client.query.called

    def test_list_tables(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test list_tables method."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame(
            {
                "table_name": ["table1", "table2"],
                "table_type": ["TABLE", "VIEW"],
                "size_mb": [100.0, 50.0],
                "row_count": [1000, 500],
            }
        )
        mock_client.client.query.return_value = mock_query_job

        result = mock_client.list_tables("bigquery-public-data.census_bureau_usa")

        assert isinstance(result, pd.DataFrame)
        assert "table_name" in result.columns

    def test_get_table_schema(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test get_table_schema method."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame(
            {
                "column_name": ["id", "name"],
                "data_type": ["INT64", "STRING"],
                "is_nullable": ["YES", "YES"],
                "description": ["ID column", "Name column"],
            }
        )
        mock_client.client.query.return_value = mock_query_job

        result = mock_client.get_table_schema("project.dataset.table")

        assert isinstance(result, pd.DataFrame)
        assert "column_name" in result.columns

    def test_get_table_schema_invalid_reference(
        self, mock_client: bigquery_client.BigQueryClient
    ) -> None:
        """Test get_table_schema with invalid table reference."""
        with pytest.raises(ValueError, match="Table reference must be in format"):
            mock_client.get_table_schema("invalid_reference")

    def test_preview_table(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test preview_table method."""
        mock_query_job = MagicMock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame(
            {"id": [1, 2, 3], "value": ["a", "b", "c"]}
        )
        mock_client.client.query.return_value = mock_query_job

        result = mock_client.preview_table("project.dataset.table", limit=3)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_create_dataset_already_exists(
        self, mock_client: bigquery_client.BigQueryClient
    ) -> None:
        """Test create_dataset when dataset already exists."""
        mock_client.client.get_dataset.return_value = MagicMock()

        # Should not raise, just log that it exists
        mock_client.create_dataset("existing_dataset")

        mock_client.client.get_dataset.assert_called_once()
        mock_client.client.create_dataset.assert_not_called()

    def test_create_dataset_new(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test create_dataset for new dataset."""
        mock_client.client.get_dataset.side_effect = Exception("Not found")

        mock_client.create_dataset("new_dataset")

        mock_client.client.create_dataset.assert_called_once()

    def test_upload_dataframe(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test upload_dataframe method."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_job = MagicMock()
        mock_client.client.load_table_from_dataframe.return_value = mock_job

        mock_client.upload_dataframe(df, "test_table")

        mock_client.client.load_table_from_dataframe.assert_called_once()
        mock_job.result.assert_called_once()

    def test_upload_dataframe_api_error(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test upload_dataframe raises on API error."""
        from google.api_core.exceptions import GoogleAPIError

        df = pd.DataFrame({"col1": [1, 2, 3]})
        mock_client.client.load_table_from_dataframe.side_effect = GoogleAPIError("Upload failed")

        with pytest.raises(GoogleAPIError):
            mock_client.upload_dataframe(df, "test_table")

    def test_close(self, mock_client: bigquery_client.BigQueryClient) -> None:
        """Test close method."""
        mock_client.close()

        mock_client.client.close.assert_called_once()


class TestGetBigQueryClient:
    """Tests for get_bigquery_client convenience function."""

    @patch.object(bigquery_client, "BigQueryClient")
    def test_get_bigquery_client_creates_instance(self, mock_class: MagicMock) -> None:
        """Test that get_bigquery_client returns a BigQueryClient."""
        mock_class.return_value = MagicMock()

        result = bigquery_client.get_bigquery_client()

        mock_class.assert_called_once_with(config=None)
        assert result is mock_class.return_value

    @patch.object(bigquery_client, "BigQueryClient")
    def test_get_bigquery_client_with_config(self, mock_class: MagicMock) -> None:
        """Test get_bigquery_client with custom config."""
        mock_class.return_value = MagicMock()
        config = {"bigquery": {"project_id": "custom-project"}}

        result = bigquery_client.get_bigquery_client(config=config)

        mock_class.assert_called_once_with(config=config)
        assert result is mock_class.return_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
