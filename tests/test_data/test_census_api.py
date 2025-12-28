"""
Unit tests for Census API data fetcher.

Tests the CensusDataFetcher class for proper initialization, API requests,
caching, and error handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

# Mock imports to allow tests to run without dependencies
try:
    from cohort_projections.data.fetch.census_api import CensusDataFetcher

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="Required dependencies not installed")


class TestCensusDataFetcherInitialization:
    """Test CensusDataFetcher initialization."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_init_default_cache_dir(self):
        """Test initialization with default cache directory."""
        fetcher = CensusDataFetcher()

        assert fetcher.STATE_FIPS == "38"
        assert fetcher.STATE_NAME == "North Dakota"
        assert fetcher.cache_dir.exists()
        assert (fetcher.cache_dir / "pep").exists()
        assert (fetcher.cache_dir / "acs").exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_init_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "custom_cache"
            fetcher = CensusDataFetcher(cache_dir=custom_cache)

            assert fetcher.cache_dir == custom_cache
            assert fetcher.cache_dir.exists()
            assert (fetcher.cache_dir / "pep").exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        api_key = "test_api_key_12345"
        fetcher = CensusDataFetcher(api_key=api_key)

        assert fetcher.api_key == api_key

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_init_retry_settings(self):
        """Test initialization with custom retry settings."""
        fetcher = CensusDataFetcher(max_retries=5, retry_delay=10)

        assert fetcher.max_retries == 5
        assert fetcher.retry_delay == 10


class TestCensusDataFetcherRequests:
    """Test HTTP request handling."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("requests.get")
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "success"
        mock_get.return_value = mock_response

        fetcher = CensusDataFetcher()
        response = fetcher._make_request("http://test.url", {}, "Test request")

        assert response.status_code == 200
        assert response.text == "success"
        mock_get.assert_called_once()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("requests.get")
    def test_make_request_with_api_key(self, mock_get):
        """Test request includes API key when provided."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        api_key = "test_key"
        fetcher = CensusDataFetcher(api_key=api_key)
        fetcher._make_request("http://test.url", {"param": "value"}, "Test")

        # Check that API key was added to params
        call_args = mock_get.call_args
        assert call_args[1]["params"]["key"] == api_key

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("requests.get")
    @patch("time.sleep")
    def test_make_request_retry_on_failure(self, mock_sleep, mock_get):
        """Test retry logic on failed requests."""
        # Mock first two calls to fail, third to succeed
        mock_get.side_effect = [
            requests.RequestException("Failure 1"),
            requests.RequestException("Failure 2"),
            Mock(status_code=200),
        ]

        fetcher = CensusDataFetcher(max_retries=3, retry_delay=1)
        response = fetcher._make_request("http://test.url", {}, "Test")

        assert response.status_code == 200
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("requests.get")
    def test_make_request_max_retries_exceeded(self, mock_get):
        """Test exception raised when max retries exceeded."""
        mock_get.side_effect = requests.RequestException("Always fail")

        fetcher = CensusDataFetcher(max_retries=2)

        with pytest.raises(requests.RequestException):
            fetcher._make_request("http://test.url", {}, "Test")

        assert mock_get.call_count == 2


class TestCensusDataFetcherCaching:
    """Test data caching functionality."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_save_metadata(self):
        """Test metadata file creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            fetcher = CensusDataFetcher(cache_dir=cache_dir)

            # Create a test data file
            test_file = cache_dir / "pep" / "test_data.parquet"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.touch()

            # Save metadata
            fetcher._save_metadata(test_file, "PEP", 2024, "http://test.api", 100)

            # Check metadata file exists
            metadata_file = cache_dir / "pep" / "test_data_metadata.json"
            assert metadata_file.exists()

            # Check metadata contents
            with open(metadata_file) as f:
                metadata = json.load(f)

            assert metadata["source"] == "PEP"
            assert metadata["vintage_year"] == 2024
            assert metadata["state_fips"] == "38"
            assert metadata["record_count"] == 100

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_get_cached_data_exists(self):
        """Test retrieving existing cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            fetcher = CensusDataFetcher(cache_dir=cache_dir)

            # Create test data
            test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            cache_file = cache_dir / "pep" / "pep_state_2024.parquet"
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            test_df.to_parquet(cache_file, index=False)

            # Retrieve cached data
            cached_df = fetcher.get_cached_data("pep", "state", 2024)

            assert cached_df is not None
            assert len(cached_df) == 3
            assert list(cached_df.columns) == ["col1", "col2"]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_get_cached_data_not_exists(self):
        """Test retrieving non-existent cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            fetcher = CensusDataFetcher(cache_dir=cache_dir)

            cached_df = fetcher.get_cached_data("pep", "state", 2024)

            assert cached_df is None

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_list_cached_files(self):
        """Test listing cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            fetcher = CensusDataFetcher(cache_dir=cache_dir)

            # Create some test cache files
            (cache_dir / "pep" / "pep_state_2024.parquet").touch()
            (cache_dir / "pep" / "pep_county_2024.parquet").touch()
            (cache_dir / "acs" / "acs5_place_2023.parquet").touch()

            cached_files = fetcher.list_cached_files()

            assert len(cached_files["pep"]) == 2
            assert len(cached_files["acs"]) == 1
            assert len(cached_files["decennial"]) == 0


class TestCensusDataFetcherDataMethods:
    """Test data fetching methods."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.data.fetch.census_api.CensusDataFetcher._make_request")
    def test_fetch_pep_state_data(self, mock_request):
        """Test fetching state-level PEP data."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = [
            ["state", "AGE", "SEX", "RACE", "HISP", "POP", "DATE_CODE", "DATE_DESC"],
            ["38", "0", "0", "0", "0", "779094", "12", "July 1, 2024"],
        ]
        mock_request.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = CensusDataFetcher(cache_dir=Path(tmpdir))
            df = fetcher.fetch_pep_state_data(vintage=2024)

            assert len(df) == 1
            assert "POP" in df.columns
            assert df["POP"].iloc[0] == 779094

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.data.fetch.census_api.CensusDataFetcher._make_request")
    def test_fetch_pep_county_data(self, mock_request):
        """Test fetching county-level PEP data."""
        # Mock API response with multiple counties
        mock_response = Mock()
        mock_response.json.return_value = [
            ["state", "county", "AGE", "SEX", "POP"],
            ["38", "001", "0", "0", "3000"],
            ["38", "003", "0", "0", "4000"],
        ]
        mock_request.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = CensusDataFetcher(cache_dir=Path(tmpdir))
            df = fetcher.fetch_pep_county_data(vintage=2024)

            assert len(df) == 2
            assert df["county"].nunique() == 2

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.data.fetch.census_api.CensusDataFetcher._make_request")
    def test_fetch_acs_place_data(self, mock_request):
        """Test fetching ACS place data."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = [
            ["NAME", "B01001_001E", "state", "place"],
            ["Fargo city, North Dakota", "125990", "38", "25700"],
            ["Bismarck city, North Dakota", "73622", "38", "07200"],
            ["Grand Forks CDP, North Dakota", "5000", "38", "33000"],
        ]
        mock_request.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = CensusDataFetcher(cache_dir=Path(tmpdir))
            df = fetcher.fetch_acs_place_data(year=2023)

            assert len(df) == 3
            assert "is_cdp" in df.columns
            assert "place_type" in df.columns
            assert df["is_cdp"].sum() == 1  # One CDP


class TestCensusDataFetcherConstants:
    """Test class constants and attributes."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_state_constants(self):
        """Test state-related constants."""
        assert CensusDataFetcher.STATE_FIPS == "38"
        assert CensusDataFetcher.STATE_NAME == "North Dakota"

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_api_urls(self):
        """Test API URL templates."""
        assert "{vintage}" in CensusDataFetcher.PEP_BASE_URL
        assert "{year}" in CensusDataFetcher.ACS_BASE_URL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
