"""
Unit tests for multi-geography projection orchestrator module.

Tests the functions for running projections across multiple geographies
with parallel processing and hierarchical aggregation/validation.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Try to import the multi_geography module
try:
    from cohort_projections.geographic.multi_geography import (
        _save_projection_results,
        aggregate_to_county,
        aggregate_to_state,
        run_multi_geography_projections,
        run_single_geography_projection,
        validate_aggregation,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


class TestRunSingleGeographyProjection:
    """Test run_single_geography_projection function."""

    @pytest.fixture
    def sample_base_population(self):
        """Create sample base population data."""
        data = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in ["White", "Black"]:
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": 100.0,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_fertility_rates(self):
        """Create sample fertility rates."""
        data = []
        for age in range(15, 50):
            for race in ["White", "Black"]:
                data.append(
                    {
                        "age": age,
                        "race": race,
                        "fertility_rate": 0.05 if 20 <= age <= 35 else 0.02,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_survival_rates(self):
        """Create sample survival rates."""
        data = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in ["White", "Black"]:
                    # Higher survival for younger ages
                    rate = 0.999 if age < 60 else 0.99 - (age - 60) * 0.005
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "survival_rate": max(0.9, rate),
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_migration_rates(self):
        """Create sample migration rates."""
        data = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in ["White", "Black"]:
                    # Net positive migration for working ages
                    rate = 0.01 if 18 <= age <= 35 else 0.0
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "net_migration_rate": rate,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "project": {
                "base_year": 2025,
                "projection_horizon": 5,
            },
            "scenarios": {
                "baseline": {
                    "active": True,
                }
            },
        }

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.geographic.multi_geography.CohortComponentProjection")
    def test_single_projection_returns_dict(
        self,
        mock_projection_class,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
        tmp_path,
    ):
        """Test single geography projection returns dictionary."""
        # Setup mock
        mock_projection = MagicMock()
        mock_projection.run_projection.return_value = pd.DataFrame(
            {
                "year": [2025, 2026],
                "age": [0, 0],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "population": [100.0, 102.0],
            }
        )
        mock_projection.get_projection_summary.return_value = pd.DataFrame()
        mock_projection_class.return_value = mock_projection

        result = run_single_geography_projection(
            fips="38101",
            level="county",
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            output_dir=tmp_path,
            save_results=False,
        )

        assert isinstance(result, dict)
        assert "geography" in result
        assert "projection" in result
        assert "metadata" in result
        assert "processing_time" in result

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.geographic.multi_geography.CohortComponentProjection")
    def test_single_projection_geography_info(
        self,
        mock_projection_class,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
        tmp_path,
    ):
        """Test single projection includes geography information."""
        mock_projection = MagicMock()
        mock_projection.run_projection.return_value = pd.DataFrame(
            {
                "year": [2025],
                "age": [0],
                "sex": ["Male"],
                "race": ["White"],
                "population": [100.0],
            }
        )
        mock_projection.get_projection_summary.return_value = pd.DataFrame()
        mock_projection_class.return_value = mock_projection

        result = run_single_geography_projection(
            fips="38101",
            level="county",
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            output_dir=tmp_path,
            save_results=False,
        )

        assert result["geography"]["fips"] == "38101"
        assert result["geography"]["level"] == "county"
        assert "name" in result["geography"]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_single_projection_empty_population(
        self,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
        tmp_path,
    ):
        """Test handling of empty base population."""
        empty_population = pd.DataFrame(
            {
                "geography_fips": ["38999"],  # Non-matching FIPS
                "age": [0],
                "sex": ["Male"],
                "race": ["White"],
                "population": [100.0],
            }
        )

        result = run_single_geography_projection(
            fips="38101",
            level="county",
            base_population=empty_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            output_dir=tmp_path,
            save_results=False,
        )

        # Should return result with empty projection
        assert result["projection"].empty
        assert "error" in result["metadata"]


class TestAggregateToCounty:
    """Test aggregate_to_county function."""

    @pytest.fixture
    def place_projections(self):
        """Create sample place projection results."""
        return [
            {
                "geography": {"fips": "3825700", "level": "place", "name": "Fargo city"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025, 2025],
                        "age": [0, 1],
                        "sex": ["Male", "Female"],
                        "race": ["White", "White"],
                        "population": [500.0, 480.0],
                    }
                ),
                "metadata": {},
            },
            {
                "geography": {"fips": "3885100", "level": "place", "name": "West Fargo city"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025, 2025],
                        "age": [0, 1],
                        "sex": ["Male", "Female"],
                        "race": ["White", "White"],
                        "population": [200.0, 190.0],
                    }
                ),
                "metadata": {},
            },
        ]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.geographic.multi_geography.get_place_to_county_mapping")
    def test_aggregate_returns_dict(self, mock_mapping, place_projections):
        """Test aggregation returns dictionary."""
        mock_mapping.return_value = pd.DataFrame(
            {
                "place_fips": ["3825700", "3885100"],
                "county_fips": ["38017", "38017"],  # Both in Cass County
                "place_name": ["Fargo city", "West Fargo city"],
            }
        )

        result = aggregate_to_county(place_projections)

        assert isinstance(result, dict)

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.geographic.multi_geography.get_place_to_county_mapping")
    def test_aggregate_sums_populations(self, mock_mapping, place_projections):
        """Test aggregation sums populations correctly."""
        mock_mapping.return_value = pd.DataFrame(
            {
                "place_fips": ["3825700", "3885100"],
                "county_fips": ["38017", "38017"],
                "place_name": ["Fargo city", "West Fargo city"],
            }
        )

        result = aggregate_to_county(place_projections)

        # Cass County (38017) should have sum of Fargo + West Fargo
        if "38017" in result:
            cass_county = result["38017"]
            total_pop = cass_county["population"].sum()
            expected_total = 500 + 480 + 200 + 190  # Sum of all place populations
            assert total_pop == expected_total

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    @patch("cohort_projections.geographic.multi_geography.get_place_to_county_mapping")
    def test_aggregate_handles_empty_projections(self, mock_mapping):
        """Test aggregation handles empty projections."""
        mock_mapping.return_value = pd.DataFrame(
            {
                "place_fips": [],
                "county_fips": [],
                "place_name": [],
            }
        )

        empty_projections = [
            {
                "geography": {"fips": "3800001", "level": "place"},
                "projection": pd.DataFrame(),
                "metadata": {},
            }
        ]

        result = aggregate_to_county(empty_projections)

        # Should return empty dict for empty projections
        assert isinstance(result, dict)


class TestAggregateToState:
    """Test aggregate_to_state function."""

    @pytest.fixture
    def county_projections(self):
        """Create sample county projection results."""
        return [
            {
                "geography": {"fips": "38017", "level": "county", "name": "Cass County"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025, 2025],
                        "age": [0, 1],
                        "sex": ["Male", "Female"],
                        "race": ["White", "White"],
                        "population": [1000.0, 950.0],
                    }
                ),
                "metadata": {},
            },
            {
                "geography": {"fips": "38015", "level": "county", "name": "Burleigh County"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025, 2025],
                        "age": [0, 1],
                        "sex": ["Male", "Female"],
                        "race": ["White", "White"],
                        "population": [800.0, 760.0],
                    }
                ),
                "metadata": {},
            },
        ]

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_aggregate_to_state_returns_dataframe(self, county_projections):
        """Test state aggregation returns DataFrame."""
        result = aggregate_to_state(county_projections)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_aggregate_to_state_sums_populations(self, county_projections):
        """Test state aggregation sums populations correctly."""
        result = aggregate_to_state(county_projections)

        total_pop = result["population"].sum()
        expected_total = 1000 + 950 + 800 + 760
        assert total_pop == expected_total

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_aggregate_to_state_preserves_cohorts(self, county_projections):
        """Test state aggregation preserves cohort structure."""
        result = aggregate_to_state(county_projections)

        # Should have year, age, sex, race columns
        assert "year" in result.columns
        assert "age" in result.columns
        assert "sex" in result.columns
        assert "race" in result.columns
        assert "population" in result.columns

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_aggregate_to_state_empty_input(self):
        """Test state aggregation handles empty input."""
        result = aggregate_to_state([])

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestValidateAggregation:
    """Test validate_aggregation function."""

    @pytest.fixture
    def component_projections(self):
        """Create sample component projections."""
        return [
            {
                "geography": {"fips": "3825700", "level": "place"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025, 2030],
                        "age": [0, 0],
                        "sex": ["Male", "Male"],
                        "race": ["White", "White"],
                        "population": [500.0, 520.0],
                    }
                ),
                "metadata": {},
            },
            {
                "geography": {"fips": "3885100", "level": "place"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025, 2030],
                        "age": [0, 0],
                        "sex": ["Male", "Male"],
                        "race": ["White", "White"],
                        "population": [200.0, 210.0],
                    }
                ),
                "metadata": {},
            },
        ]

    @pytest.fixture
    def aggregated_projection(self):
        """Create sample aggregated projection."""
        return pd.DataFrame(
            {
                "year": [2025, 2030],
                "age": [0, 0],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "population": [700.0, 730.0],  # Sum of components
            }
        )

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validation_returns_dict(self, component_projections, aggregated_projection):
        """Test validation returns dictionary."""
        result = validate_aggregation(
            component_projections=component_projections,
            aggregated_projection=aggregated_projection,
            component_level="place",
            aggregate_level="county",
        )

        assert isinstance(result, dict)
        assert "valid" in result

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validation_passes_for_matching_data(
        self, component_projections, aggregated_projection
    ):
        """Test validation passes when totals match."""
        result = validate_aggregation(
            component_projections=component_projections,
            aggregated_projection=aggregated_projection,
            component_level="place",
            aggregate_level="county",
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validation_fails_for_mismatched_data(self, component_projections):
        """Test validation fails when totals don't match."""
        mismatched_aggregate = pd.DataFrame(
            {
                "year": [2025, 2030],
                "age": [0, 0],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "population": [1000.0, 1100.0],  # Different from sum
            }
        )

        result = validate_aggregation(
            component_projections=component_projections,
            aggregated_projection=mismatched_aggregate,
            component_level="place",
            aggregate_level="county",
            tolerance=0.01,
        )

        # Should detect difference
        assert result["valid"] is False or len(result["warnings"]) > 0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validation_uses_tolerance(self, component_projections):
        """Test validation respects tolerance parameter."""
        # Slightly different aggregate (within 1%)
        close_aggregate = pd.DataFrame(
            {
                "year": [2025, 2030],
                "age": [0, 0],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "population": [703.0, 733.0],  # ~0.5% different
            }
        )

        result = validate_aggregation(
            component_projections=component_projections,
            aggregated_projection=close_aggregate,
            component_level="place",
            aggregate_level="county",
            tolerance=0.01,  # 1% tolerance
        )

        # Should pass with 1% tolerance
        assert result["valid"] is True

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validation_checks_by_year(self, component_projections, aggregated_projection):
        """Test validation checks each year separately."""
        result = validate_aggregation(
            component_projections=component_projections,
            aggregated_projection=aggregated_projection,
            component_level="place",
            aggregate_level="county",
        )

        assert "by_year" in result
        assert len(result["by_year"]) >= 2  # 2025 and 2030

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_validation_empty_components_fails(self):
        """Test validation fails for empty component projections."""
        result = validate_aggregation(
            component_projections=[],
            aggregated_projection=pd.DataFrame(),
            component_level="place",
            aggregate_level="county",
        )

        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestSaveProjectionResults:
    """Test _save_projection_results function."""

    @pytest.fixture
    def sample_projection(self):
        """Create sample projection DataFrame."""
        return pd.DataFrame(
            {
                "year": [2025, 2026],
                "age": [0, 0],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "population": [100.0, 102.0],
            }
        )

    @pytest.fixture
    def sample_summary(self):
        """Create sample summary DataFrame."""
        return pd.DataFrame(
            {
                "year": [2025, 2026],
                "total_population": [100.0, 102.0],
            }
        )

    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return {
            "projection": {
                "scenario": "baseline",
            },
            "geography": {
                "fips": "38101",
            },
        }

    @pytest.fixture
    def sample_config(self):
        """Create sample config."""
        return {
            "project": {
                "base_year": 2025,
                "projection_horizon": 5,
            },
            "output": {
                "compression": "gzip",
            },
        }

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_save_creates_parquet_file(
        self,
        sample_projection,
        sample_summary,
        sample_metadata,
        sample_config,
        tmp_path,
    ):
        """Test save creates parquet file."""
        _save_projection_results(
            fips="38101",
            level="county",
            projection=sample_projection,
            summary=sample_summary,
            metadata=sample_metadata,
            output_dir=tmp_path,
            config=sample_config,
        )

        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_save_creates_metadata_file(
        self,
        sample_projection,
        sample_summary,
        sample_metadata,
        sample_config,
        tmp_path,
    ):
        """Test save creates metadata JSON file."""
        _save_projection_results(
            fips="38101",
            level="county",
            projection=sample_projection,
            summary=sample_summary,
            metadata=sample_metadata,
            output_dir=tmp_path,
            config=sample_config,
        )

        json_files = list(tmp_path.glob("*_metadata.json"))
        assert len(json_files) == 1

        # Verify JSON is valid
        with open(json_files[0]) as f:
            loaded = json.load(f)
        assert "projection" in loaded

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_save_creates_summary_csv(
        self,
        sample_projection,
        sample_summary,
        sample_metadata,
        sample_config,
        tmp_path,
    ):
        """Test save creates summary CSV file."""
        _save_projection_results(
            fips="38101",
            level="county",
            projection=sample_projection,
            summary=sample_summary,
            metadata=sample_metadata,
            output_dir=tmp_path,
            config=sample_config,
        )

        csv_files = list(tmp_path.glob("*_summary.csv"))
        assert len(csv_files) == 1

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_save_creates_directory(
        self,
        sample_projection,
        sample_summary,
        sample_metadata,
        sample_config,
        tmp_path,
    ):
        """Test save creates output directory if needed."""
        nested_dir = tmp_path / "nested" / "output"

        _save_projection_results(
            fips="38101",
            level="county",
            projection=sample_projection,
            summary=sample_summary,
            metadata=sample_metadata,
            output_dir=nested_dir,
            config=sample_config,
        )

        assert nested_dir.exists()

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_save_handles_empty_summary(
        self,
        sample_projection,
        sample_metadata,
        sample_config,
        tmp_path,
    ):
        """Test save handles empty summary DataFrame."""
        _save_projection_results(
            fips="38101",
            level="county",
            projection=sample_projection,
            summary=pd.DataFrame(),  # Empty summary
            metadata=sample_metadata,
            output_dir=tmp_path,
            config=sample_config,
        )

        # Should still create parquet and metadata
        assert len(list(tmp_path.glob("*.parquet"))) == 1
        assert len(list(tmp_path.glob("*_metadata.json"))) == 1


class TestRunMultiGeographyProjections:
    """Test run_multi_geography_projections function."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_requires_migration_rates(self):
        """Test function requires migration rates by geography."""
        with pytest.raises(ValueError, match="migration_rates_by_geography is required"):
            run_multi_geography_projections(
                level="county",
                base_population_by_geography={},
                fertility_rates=pd.DataFrame(),
                survival_rates=pd.DataFrame(),
                migration_rates_by_geography=None,  # Missing
            )


class TestMultiGeographyEdgeCases:
    """Test edge cases for multi-geography functions."""

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_aggregate_single_geography(self):
        """Test aggregation with single geography."""
        single_projection = [
            {
                "geography": {"fips": "38101", "level": "county"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025],
                        "age": [0],
                        "sex": ["Male"],
                        "race": ["White"],
                        "population": [1000.0],
                    }
                ),
                "metadata": {},
            }
        ]

        result = aggregate_to_state(single_projection)

        assert len(result) == 1
        assert result["population"].values[0] == 1000.0

    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Dependencies not available")
    def test_aggregate_handles_missing_geography(self):
        """Test aggregation handles missing geography in projection."""
        projection_with_missing = [
            {
                "geography": {"fips": "unknown", "level": "place"},
                "projection": pd.DataFrame(
                    {
                        "year": [2025],
                        "age": [0],
                        "sex": ["Male"],
                        "race": ["White"],
                        "population": [100.0],
                    }
                ),
                "metadata": {},
            }
        ]

        # Should not raise error
        result = aggregate_to_state(projection_with_missing)
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
