"""
Unit tests for base_population.py data processing module.

Tests population harmonization, cohort matrix creation, and validation functions.
Uses synthetic DataFrames as fixtures - does not depend on actual data files.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cohort_projections.data.process.base_population import (
    RACE_ETHNICITY_MAP,
    create_cohort_matrix,
    get_cohort_summary,
    harmonize_race_categories,
    validate_cohort_matrix,
)


class TestHarmonizeRaceCategories:
    """Tests for harmonize_race_categories function."""

    @pytest.fixture
    def sample_raw_population_with_race(self):
        """Sample population data with various race column names."""
        return pd.DataFrame(
            {
                "age": [0, 1, 2, 3, 4],
                "sex": ["Male", "Male", "Female", "Female", "Male"],
                "race": ["WA_NH", "BA_NH", "IA_NH", "H", "AA_NH"],
                "population": [1000, 500, 200, 600, 400],
            }
        )

    @pytest.fixture
    def sample_raw_population_with_race_ethnicity(self):
        """Sample population data with race_ethnicity column."""
        return pd.DataFrame(
            {
                "age": [0, 1, 2],
                "sex": ["Male", "Female", "Male"],
                "race_ethnicity": ["WA_NH", "BA_NH", "H"],
                "population": [1000, 500, 600],
            }
        )

    @pytest.fixture
    def sample_raw_population_with_origin(self):
        """Sample population data with ORIGIN column."""
        return pd.DataFrame(
            {
                "age": [0, 1],
                "sex": ["Male", "Female"],
                "ORIGIN": ["NHWA", "HISP"],
                "population": [1000, 600],
            }
        )

    def test_harmonize_race_categories_basic(self, sample_raw_population_with_race):
        """Test basic race category harmonization."""
        result = harmonize_race_categories(sample_raw_population_with_race)

        assert "race_ethnicity" in result.columns
        assert "race" not in result.columns  # Original column should be dropped

        # Check mappings are correct
        expected_races = [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Hispanic (any race)",
            "Asian/PI alone, Non-Hispanic",
        ]
        assert list(result["race_ethnicity"]) == expected_races

    def test_harmonize_race_categories_with_race_ethnicity_column(
        self, sample_raw_population_with_race_ethnicity
    ):
        """Test harmonization when column is already named race_ethnicity."""
        result = harmonize_race_categories(sample_raw_population_with_race_ethnicity)

        assert "race_ethnicity" in result.columns
        assert len(result) == 3

    def test_harmonize_race_categories_with_origin_column(self, sample_raw_population_with_origin):
        """Test harmonization with ORIGIN column name."""
        result = harmonize_race_categories(sample_raw_population_with_origin)

        assert "race_ethnicity" in result.columns
        assert result["race_ethnicity"].iloc[0] == "White alone, Non-Hispanic"
        assert result["race_ethnicity"].iloc[1] == "Hispanic (any race)"

    def test_harmonize_race_categories_no_race_column(self):
        """Test that ValueError is raised when no race column found."""
        df = pd.DataFrame({"age": [0, 1], "sex": ["Male", "Female"], "population": [1000, 500]})

        with pytest.raises(ValueError, match="No race/ethnicity column found"):
            harmonize_race_categories(df)

    def test_harmonize_race_categories_unmapped_values(self):
        """Test handling of unmapped race categories."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2],
                "sex": ["Male", "Female", "Male"],
                "race": ["WA_NH", "UNKNOWN_RACE", "BA_NH"],
                "population": [1000, 500, 600],
            }
        )

        result = harmonize_race_categories(df)

        # Unmapped values should be dropped
        assert len(result) == 2
        assert "UNKNOWN_RACE" not in result["race_ethnicity"].values

    def test_harmonize_race_categories_preserves_original_data(
        self, sample_raw_population_with_race
    ):
        """Test that original DataFrame is not modified."""
        original = sample_raw_population_with_race.copy()
        harmonize_race_categories(sample_raw_population_with_race)

        pd.testing.assert_frame_equal(original, sample_raw_population_with_race)

    def test_harmonize_race_categories_all_mappings(self):
        """Test all defined race mappings are correctly applied."""
        # Create a DataFrame with all mapped race codes
        all_codes = list(RACE_ETHNICITY_MAP.keys())
        df = pd.DataFrame(
            {
                "age": [25] * len(all_codes),
                "sex": ["Male"] * len(all_codes),
                "race": all_codes,
                "population": [100] * len(all_codes),
            }
        )

        result = harmonize_race_categories(df)

        # All should be successfully mapped
        assert len(result) == len(all_codes)
        assert not result["race_ethnicity"].isna().any()


class TestCreateCohortMatrix:
    """Tests for create_cohort_matrix function."""

    @pytest.fixture
    def sample_harmonized_population(self):
        """Sample harmonized population data for cohort matrix creation."""
        return pd.DataFrame(
            {
                "age": [25, 25, 30, 30],
                "sex": ["male", "female", "male", "female"],
                "race_ethnicity": [
                    "White alone, Non-Hispanic",
                    "White alone, Non-Hispanic",
                    "Hispanic (any race)",
                    "Hispanic (any race)",
                ],
                "population": [1000, 1100, 500, 550],
            }
        )

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = MagicMock()
        config.get_parameter.side_effect = lambda *args, **kwargs: {
            ("demographics",): {
                "sex": ["Male", "Female"],
                "race_ethnicity": {
                    "categories": [
                        "White alone, Non-Hispanic",
                        "Black alone, Non-Hispanic",
                        "AIAN alone, Non-Hispanic",
                        "Asian/PI alone, Non-Hispanic",
                        "Two or more races, Non-Hispanic",
                        "Hispanic (any race)",
                    ]
                },
                "age_groups": {"min_age": 0, "max_age": 90},
            },
            ("project", "base_year"): 2020,
        }.get(args, kwargs.get("default"))
        return config

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_create_cohort_matrix_basic(self, mock_config_loader, sample_harmonized_population):
        """Test basic cohort matrix creation."""
        mock_config = MagicMock()
        demographics_config = {
            "sex": ["Male", "Female"],
            "race_ethnicity": {
                "categories": [
                    "White alone, Non-Hispanic",
                    "Black alone, Non-Hispanic",
                    "AIAN alone, Non-Hispanic",
                    "Asian/PI alone, Non-Hispanic",
                    "Two or more races, Non-Hispanic",
                    "Hispanic (any race)",
                ]
            },
            "age_groups": {"min_age": 0, "max_age": 90},
        }
        mock_config.get_parameter.side_effect = lambda *args, **kwargs: {
            ("demographics",): demographics_config,
            ("project", "base_year"): 2020,
        }.get(args, kwargs.get("default"))
        mock_config_loader.return_value = mock_config

        result = create_cohort_matrix(
            sample_harmonized_population, geography_level="state", geography_id="38"
        )

        assert "geography_level" in result.columns
        assert "geography_id" in result.columns
        assert result["geography_level"].iloc[0] == "state"
        assert result["geography_id"].iloc[0] == "38"

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_create_cohort_matrix_sex_standardization(
        self, mock_config_loader, sample_harmonized_population
    ):
        """Test that sex values are standardized to Title case."""
        mock_config = MagicMock()
        demographics_config = {
            "sex": ["Male", "Female"],
            "race_ethnicity": {
                "categories": [
                    "White alone, Non-Hispanic",
                    "Black alone, Non-Hispanic",
                    "AIAN alone, Non-Hispanic",
                    "Asian/PI alone, Non-Hispanic",
                    "Two or more races, Non-Hispanic",
                    "Hispanic (any race)",
                ]
            },
            "age_groups": {"min_age": 0, "max_age": 90},
        }
        mock_config.get_parameter.side_effect = lambda *args, **kwargs: {
            ("demographics",): demographics_config,
            ("project", "base_year"): 2020,
        }.get(args, kwargs.get("default"))
        mock_config_loader.return_value = mock_config

        result = create_cohort_matrix(sample_harmonized_population, geography_level="state")

        # All sex values should be Title case
        assert set(result["sex"].unique()) <= {"Male", "Female"}

    def test_create_cohort_matrix_missing_columns(self):
        """Test that ValueError is raised for missing required columns."""
        df = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["Male", "Female"],
                # Missing 'race_ethnicity' and 'population'
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            create_cohort_matrix(df, geography_level="state")

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_create_cohort_matrix_fills_missing_cohorts(self, mock_config_loader):
        """Test that missing age-sex-race combinations are filled with 0."""
        mock_config = MagicMock()
        demographics_config = {
            "sex": ["Male", "Female"],
            "race_ethnicity": {
                "categories": ["White alone, Non-Hispanic", "Black alone, Non-Hispanic"]
            },
            "age_groups": {"min_age": 0, "max_age": 2},  # Small range for testing
        }
        mock_config.get_parameter.side_effect = lambda *args, **kwargs: {
            ("demographics",): demographics_config,
            ("project", "base_year"): 2020,
        }.get(args, kwargs.get("default"))
        mock_config_loader.return_value = mock_config

        # Only provide partial data
        df = pd.DataFrame(
            {
                "age": [0],
                "sex": ["Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                "population": [1000],
            }
        )

        result = create_cohort_matrix(df, geography_level="state")

        # Should have all combinations (3 ages * 2 sexes * 2 races = 12)
        expected_rows = 3 * 2 * 2
        assert len(result) == expected_rows

        # Missing combinations should be filled with 0
        assert (result["population"] == 0).sum() > 0


class TestValidateCohortMatrix:
    """Tests for validate_cohort_matrix function."""

    @pytest.fixture
    def valid_state_cohort_matrix(self):
        """Create a valid state-level cohort matrix."""
        ages = list(range(91))
        sexes = ["Male", "Female"]
        races = [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ]

        records = []
        for age in ages:
            for sex in sexes:
                for race in races:
                    records.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race_ethnicity": race,
                            "population": 100,  # Uniform population for testing
                        }
                    )

        return pd.DataFrame(records)

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_validate_cohort_matrix_valid_state(
        self, mock_config_loader, valid_state_cohort_matrix
    ):
        """Test validation of valid state cohort matrix."""
        mock_config = MagicMock()
        mock_config.get_parameter.return_value = {
            "age_groups": {"min_age": 0, "max_age": 90},
            "sex": ["Male", "Female"],
            "race_ethnicity": {
                "categories": [
                    "White alone, Non-Hispanic",
                    "Black alone, Non-Hispanic",
                    "AIAN alone, Non-Hispanic",
                    "Asian/PI alone, Non-Hispanic",
                    "Two or more races, Non-Hispanic",
                    "Hispanic (any race)",
                ]
            },
        }
        mock_config_loader.return_value = mock_config

        result = validate_cohort_matrix(valid_state_cohort_matrix, "state")

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_validate_cohort_matrix_negative_population(self, mock_config_loader):
        """Test that negative populations are flagged as errors."""
        mock_config = MagicMock()
        mock_config.get_parameter.return_value = {
            "age_groups": {"min_age": 0, "max_age": 2},
            "sex": ["Male", "Female"],
            "race_ethnicity": {"categories": ["White alone, Non-Hispanic"]},
        }
        mock_config_loader.return_value = mock_config

        df = pd.DataFrame(
            {
                "age": [0, 1, 2, 0, 1, 2],
                "sex": ["Male", "Male", "Male", "Female", "Female", "Female"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 6,
                "population": [100, -50, 200, 150, 100, 200],  # Negative value
            }
        )

        result = validate_cohort_matrix(df, "state")

        assert result["valid"] is False
        assert any("Negative" in error for error in result["errors"])

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_validate_cohort_matrix_zero_total_population(self, mock_config_loader):
        """Test that zero total population is flagged as error."""
        mock_config = MagicMock()
        mock_config.get_parameter.return_value = {
            "age_groups": {"min_age": 0, "max_age": 1},
            "sex": ["Male", "Female"],
            "race_ethnicity": {"categories": ["White alone, Non-Hispanic"]},
        }
        mock_config_loader.return_value = mock_config

        df = pd.DataFrame(
            {
                "age": [0, 1, 0, 1],
                "sex": ["Male", "Male", "Female", "Female"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 4,
                "population": [0, 0, 0, 0],  # All zero
            }
        )

        result = validate_cohort_matrix(df, "state")

        assert result["valid"] is False
        assert any("zero" in error.lower() for error in result["errors"])

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_validate_cohort_matrix_unusual_sex_ratio_warning(self, mock_config_loader):
        """Test that unusual sex ratios generate warnings."""
        mock_config = MagicMock()
        mock_config.get_parameter.return_value = {
            "age_groups": {"min_age": 0, "max_age": 0},
            "sex": ["Male", "Female"],
            "race_ethnicity": {"categories": ["White alone, Non-Hispanic"]},
        }
        mock_config_loader.return_value = mock_config

        df = pd.DataFrame(
            {
                "age": [0, 0],
                "sex": ["Male", "Female"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2,
                "population": [300, 100],  # 3:1 ratio - unusual
            }
        )

        result = validate_cohort_matrix(df, "state")

        # Should have warnings about sex ratio
        assert len(result["warnings"]) > 0

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_validate_cohort_matrix_county_incomplete(self, mock_config_loader):
        """Test validation of county data with incomplete cohorts."""
        mock_config = MagicMock()
        mock_config.get_parameter.return_value = {
            "age_groups": {"min_age": 0, "max_age": 1},
            "sex": ["Male", "Female"],
            "race_ethnicity": {"categories": ["White alone, Non-Hispanic"]},
        }
        mock_config_loader.return_value = mock_config

        # Two counties, one with complete data, one incomplete
        df = pd.DataFrame(
            {
                "geography_id": ["38001", "38001", "38001", "38001", "38003"],
                "age": [0, 1, 0, 1, 0],
                "sex": ["Male", "Male", "Female", "Female", "Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 5,
                "population": [100, 100, 100, 100, 100],
            }
        )

        result = validate_cohort_matrix(df, "county", expected_counties=2)

        assert result["valid"] is False


class TestGetCohortSummary:
    """Tests for get_cohort_summary function."""

    @pytest.fixture
    def sample_cohort_matrix(self):
        """Sample cohort matrix for summary testing."""
        return pd.DataFrame(
            {
                "age": [5, 25, 70, 5, 25, 70],
                "sex": ["Male", "Male", "Male", "Female", "Female", "Female"],
                "race_ethnicity": [
                    "White alone, Non-Hispanic",
                    "White alone, Non-Hispanic",
                    "White alone, Non-Hispanic",
                    "Hispanic (any race)",
                    "Hispanic (any race)",
                    "Hispanic (any race)",
                ],
                "population": [1000, 2000, 500, 1100, 2200, 600],
            }
        )

    def test_get_cohort_summary_total(self, sample_cohort_matrix):
        """Test that summary includes correct total."""
        result = get_cohort_summary(sample_cohort_matrix)

        total_row = result[result["group"] == "All"]
        assert len(total_row) == 1
        assert total_row["population"].iloc[0] == 7400

    def test_get_cohort_summary_by_sex(self, sample_cohort_matrix):
        """Test summary breakdown by sex."""
        result = get_cohort_summary(sample_cohort_matrix)

        male_pop = result[(result["category"] == "Sex") & (result["group"] == "Male")][
            "population"
        ].iloc[0]
        female_pop = result[(result["category"] == "Sex") & (result["group"] == "Female")][
            "population"
        ].iloc[0]

        assert male_pop == 3500
        assert female_pop == 3900

    def test_get_cohort_summary_by_race(self, sample_cohort_matrix):
        """Test summary breakdown by race/ethnicity."""
        result = get_cohort_summary(sample_cohort_matrix)

        race_results = result[result["category"] == "Race/Ethnicity"]
        assert len(race_results) == 2

    def test_get_cohort_summary_by_age_group(self, sample_cohort_matrix):
        """Test summary breakdown by age groups."""
        result = get_cohort_summary(sample_cohort_matrix)

        age_results = result[result["category"] == "Age Group"]
        assert len(age_results) == 3  # 0-17, 18-64, 65+

        # Check 0-17 group (only age 5 qualifies)
        under18 = result[(result["category"] == "Age Group") & (result["group"] == "0-17")]
        assert under18["population"].iloc[0] == 2100  # 1000 + 1100

    def test_get_cohort_summary_percentages(self, sample_cohort_matrix):
        """Test that percentages are calculated correctly."""
        result = get_cohort_summary(sample_cohort_matrix)

        # Percentages should sum to 100 within each category
        assert "percentage" in result.columns

        # Total should be 100%
        total_pct = result[result["group"] == "All"]["percentage"].iloc[0]
        assert abs(total_pct - 100.0) < 0.01


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_harmonize_empty_dataframe(self):
        """Test harmonization of empty DataFrame."""
        df = pd.DataFrame({"age": [], "sex": [], "race": [], "population": []})

        result = harmonize_race_categories(df)
        assert len(result) == 0

    def test_harmonize_all_unmapped_categories(self):
        """Test when all categories are unmapped."""
        df = pd.DataFrame(
            {
                "age": [0, 1],
                "sex": ["Male", "Female"],
                "race": ["INVALID1", "INVALID2"],
                "population": [100, 100],
            }
        )

        result = harmonize_race_categories(df)
        assert len(result) == 0

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_create_cohort_matrix_age_capping(self, mock_config_loader):
        """Test that ages above max_age are capped."""
        mock_config = MagicMock()
        demographics_config = {
            "sex": ["Male"],
            "race_ethnicity": {"categories": ["White alone, Non-Hispanic"]},
            "age_groups": {"min_age": 0, "max_age": 90},
        }
        mock_config.get_parameter.side_effect = lambda *args, **kwargs: {
            ("demographics",): demographics_config,
            ("project", "base_year"): 2020,
        }.get(args, kwargs.get("default"))
        mock_config_loader.return_value = mock_config

        df = pd.DataFrame(
            {
                "age": [95, 100, 110],  # Ages above 90
                "sex": ["Male", "Male", "Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "population": [10, 5, 2],
            }
        )

        result = create_cohort_matrix(df, geography_level="state")

        # All should be grouped into age 90
        age_90_pop = result[result["age"] == 90]["population"].sum()
        assert age_90_pop == 17  # 10 + 5 + 2

    @patch("cohort_projections.data.process.base_population.ConfigLoader")
    def test_create_cohort_matrix_aggregates_duplicates(self, mock_config_loader):
        """Test that duplicate cohorts are aggregated correctly."""
        mock_config = MagicMock()
        demographics_config = {
            "sex": ["Male"],
            "race_ethnicity": {"categories": ["White alone, Non-Hispanic"]},
            "age_groups": {"min_age": 0, "max_age": 2},
        }
        mock_config.get_parameter.side_effect = lambda *args, **kwargs: {
            ("demographics",): demographics_config,
            ("project", "base_year"): 2020,
        }.get(args, kwargs.get("default"))
        mock_config_loader.return_value = mock_config

        # Duplicate age-sex-race combinations
        df = pd.DataFrame(
            {
                "age": [0, 0, 0],
                "sex": ["Male", "Male", "Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "population": [100, 200, 300],
            }
        )

        result = create_cohort_matrix(df, geography_level="state")

        # Should be summed
        age_0_male_white = result[
            (result["age"] == 0)
            & (result["sex"] == "Male")
            & (result["race_ethnicity"] == "White alone, Non-Hispanic")
        ]["population"].iloc[0]

        assert age_0_male_white == 600


class TestRaceEthnicityMap:
    """Tests for the RACE_ETHNICITY_MAP constant."""

    def test_all_expected_categories_mapped(self):
        """Test that all expected output categories are represented."""
        expected_outputs = {
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        }

        actual_outputs = set(RACE_ETHNICITY_MAP.values())

        assert expected_outputs == actual_outputs

    def test_map_values_are_valid_strings(self):
        """Test that all map values are non-empty strings."""
        for key, value in RACE_ETHNICITY_MAP.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(key) > 0
            assert len(value) > 0
