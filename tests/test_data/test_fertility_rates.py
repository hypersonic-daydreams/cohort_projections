"""
Unit tests for fertility_rates.py data processing module.

Tests ASFR processing, rate validation, and fertility rate table creation.
Uses synthetic DataFrames as fixtures - does not depend on actual data files.
"""

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.process.fertility_rates import (
    SEER_RACE_ETHNICITY_MAP,
    calculate_average_fertility_rates,
    create_fertility_rate_table,
    harmonize_fertility_race_categories,
    load_seer_fertility_data,
    validate_fertility_rates,
)


class TestLoadSeerFertilityData:
    """Tests for load_seer_fertility_data function."""

    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Create a temporary CSV file with sample fertility data."""
        data = pd.DataFrame(
            {
                "year": [2020, 2020, 2021, 2021],
                "age": [20, 25, 20, 25],
                "race": ["White NH", "White NH", "Hispanic", "Hispanic"],
                "fertility_rate": [0.05, 0.08, 0.06, 0.09],
                "population": [1000, 1200, 800, 900],
            }
        )
        file_path = tmp_path / "fertility_data.csv"
        data.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def sample_txt_file(self, tmp_path):
        """Create a temporary tab-delimited file with sample fertility data."""
        data = pd.DataFrame(
            {
                "year": [2020, 2021],
                "age": [20, 25],
                "race": ["White NH", "Hispanic"],
                "fertility_rate": [0.05, 0.09],
            }
        )
        file_path = tmp_path / "fertility_data.txt"
        data.to_csv(file_path, sep="\t", index=False)
        return file_path

    def test_load_seer_fertility_data_csv(self, sample_csv_file):
        """Test loading fertility data from CSV file."""
        result = load_seer_fertility_data(sample_csv_file)

        assert len(result) == 4
        assert "age" in result.columns
        assert "fertility_rate" in result.columns

    def test_load_seer_fertility_data_txt(self, sample_txt_file):
        """Test loading fertility data from tab-delimited file."""
        result = load_seer_fertility_data(sample_txt_file)

        assert len(result) == 2
        assert "age" in result.columns

    def test_load_seer_fertility_data_with_year_filter(self, sample_csv_file):
        """Test filtering data by year range."""
        result = load_seer_fertility_data(sample_csv_file, year_range=(2020, 2020))

        assert len(result) == 2
        assert (result["year"] == 2020).all()

    def test_load_seer_fertility_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_seer_fertility_data("/nonexistent/path/data.csv")

    def test_load_seer_fertility_data_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported format."""
        file_path = tmp_path / "data.xyz"
        file_path.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_seer_fertility_data(file_path)

    def test_load_seer_fertility_data_standardizes_column_names(self, sample_csv_file):
        """Test that column names are standardized to lowercase."""
        result = load_seer_fertility_data(sample_csv_file)

        # All column names should be lowercase
        assert all(col == col.lower() for col in result.columns)

    def test_load_seer_fertility_data_age_of_mother_column(self, tmp_path):
        """Test handling of 'age_of_mother' column name."""
        data = pd.DataFrame(
            {
                "year": [2020],
                "age_of_mother": [25],  # Alternative column name
                "race": ["White NH"],
                "fertility_rate": [0.08],
            }
        )
        file_path = tmp_path / "fertility_data.csv"
        data.to_csv(file_path, index=False)

        result = load_seer_fertility_data(file_path)

        assert "age" in result.columns

    def test_load_seer_fertility_data_no_age_column(self, tmp_path):
        """Test that ValueError is raised when no age column found."""
        data = pd.DataFrame({"year": [2020], "race": ["White NH"], "fertility_rate": [0.08]})
        file_path = tmp_path / "fertility_data.csv"
        data.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="No age column found"):
            load_seer_fertility_data(file_path)


class TestHarmonizeFertilityRaceCategories:
    """Tests for harmonize_fertility_race_categories function."""

    @pytest.fixture
    def sample_fertility_data(self):
        """Sample fertility data with SEER race codes."""
        return pd.DataFrame(
            {
                "age": [20, 25, 30],
                "race": ["White NH", "Hispanic", "Black NH"],
                "fertility_rate": [0.05, 0.07, 0.06],
            }
        )

    def test_harmonize_fertility_race_categories_basic(self, sample_fertility_data):
        """Test basic race category harmonization."""
        result = harmonize_fertility_race_categories(sample_fertility_data)

        assert "race_ethnicity" in result.columns
        assert "race" not in result.columns

        expected = ["White alone, Non-Hispanic", "Hispanic (any race)", "Black alone, Non-Hispanic"]
        assert list(result["race_ethnicity"]) == expected

    def test_harmonize_fertility_race_categories_numeric_codes(self):
        """Test harmonization of numeric SEER codes."""
        df = pd.DataFrame(
            {
                "age": [20, 25],
                "race": ["1", "6"],  # Numeric codes
                "fertility_rate": [0.05, 0.07],
            }
        )

        result = harmonize_fertility_race_categories(df)

        assert result["race_ethnicity"].iloc[0] == "White alone, Non-Hispanic"
        assert result["race_ethnicity"].iloc[1] == "Hispanic (any race)"

    def test_harmonize_fertility_race_categories_no_race_column(self):
        """Test that ValueError is raised when no race column found."""
        df = pd.DataFrame({"age": [20], "fertility_rate": [0.05]})

        with pytest.raises(ValueError, match="No race/ethnicity column found"):
            harmonize_fertility_race_categories(df)

    def test_harmonize_fertility_race_categories_unmapped_dropped(self):
        """Test that unmapped race categories are dropped."""
        df = pd.DataFrame(
            {
                "age": [20, 25, 30],
                "race": ["White NH", "UNKNOWN", "Hispanic"],
                "fertility_rate": [0.05, 0.06, 0.07],
            }
        )

        result = harmonize_fertility_race_categories(df)

        assert len(result) == 2
        assert "UNKNOWN" not in result["race_ethnicity"].values

    def test_harmonize_fertility_race_categories_preserves_data(self, sample_fertility_data):
        """Test that fertility rate data is preserved."""
        result = harmonize_fertility_race_categories(sample_fertility_data)

        assert "fertility_rate" in result.columns
        assert list(result["fertility_rate"]) == [0.05, 0.07, 0.06]


class TestCalculateAverageFertilityRates:
    """Tests for calculate_average_fertility_rates function."""

    @pytest.fixture
    def multi_year_fertility_data(self):
        """Sample multi-year fertility data for averaging."""
        return pd.DataFrame(
            {
                "year": [2018, 2019, 2020, 2018, 2019, 2020],
                "age": [25, 25, 25, 30, 30, 30],
                "race_ethnicity": ["Hispanic (any race)"] * 6,
                "fertility_rate": [0.09, 0.088, 0.092, 0.07, 0.068, 0.072],
                "population": [1000, 1100, 1050, 800, 850, 825],
            }
        )

    @pytest.fixture
    def multi_year_no_weights(self):
        """Sample data without population weights."""
        return pd.DataFrame(
            {
                "year": [2018, 2019, 2020],
                "age": [25, 25, 25],
                "race_ethnicity": ["Hispanic (any race)"] * 3,
                "fertility_rate": [0.09, 0.10, 0.11],
            }
        )

    def test_calculate_average_rates_simple_mean(self, multi_year_no_weights):
        """Test simple mean calculation when no weights available."""
        result = calculate_average_fertility_rates(multi_year_no_weights)

        assert len(result) == 1
        expected_rate = (0.09 + 0.10 + 0.11) / 3
        assert abs(result["fertility_rate"].iloc[0] - expected_rate) < 0.0001

    def test_calculate_average_rates_weighted(self, multi_year_fertility_data):
        """Test weighted average calculation."""
        # Need to add births column for weighted average
        df = multi_year_fertility_data.copy()
        df["births"] = df["fertility_rate"] * df["population"]

        result = calculate_average_fertility_rates(df)

        assert "fertility_rate" in result.columns
        assert "age" in result.columns
        assert "race_ethnicity" in result.columns

    def test_calculate_average_rates_multiple_cohorts(self):
        """Test averaging with multiple age-race combinations."""
        df = pd.DataFrame(
            {
                "year": [2018, 2019, 2018, 2019],
                "age": [25, 25, 30, 30],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2 + ["Hispanic (any race)"] * 2,
                "fertility_rate": [0.08, 0.09, 0.07, 0.075],
            }
        )

        result = calculate_average_fertility_rates(df)

        assert len(result) == 2
        assert set(result["race_ethnicity"]) == {"White alone, Non-Hispanic", "Hispanic (any race)"}

    def test_calculate_average_rates_missing_columns(self):
        """Test that ValueError is raised for missing columns."""
        df = pd.DataFrame(
            {
                "age": [25],
                "year": [2020],
                # Missing fertility_rate and race_ethnicity
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_average_fertility_rates(df)

    def test_calculate_average_rates_nan_handling(self):
        """Test that NaN values in averaged rates are set to 0."""
        df = pd.DataFrame(
            {
                "age": [25, 25],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2,
                "fertility_rate": [np.nan, np.nan],
            }
        )

        result = calculate_average_fertility_rates(df)

        assert not result["fertility_rate"].isna().any()
        assert result["fertility_rate"].iloc[0] == 0.0


class TestCreateFertilityRateTable:
    """Tests for create_fertility_rate_table function."""

    @pytest.fixture
    def sample_averaged_rates(self):
        """Sample averaged fertility rates."""
        return pd.DataFrame(
            {
                "age": [20, 25, 30],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "fertility_rate": [0.05, 0.08, 0.07],
            }
        )

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "demographics": {
                "race_ethnicity": {
                    "categories": [
                        "White alone, Non-Hispanic",
                        "Black alone, Non-Hispanic",
                        "AIAN alone, Non-Hispanic",
                        "Asian/PI alone, Non-Hispanic",
                        "Two or more races, Non-Hispanic",
                        "Hispanic (any race)",
                    ]
                }
            },
            "rates": {"fertility": {"apply_to_ages": [15, 49]}},
        }

    def test_create_fertility_rate_table_fills_missing(self, sample_averaged_rates, mock_config):
        """Test that missing age-race combinations are filled with 0."""
        result = create_fertility_rate_table(
            sample_averaged_rates, validate=False, config=mock_config
        )

        # Should have 35 ages (15-49) * 6 races = 210 rows
        expected_rows = 35 * 6
        assert len(result) == expected_rows

    def test_create_fertility_rate_table_filters_reproductive_ages(self, mock_config):
        """Test that only reproductive ages (15-49) are included."""
        df = pd.DataFrame(
            {
                "age": [10, 15, 30, 50, 60],  # Some outside reproductive range
                "race_ethnicity": ["White alone, Non-Hispanic"] * 5,
                "fertility_rate": [0.01, 0.05, 0.08, 0.01, 0.001],
            }
        )

        result = create_fertility_rate_table(df, validate=False, config=mock_config)

        assert result["age"].min() == 15
        assert result["age"].max() == 49

    def test_create_fertility_rate_table_negative_rates_clipped(self, mock_config):
        """Test that negative fertility rates are clipped to 0."""
        df = pd.DataFrame(
            {
                "age": [20, 25],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2,
                "fertility_rate": [-0.05, 0.08],  # Negative value
            }
        )

        result = create_fertility_rate_table(df, validate=False, config=mock_config)

        assert (result["fertility_rate"] >= 0).all()

    def test_create_fertility_rate_table_missing_columns(self, mock_config):
        """Test that ValueError is raised for missing columns."""
        df = pd.DataFrame(
            {
                "age": [20],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                # Missing fertility_rate
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            create_fertility_rate_table(df, validate=False, config=mock_config)

    def test_create_fertility_rate_table_adds_metadata(self, sample_averaged_rates, mock_config):
        """Test that processing date is added."""
        result = create_fertility_rate_table(
            sample_averaged_rates, validate=False, config=mock_config
        )

        assert "processing_date" in result.columns


class TestValidateFertilityRates:
    """Tests for validate_fertility_rates function."""

    @pytest.fixture
    def valid_fertility_rates(self):
        """Create valid fertility rate table."""
        ages = list(range(15, 50))
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
            for race in races:
                # Realistic fertility pattern
                rate = 0.001 + 0.004 * np.exp(-((age - 28) ** 2) / 50)
                records.append({"age": age, "race_ethnicity": race, "fertility_rate": rate})

        return pd.DataFrame(records)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for validation."""
        return {
            "demographics": {
                "race_ethnicity": {
                    "categories": [
                        "White alone, Non-Hispanic",
                        "Black alone, Non-Hispanic",
                        "AIAN alone, Non-Hispanic",
                        "Asian/PI alone, Non-Hispanic",
                        "Two or more races, Non-Hispanic",
                        "Hispanic (any race)",
                    ]
                }
            },
            "rates": {"fertility": {"apply_to_ages": [15, 49]}},
        }

    def test_validate_fertility_rates_valid(self, valid_fertility_rates, mock_config):
        """Test validation of valid fertility rates."""
        result = validate_fertility_rates(valid_fertility_rates, mock_config)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert "tfr_by_race" in result
        assert "overall_tfr" in result

    def test_validate_fertility_rates_missing_ages(self, mock_config):
        """Test that missing ages are flagged as errors."""
        df = pd.DataFrame(
            {
                "age": [20, 25, 30],  # Missing most reproductive ages
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "fertility_rate": [0.05, 0.08, 0.07],
            }
        )

        result = validate_fertility_rates(df, mock_config)

        assert result["valid"] is False
        assert any("Missing ages" in error for error in result["errors"])

    def test_validate_fertility_rates_missing_races(self, mock_config):
        """Test that missing race categories are flagged."""
        ages = list(range(15, 50))
        df = pd.DataFrame(
            {
                "age": ages,
                "race_ethnicity": ["White alone, Non-Hispanic"] * len(ages),
                "fertility_rate": [0.05] * len(ages),
            }
        )

        result = validate_fertility_rates(df, mock_config)

        assert result["valid"] is False
        assert any("Missing race" in error for error in result["errors"])

    def test_validate_fertility_rates_negative_rates(self, valid_fertility_rates, mock_config):
        """Test that negative rates are flagged as errors."""
        df = valid_fertility_rates.copy()
        df.loc[0, "fertility_rate"] = -0.05

        result = validate_fertility_rates(df, mock_config)

        assert result["valid"] is False
        assert any("Negative" in error for error in result["errors"])

    def test_validate_fertility_rates_high_rates_warning(self, valid_fertility_rates, mock_config):
        """Test that very high rates generate warnings."""
        df = valid_fertility_rates.copy()
        df.loc[0, "fertility_rate"] = 0.20  # Implausibly high

        result = validate_fertility_rates(df, mock_config)

        assert len(result["warnings"]) > 0
        assert any("high" in warning.lower() for warning in result["warnings"])

    def test_validate_fertility_rates_tfr_calculation(self, valid_fertility_rates, mock_config):
        """Test that TFR is calculated correctly."""
        result = validate_fertility_rates(valid_fertility_rates, mock_config)

        assert "tfr_by_race" in result
        assert len(result["tfr_by_race"]) == 6

        # TFR is the sum of age-specific fertility rates (35 ages with rate ~0.002-0.005)
        # Should be low but positive
        for _race, tfr in result["tfr_by_race"].items():
            assert tfr > 0  # Should have positive TFR
            assert tfr < 5.0  # Should be less than 5 (reasonable upper bound)

    def test_validate_fertility_rates_low_tfr_warning(self, mock_config):
        """Test that very low TFR generates warning."""
        ages = list(range(15, 50))
        races = mock_config["demographics"]["race_ethnicity"]["categories"]

        records = []
        for age in ages:
            for race in races:
                records.append(
                    {
                        "age": age,
                        "race_ethnicity": race,
                        "fertility_rate": 0.005,  # Very low, TFR ~0.175
                    }
                )

        df = pd.DataFrame(records)
        result = validate_fertility_rates(df, mock_config)

        # The very low TFR should trigger a warning
        # The warning might say "low TFR" or "Very low TFR"
        assert any(
            "low" in warning.lower() and "tfr" in warning.lower() for warning in result["warnings"]
        )

    def test_validate_fertility_rates_missing_columns(self, mock_config):
        """Test validation with missing required columns."""
        df = pd.DataFrame(
            {
                "age": [20],
                "fertility_rate": [0.05],
                # Missing race_ethnicity
            }
        )

        result = validate_fertility_rates(df, mock_config)

        assert result["valid"] is False
        assert any("Missing required columns" in error for error in result["errors"])


class TestSeerRaceEthnicityMap:
    """Tests for the SEER_RACE_ETHNICITY_MAP constant."""

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

        actual_outputs = set(SEER_RACE_ETHNICITY_MAP.values())

        assert expected_outputs == actual_outputs

    def test_common_variations_covered(self):
        """Test that common SEER race code variations are covered."""
        common_variations = [
            "White Non-Hispanic",
            "White NH",
            "NH White",
            "Black Non-Hispanic",
            "Hispanic",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",  # Numeric codes
        ]

        for variation in common_variations:
            assert variation in SEER_RACE_ETHNICITY_MAP


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe_harmonization(self):
        """Test harmonization of empty DataFrame."""
        df = pd.DataFrame({"age": [], "race": [], "fertility_rate": []})

        result = harmonize_fertility_race_categories(df)
        assert len(result) == 0

    def test_single_record_averaging(self):
        """Test averaging with single record."""
        df = pd.DataFrame(
            {"age": [25], "race_ethnicity": ["White alone, Non-Hispanic"], "fertility_rate": [0.08]}
        )

        result = calculate_average_fertility_rates(df)

        assert len(result) == 1
        assert result["fertility_rate"].iloc[0] == 0.08

    def test_all_zero_fertility_rates(self):
        """Test handling of all-zero fertility rates."""
        mock_config = {
            "demographics": {"race_ethnicity": {"categories": ["White alone, Non-Hispanic"]}},
            "rates": {
                "fertility": {
                    "apply_to_ages": [15, 17]  # Small range
                }
            },
        }

        df = pd.DataFrame(
            {
                "age": [15, 16, 17],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "fertility_rate": [0.0, 0.0, 0.0],
            }
        )

        result = create_fertility_rate_table(df, validate=False, config=mock_config)

        assert (result["fertility_rate"] == 0).all()

    def test_very_small_fertility_rates(self):
        """Test handling of very small fertility rates."""
        df = pd.DataFrame(
            {
                "age": [45, 46, 47],  # Older reproductive ages
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "fertility_rate": [0.0001, 0.00005, 0.00001],
            }
        )

        result = calculate_average_fertility_rates(df)

        # Should not be zero but very small
        assert result["fertility_rate"].iloc[0] > 0
        assert result["fertility_rate"].iloc[0] < 0.001

    def test_whitespace_in_race_codes(self):
        """Test handling of whitespace in race codes."""
        df = pd.DataFrame(
            {
                "age": [20, 25],
                "race": ["  White NH  ", "  Hispanic  "],  # Extra whitespace
                "fertility_rate": [0.05, 0.07],
            }
        )

        result = harmonize_fertility_race_categories(df)

        # Should handle whitespace correctly
        assert "White alone, Non-Hispanic" in result["race_ethnicity"].values
        assert "Hispanic (any race)" in result["race_ethnicity"].values
