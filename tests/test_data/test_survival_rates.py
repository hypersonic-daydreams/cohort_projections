"""
Unit tests for survival_rates.py data processing module.

Tests life table processing, survival rate calculations, and validation.
Uses synthetic DataFrames as fixtures - does not depend on actual data files.
"""

import pandas as pd
import pytest

from cohort_projections.data.process.survival_rates import (
    SEER_MORTALITY_RACE_MAP,
    apply_mortality_improvement,
    calculate_life_expectancy,
    calculate_survival_rates_from_life_table,
    create_survival_rate_table,
    harmonize_mortality_race_categories,
    load_life_table_data,
    validate_survival_rates,
)


class TestLoadLifeTableData:
    """Tests for load_life_table_data function."""

    @pytest.fixture
    def sample_life_table_csv(self, tmp_path):
        """Create a temporary CSV file with sample life table data."""
        data = pd.DataFrame(
            {
                "year": [2020] * 6,
                "age": [0, 1, 2, 0, 1, 2],
                "sex": ["Male", "Male", "Male", "Female", "Female", "Female"],
                "race": ["White NH"] * 6,
                "qx": [0.006, 0.0005, 0.0004, 0.005, 0.0004, 0.0003],
                "lx": [100000, 99400, 99350, 100000, 99500, 99460],
            }
        )
        file_path = tmp_path / "life_table.csv"
        data.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def sample_life_table_txt(self, tmp_path):
        """Create a temporary tab-delimited life table file."""
        data = pd.DataFrame(
            {
                "year": [2020] * 2,
                "age": [0, 1],
                "sex": ["Male", "Female"],
                "race": ["White NH", "Hispanic"],
                "qx": [0.006, 0.005],
                "lx": [100000, 100000],
            }
        )
        file_path = tmp_path / "life_table.txt"
        data.to_csv(file_path, sep="\t", index=False)
        return file_path

    def test_load_life_table_data_csv(self, sample_life_table_csv):
        """Test loading life table from CSV file."""
        result = load_life_table_data(sample_life_table_csv)

        assert len(result) == 6
        assert "age" in result.columns
        assert "sex" in result.columns
        assert "qx" in result.columns
        assert "lx" in result.columns

    def test_load_life_table_data_txt(self, sample_life_table_txt):
        """Test loading life table from tab-delimited file."""
        result = load_life_table_data(sample_life_table_txt)

        assert len(result) == 2
        assert "age" in result.columns

    def test_load_life_table_data_with_year_filter(self, sample_life_table_csv):
        """Test filtering life table by year."""
        result = load_life_table_data(sample_life_table_csv, year=2020)

        assert len(result) == 6
        assert (result["year"] == 2020).all()

    def test_load_life_table_data_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError):
            load_life_table_data("/nonexistent/path/life_table.csv")

    def test_load_life_table_data_unsupported_format(self, tmp_path):
        """Test that ValueError is raised for unsupported format."""
        file_path = tmp_path / "data.xyz"
        file_path.write_text("some data")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_life_table_data(file_path)

    def test_load_life_table_data_standardizes_sex(self, tmp_path):
        """Test that sex values are standardized."""
        data = pd.DataFrame(
            {
                "age": [0, 0, 0, 0],
                "sex": ["M", "F", "Males", "Females"],
                "race": ["White NH"] * 4,
                "qx": [0.006, 0.005, 0.006, 0.005],
                "lx": [100000] * 4,
            }
        )
        file_path = tmp_path / "life_table.csv"
        data.to_csv(file_path, index=False)

        result = load_life_table_data(file_path)

        assert set(result["sex"].unique()) == {"Male", "Female"}

    def test_load_life_table_data_no_age_column(self, tmp_path):
        """Test that ValueError is raised when no age column found."""
        data = pd.DataFrame({"sex": ["Male"], "race": ["White NH"], "qx": [0.006]})
        file_path = tmp_path / "life_table.csv"
        data.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="No age column found"):
            load_life_table_data(file_path)

    def test_load_life_table_data_no_sex_column(self, tmp_path):
        """Test that ValueError is raised when no sex column found."""
        data = pd.DataFrame({"age": [0], "race": ["White NH"], "qx": [0.006]})
        file_path = tmp_path / "life_table.csv"
        data.to_csv(file_path, index=False)

        with pytest.raises(ValueError, match="No sex/gender column found"):
            load_life_table_data(file_path)


class TestHarmonizeMortalityRaceCategories:
    """Tests for harmonize_mortality_race_categories function."""

    @pytest.fixture
    def sample_life_table_data(self):
        """Sample life table data with SEER race codes."""
        return pd.DataFrame(
            {
                "age": [0, 1, 2],
                "sex": ["Male", "Female", "Male"],
                "race": ["White NH", "Hispanic", "Black NH"],
                "qx": [0.006, 0.005, 0.007],
            }
        )

    def test_harmonize_mortality_race_categories_basic(self, sample_life_table_data):
        """Test basic race category harmonization."""
        result = harmonize_mortality_race_categories(sample_life_table_data)

        assert "race_ethnicity" in result.columns
        assert "race" not in result.columns

        expected = ["White alone, Non-Hispanic", "Hispanic (any race)", "Black alone, Non-Hispanic"]
        assert list(result["race_ethnicity"]) == expected

    def test_harmonize_mortality_race_categories_numeric_codes(self):
        """Test harmonization of numeric SEER codes."""
        df = pd.DataFrame(
            {
                "age": [0, 0],
                "sex": ["Male", "Female"],
                "race": ["1", "6"],  # Numeric codes
                "qx": [0.006, 0.005],
            }
        )

        result = harmonize_mortality_race_categories(df)

        assert result["race_ethnicity"].iloc[0] == "White alone, Non-Hispanic"
        assert result["race_ethnicity"].iloc[1] == "Hispanic (any race)"

    def test_harmonize_mortality_race_categories_no_race_column(self):
        """Test that ValueError is raised when no race column found."""
        df = pd.DataFrame({"age": [0], "sex": ["Male"], "qx": [0.006]})

        with pytest.raises(ValueError, match="No race/ethnicity column found"):
            harmonize_mortality_race_categories(df)

    def test_harmonize_mortality_race_categories_unmapped_dropped(self):
        """Test that unmapped race categories are dropped."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 0],
                "sex": ["Male"] * 3,
                "race": ["White NH", "UNKNOWN", "Hispanic"],
                "qx": [0.006, 0.007, 0.005],
            }
        )

        result = harmonize_mortality_race_categories(df)

        assert len(result) == 2


class TestCalculateSurvivalRatesFromLifeTable:
    """Tests for calculate_survival_rates_from_life_table function."""

    @pytest.fixture
    def sample_life_table_lx(self):
        """Sample life table with lx column."""
        return pd.DataFrame(
            {
                "age": [0, 1, 2, 0, 1, 2],
                "sex": ["Male"] * 3 + ["Female"] * 3,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 6,
                "lx": [100000, 99400, 99350, 100000, 99500, 99460],
                "qx": [0.006, 0.0005, 0.0004, 0.005, 0.0005, 0.0003],
            }
        )

    @pytest.fixture
    def sample_life_table_qx(self):
        """Sample life table with qx column only."""
        return pd.DataFrame(
            {
                "age": [0, 1, 2],
                "sex": ["Male"] * 3,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "qx": [0.006, 0.0005, 0.0004],
            }
        )

    def test_calculate_survival_rates_lx_method(self, sample_life_table_lx):
        """Test survival rate calculation using lx method."""
        result = calculate_survival_rates_from_life_table(sample_life_table_lx, method="lx")

        assert "survival_rate" in result.columns
        assert len(result) == 6

        # Check Male age 0 survival rate: 99400 / 100000 = 0.994
        male_age_0 = result[(result["sex"] == "Male") & (result["age"] == 0)]
        assert abs(male_age_0["survival_rate"].iloc[0] - 0.994) < 0.001

    def test_calculate_survival_rates_qx_method(self, sample_life_table_qx):
        """Test survival rate calculation using qx method."""
        result = calculate_survival_rates_from_life_table(sample_life_table_qx, method="qx")

        # S(x) = 1 - q(x)
        # Age 0: 1 - 0.006 = 0.994
        age_0 = result[result["age"] == 0]
        assert abs(age_0["survival_rate"].iloc[0] - 0.994) < 0.001

    def test_calculate_survival_rates_invalid_method(self, sample_life_table_lx):
        """Test that ValueError is raised for invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            calculate_survival_rates_from_life_table(sample_life_table_lx, method="invalid")

    def test_calculate_survival_rates_missing_lx(self):
        """Test that ValueError is raised when lx column missing for lx method."""
        df = pd.DataFrame(
            {
                "age": [0],
                "sex": ["Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                "qx": [0.006],
            }
        )

        with pytest.raises(ValueError, match="requires 'lx' column"):
            calculate_survival_rates_from_life_table(df, method="lx")

    def test_calculate_survival_rates_range_valid(self, sample_life_table_lx):
        """Test that survival rates are in valid range [0, 1]."""
        result = calculate_survival_rates_from_life_table(sample_life_table_lx, method="lx")

        assert (result["survival_rate"] >= 0).all()
        assert (result["survival_rate"] <= 1).all()

    def test_calculate_survival_rates_age_90_plus(self):
        """Test special handling of age 90+ open-ended group."""
        df = pd.DataFrame(
            {
                "age": [89, 90],
                "sex": ["Male"] * 2,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2,
                "lx": [50000, 35000],
                "tx": [150000, 100000],
                "qx": [0.10, 0.35],
            }
        )

        result = calculate_survival_rates_from_life_table(df, method="lx")

        # Age 90 should have special handling
        age_90 = result[result["age"] == 90]
        assert len(age_90) == 1
        # Survival rate should be calculated for 90+


class TestApplyMortalityImprovement:
    """Tests for apply_mortality_improvement function."""

    @pytest.fixture
    def sample_survival_rates(self):
        """Sample survival rates for testing improvement."""
        return pd.DataFrame(
            {
                "age": [65, 70, 75],
                "sex": ["Female"] * 3,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "survival_rate": [0.980, 0.970, 0.950],
            }
        )

    def test_apply_mortality_improvement_basic(self, sample_survival_rates):
        """Test basic mortality improvement application."""
        result = apply_mortality_improvement(
            sample_survival_rates, base_year=2020, projection_year=2030, improvement_factor=0.005
        )

        # Survival rates should increase (mortality decreases)
        assert (result["survival_rate"] > sample_survival_rates["survival_rate"]).all()

    def test_apply_mortality_improvement_no_change_same_year(self, sample_survival_rates):
        """Test that no improvement is applied when projection_year <= base_year."""
        result = apply_mortality_improvement(
            sample_survival_rates, base_year=2020, projection_year=2020, improvement_factor=0.005
        )

        pd.testing.assert_frame_equal(result, sample_survival_rates)

    def test_apply_mortality_improvement_zero_factor(self, sample_survival_rates):
        """Test that no improvement is applied when factor is 0."""
        result = apply_mortality_improvement(
            sample_survival_rates, base_year=2020, projection_year=2030, improvement_factor=0.0
        )

        pd.testing.assert_frame_equal(result, sample_survival_rates)

    def test_apply_mortality_improvement_capped_at_one(self):
        """Test that survival rates are capped at 1.0."""
        df = pd.DataFrame(
            {
                "age": [5],
                "sex": ["Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                "survival_rate": [0.9999],  # Very high already
            }
        )

        result = apply_mortality_improvement(
            df,
            base_year=2000,
            projection_year=2100,  # 100 years of improvement
            improvement_factor=0.01,
        )

        assert (result["survival_rate"] <= 1.0).all()

    def test_apply_mortality_improvement_calculation(self):
        """Test mortality improvement calculation is correct."""
        df = pd.DataFrame(
            {
                "age": [50],
                "sex": ["Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                "survival_rate": [0.99],  # Death rate = 0.01
            }
        )

        result = apply_mortality_improvement(
            df,
            base_year=2020,
            projection_year=2030,
            improvement_factor=0.01,  # 1% annual improvement
        )

        # Death rate should decrease: 0.01 * (1 - 0.01)^10
        expected_death_rate = 0.01 * (0.99**10)
        expected_survival = 1 - expected_death_rate

        assert abs(result["survival_rate"].iloc[0] - expected_survival) < 0.0001


class TestCreateSurvivalRateTable:
    """Tests for create_survival_rate_table function."""

    @pytest.fixture
    def sample_survival_rates(self):
        """Sample survival rates for table creation."""
        return pd.DataFrame(
            {
                "age": [0, 1, 2],
                "sex": ["Male"] * 3,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "survival_rate": [0.994, 0.9995, 0.9995],
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
                },
                "sex": ["Male", "Female"],
                "age_groups": {"min_age": 0, "max_age": 90},
            }
        }

    def test_create_survival_rate_table_fills_missing(self, sample_survival_rates, mock_config):
        """Test that missing cohorts are filled with defaults."""
        result = create_survival_rate_table(
            sample_survival_rates, validate=False, config=mock_config
        )

        # Should have 91 ages * 2 sexes * 6 races = 1092 rows
        expected_rows = 91 * 2 * 6
        assert len(result) == expected_rows

    def test_create_survival_rate_table_default_rates_by_age(self, mock_config):
        """Test that default rates vary by age appropriately."""
        # Empty DataFrame to get all defaults
        df = pd.DataFrame({"age": [], "sex": [], "race_ethnicity": [], "survival_rate": []})

        result = create_survival_rate_table(df, validate=False, config=mock_config)

        # Check infant default
        infant = result[(result["age"] == 0) & (result["sex"] == "Male")].iloc[0]
        assert abs(infant["survival_rate"] - 0.994) < 0.001

        # Check child default
        child = result[(result["age"] == 5) & (result["sex"] == "Male")].iloc[0]
        assert abs(child["survival_rate"] - 0.9995) < 0.001

        # Check age 90+ default
        elderly = result[(result["age"] == 90) & (result["sex"] == "Male")].iloc[0]
        assert abs(elderly["survival_rate"] - 0.65) < 0.01

    def test_create_survival_rate_table_missing_columns(self, mock_config):
        """Test that ValueError is raised for missing columns."""
        df = pd.DataFrame(
            {
                "age": [0],
                "sex": ["Male"],
                # Missing race_ethnicity and survival_rate
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            create_survival_rate_table(df, validate=False, config=mock_config)

    def test_create_survival_rate_table_rates_clipped(self, mock_config):
        """Test that survival rates are clipped to [0, 1]."""
        df = pd.DataFrame(
            {
                "age": [50, 51],
                "sex": ["Male", "Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2,
                "survival_rate": [-0.1, 1.5],  # Invalid values
            }
        )

        result = create_survival_rate_table(df, validate=False, config=mock_config)

        assert (result["survival_rate"] >= 0).all()
        assert (result["survival_rate"] <= 1).all()


class TestCalculateLifeExpectancy:
    """Tests for calculate_life_expectancy function."""

    @pytest.fixture
    def sample_full_survival_table(self):
        """Create a full survival rate table for life expectancy calculation."""
        ages = list(range(91))
        records = []

        for sex in ["Male", "Female"]:
            for age in ages:
                # Simplified survival pattern
                if age == 0:
                    rate = 0.994
                elif age < 15:
                    rate = 0.9998
                elif age < 65:
                    rate = 0.9990
                elif age < 85:
                    rate = 0.95
                else:
                    rate = 0.70

                # Females live slightly longer
                if sex == "Female":
                    rate = min(1.0, rate * 1.01)

                records.append(
                    {
                        "age": age,
                        "sex": sex,
                        "race_ethnicity": "White alone, Non-Hispanic",
                        "survival_rate": rate,
                    }
                )

        return pd.DataFrame(records)

    def test_calculate_life_expectancy_basic(self, sample_full_survival_table):
        """Test basic life expectancy calculation."""
        result = calculate_life_expectancy(sample_full_survival_table)

        assert isinstance(result, dict)
        assert "Male_White alone, Non-Hispanic" in result
        assert "Female_White alone, Non-Hispanic" in result

    def test_calculate_life_expectancy_female_higher(self, sample_full_survival_table):
        """Test that female life expectancy is typically higher."""
        result = calculate_life_expectancy(sample_full_survival_table)

        male_e0 = result["Male_White alone, Non-Hispanic"]
        female_e0 = result["Female_White alone, Non-Hispanic"]

        # Females typically live longer
        assert female_e0 > male_e0

    def test_calculate_life_expectancy_reasonable_range(self, sample_full_survival_table):
        """Test that calculated life expectancy is in reasonable range."""
        result = calculate_life_expectancy(sample_full_survival_table)

        for _key, e0 in result.items():
            # Life expectancy should be between 50 and 100 years
            assert 50 < e0 < 100


class TestValidateSurvivalRates:
    """Tests for validate_survival_rates function."""

    @pytest.fixture
    def valid_survival_rates(self):
        """Create valid survival rate table."""
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
                    # Realistic survival pattern
                    if age == 0:
                        rate = 0.994
                    elif age < 15:
                        rate = 0.9998
                    elif age < 65:
                        rate = 0.9980
                    elif age < 85:
                        rate = 0.96
                    else:
                        rate = 0.65

                    records.append(
                        {"age": age, "sex": sex, "race_ethnicity": race, "survival_rate": rate}
                    )

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
                },
                "sex": ["Male", "Female"],
                "age_groups": {"min_age": 0, "max_age": 90},
            }
        }

    def test_validate_survival_rates_valid(self, valid_survival_rates, mock_config):
        """Test validation of valid survival rates."""
        result = validate_survival_rates(valid_survival_rates, mock_config)

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_survival_rates_missing_ages(self, mock_config):
        """Test that missing ages are flagged as errors."""
        df = pd.DataFrame(
            {
                "age": [0, 1, 2],  # Missing most ages
                "sex": ["Male"] * 3,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 3,
                "survival_rate": [0.994, 0.9995, 0.9995],
            }
        )

        result = validate_survival_rates(df, mock_config)

        assert result["valid"] is False
        assert any("Missing ages" in error for error in result["errors"])

    def test_validate_survival_rates_negative_rates(self, valid_survival_rates, mock_config):
        """Test that negative rates are flagged as errors."""
        df = valid_survival_rates.copy()
        df.loc[0, "survival_rate"] = -0.05

        result = validate_survival_rates(df, mock_config)

        assert result["valid"] is False
        assert any("Negative" in error for error in result["errors"])

    def test_validate_survival_rates_over_one(self, valid_survival_rates, mock_config):
        """Test that rates > 1 are flagged as errors."""
        df = valid_survival_rates.copy()
        df.loc[0, "survival_rate"] = 1.5

        result = validate_survival_rates(df, mock_config)

        assert result["valid"] is False
        assert any("> 1.0" in error for error in result["errors"])

    def test_validate_survival_rates_low_infant_warning(self, mock_config):
        """Test that low infant survival rate generates warning."""
        ages = list(range(91))
        sexes = ["Male", "Female"]
        races = mock_config["demographics"]["race_ethnicity"]["categories"]

        records = []
        for age in ages:
            for sex in sexes:
                for race in races:
                    # 0.95 unusually low for modern infant data
                    rate = 0.95 if age == 0 else 0.99
                    records.append(
                        {"age": age, "sex": sex, "race_ethnicity": race, "survival_rate": rate}
                    )

        df = pd.DataFrame(records)
        result = validate_survival_rates(df, mock_config)

        assert any("infant" in warning.lower() for warning in result["warnings"])

    def test_validate_survival_rates_life_expectancy_calculated(
        self, valid_survival_rates, mock_config
    ):
        """Test that life expectancy is calculated during validation."""
        result = validate_survival_rates(valid_survival_rates, mock_config)

        le_key = "life_expectancy"
        assert le_key in result
        assert len(result[le_key]) > 0


class TestSeerMortalityRaceMap:
    """Tests for the SEER_MORTALITY_RACE_MAP constant."""

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

        actual_outputs = set(SEER_MORTALITY_RACE_MAP.values())

        assert expected_outputs == actual_outputs

    def test_consistent_with_fertility_map(self):
        """Test that mortality race map is consistent with fertility map."""
        from cohort_projections.data.process.fertility_rates import SEER_RACE_ETHNICITY_MAP

        # Output categories should be the same
        mortality_outputs = set(SEER_MORTALITY_RACE_MAP.values())
        fertility_outputs = set(SEER_RACE_ETHNICITY_MAP.values())

        assert mortality_outputs == fertility_outputs


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_dataframe_harmonization(self):
        """Test harmonization of empty DataFrame."""
        df = pd.DataFrame({"age": [], "sex": [], "race": [], "qx": []})

        result = harmonize_mortality_race_categories(df)
        assert len(result) == 0

    def test_survival_rate_zero_lx(self):
        """Test handling of zero lx values (division by zero)."""
        df = pd.DataFrame(
            {
                "age": [85, 86],
                "sex": ["Male"] * 2,
                "race_ethnicity": ["White alone, Non-Hispanic"] * 2,
                "lx": [0, 0],  # Zero values
                "qx": [1.0, 1.0],
            }
        )

        result = calculate_survival_rates_from_life_table(df, method="lx")

        # Should handle gracefully, not raise exception
        assert "survival_rate" in result.columns
        assert not result["survival_rate"].isna().any()

    def test_very_high_mortality_improvement(self):
        """Test mortality improvement with very high factor."""
        df = pd.DataFrame(
            {
                "age": [50],
                "sex": ["Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                "survival_rate": [0.95],
            }
        )

        result = apply_mortality_improvement(
            df,
            base_year=2000,
            projection_year=2100,
            improvement_factor=0.05,  # 5% annual - unrealistically high
        )

        # Should still be capped at 1.0
        assert result["survival_rate"].iloc[0] <= 1.0

    def test_single_age_survival_table(self):
        """Test survival rate calculation with single age."""
        df = pd.DataFrame(
            {
                "age": [50],
                "sex": ["Male"],
                "race_ethnicity": ["White alone, Non-Hispanic"],
                "qx": [0.005],
            }
        )

        result = calculate_survival_rates_from_life_table(df, method="qx")

        assert len(result) == 1
        assert abs(result["survival_rate"].iloc[0] - 0.995) < 0.001

    def test_whitespace_in_values(self):
        """Test handling of whitespace in race codes."""
        df = pd.DataFrame(
            {
                "age": [0, 0],
                "sex": ["Male", "Female"],
                "race": ["  White NH  ", "  Hispanic  "],  # Extra whitespace
                "qx": [0.006, 0.005],
            }
        )

        result = harmonize_mortality_race_categories(df)

        assert "White alone, Non-Hispanic" in result["race_ethnicity"].values
        assert "Hispanic (any race)" in result["race_ethnicity"].values
