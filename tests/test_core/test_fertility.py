"""
Unit tests for the fertility module.

Tests calculate_births, validate_fertility_rates, and apply_fertility_scenario functions.
"""

import pandas as pd
import pytest

from cohort_projections.core.fertility import (
    apply_fertility_scenario,
    calculate_births,
    validate_fertility_rates,
)


class TestCalculateBirths:
    """Tests for calculate_births function."""

    @pytest.fixture
    def sample_female_population(self):
        """Sample female population data for testing."""
        data = []
        # Create population for ages 15-49 (reproductive ages) by race
        for race in ["White", "Black", "Hispanic"]:
            for age in range(15, 50):
                data.append(
                    {
                        "age": age,
                        "race": race,
                        "population": 1000.0,  # 1000 females per age-race cohort
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_fertility_rates(self):
        """Sample fertility rates for testing."""
        data = []
        # Create fertility rates for ages 15-49 by race
        for race in ["White", "Black", "Hispanic"]:
            for age in range(15, 50):
                # Peak fertility around age 25-30
                if 20 <= age <= 34:
                    rate = 0.08
                elif 15 <= age <= 19 or 35 <= age <= 39:
                    rate = 0.04
                else:
                    rate = 0.01
                data.append(
                    {
                        "age": age,
                        "race": race,
                        "fertility_rate": rate,
                    }
                )
        return pd.DataFrame(data)

    def test_calculate_births_valid_input(self, sample_female_population, sample_fertility_rates):
        """Test calculate_births with valid input data."""
        result = calculate_births(sample_female_population, sample_fertility_rates, year=2025)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ["year", "age", "sex", "race", "population"])
        assert (result["age"] == 0).all()  # All births are age 0
        assert set(result["sex"]) == {"Male", "Female"}

    def test_calculate_births_returns_correct_columns(
        self, sample_female_population, sample_fertility_rates
    ):
        """Test that result has correct columns."""
        result = calculate_births(sample_female_population, sample_fertility_rates, year=2025)

        expected_columns = ["year", "age", "sex", "race", "population"]
        assert list(result.columns) == expected_columns

    def test_calculate_births_positive_population(
        self, sample_female_population, sample_fertility_rates
    ):
        """Test that calculated births are non-negative."""
        result = calculate_births(sample_female_population, sample_fertility_rates, year=2025)

        assert (result["population"] >= 0).all()

    def test_calculate_births_sex_ratio(self, sample_female_population, sample_fertility_rates):
        """Test that default sex ratio at birth is approximately 51% male."""
        result = calculate_births(sample_female_population, sample_fertility_rates, year=2025)

        total_births = result["population"].sum()
        male_births = result[result["sex"] == "Male"]["population"].sum()

        # Default ratio is 0.51 male
        expected_ratio = 0.51
        actual_ratio = male_births / total_births if total_births > 0 else 0

        assert abs(actual_ratio - expected_ratio) < 0.001

    def test_calculate_births_empty_population(self, sample_fertility_rates):
        """Test calculate_births with empty female population."""
        empty_pop = pd.DataFrame(columns=["age", "race", "population"])

        result = calculate_births(empty_pop, sample_fertility_rates, year=2025)

        assert result.empty
        assert all(col in result.columns for col in ["year", "age", "sex", "race", "population"])

    def test_calculate_births_no_reproductive_age_females(self, sample_fertility_rates):
        """Test with population that has no reproductive age females."""
        non_reproductive_pop = pd.DataFrame(
            {
                "age": [0, 5, 10, 55, 60, 70],
                "race": ["White"] * 6,
                "population": [100.0] * 6,
            }
        )

        result = calculate_births(non_reproductive_pop, sample_fertility_rates, year=2025)

        assert result.empty

    def test_calculate_births_with_custom_config(
        self, sample_female_population, sample_fertility_rates
    ):
        """Test calculate_births with custom configuration."""
        config = {
            "rates": {
                "fertility": {
                    "apply_to_ages": [20, 40],  # Narrower reproductive age range
                    "sex_ratio_male": 0.52,  # Custom sex ratio
                }
            }
        }

        result = calculate_births(
            sample_female_population, sample_fertility_rates, year=2025, config=config
        )

        assert not result.empty
        # Check custom sex ratio is applied
        total_births = result["population"].sum()
        male_births = result[result["sex"] == "Male"]["population"].sum()
        actual_ratio = male_births / total_births if total_births > 0 else 0

        assert abs(actual_ratio - 0.52) < 0.001

    def test_calculate_births_missing_fertility_rate_column(self, sample_female_population):
        """Test that missing fertility rates default to 0."""
        rates_without_rate = pd.DataFrame(
            {
                "age": [25, 30],
                "race": ["White", "White"],
            }
        )

        # Should handle gracefully by setting fertility_rate to 0
        result = calculate_births(sample_female_population, rates_without_rate, year=2025)

        # Result should have births but with 0 population due to missing rates
        total_births = result["population"].sum()
        assert total_births == 0.0

    def test_calculate_births_preserves_race_categories(
        self, sample_female_population, sample_fertility_rates
    ):
        """Test that births are produced for each race category."""
        result = calculate_births(sample_female_population, sample_fertility_rates, year=2025)

        input_races = set(sample_female_population["race"].unique())
        output_races = set(result["race"].unique())

        assert output_races == input_races

    def test_calculate_births_year_set_correctly(
        self, sample_female_population, sample_fertility_rates
    ):
        """Test that year is correctly set in output."""
        test_year = 2030
        result = calculate_births(sample_female_population, sample_fertility_rates, year=test_year)

        assert (result["year"] == test_year).all()

    def test_calculate_births_missing_age_column(self, sample_fertility_rates):
        """Test that ValueError is raised when age column is missing."""
        invalid_pop = pd.DataFrame(
            {
                "race": ["White", "White"],
                "population": [100.0, 100.0],
            }
        )

        with pytest.raises(ValueError, match="'age' and 'race' columns"):
            calculate_births(invalid_pop, sample_fertility_rates, year=2025)

    def test_calculate_births_missing_race_column_in_rates(self, sample_female_population):
        """Test that ValueError is raised when race column is missing in rates."""
        invalid_rates = pd.DataFrame(
            {
                "age": [25, 30],
                "fertility_rate": [0.08, 0.08],
            }
        )

        with pytest.raises(ValueError, match="'age' and 'race' columns"):
            calculate_births(sample_female_population, invalid_rates, year=2025)


class TestValidateFertilityRates:
    """Tests for validate_fertility_rates function."""

    @pytest.fixture
    def valid_fertility_rates(self):
        """Valid fertility rates for testing."""
        data = []
        for race in ["White", "Black"]:
            for age in range(15, 50):
                rate = 0.08 if 20 <= age <= 34 else 0.02
                data.append(
                    {
                        "age": age,
                        "race": race,
                        "fertility_rate": rate,
                    }
                )
        return pd.DataFrame(data)

    def test_validate_valid_rates(self, valid_fertility_rates):
        """Test validation passes for valid rates."""
        is_valid, issues = validate_fertility_rates(valid_fertility_rates)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_missing_columns(self):
        """Test validation fails when required columns are missing."""
        invalid_rates = pd.DataFrame(
            {
                "age": [25, 30],
                "rate": [0.08, 0.08],  # Wrong column name
            }
        )

        is_valid, issues = validate_fertility_rates(invalid_rates)

        assert is_valid is False
        assert any("Missing required columns" in issue for issue in issues)

    def test_validate_negative_rates(self):
        """Test validation catches negative fertility rates."""
        rates_with_negative = pd.DataFrame(
            {
                "age": [25, 30],
                "race": ["White", "White"],
                "fertility_rate": [0.08, -0.05],  # Negative rate
            }
        )

        is_valid, issues = validate_fertility_rates(rates_with_negative)

        assert is_valid is False
        assert any("Negative fertility rates" in issue for issue in issues)

    def test_validate_implausibly_high_rates(self):
        """Test validation catches implausibly high fertility rates."""
        rates_too_high = pd.DataFrame(
            {
                "age": [25, 30],
                "race": ["White", "White"],
                "fertility_rate": [0.08, 0.50],  # Implausibly high
            }
        )

        is_valid, issues = validate_fertility_rates(rates_too_high)

        assert is_valid is False
        assert any("Fertility rates > 0.35" in issue for issue in issues)

    def test_validate_with_config_outside_reproductive_age(self):
        """Test validation catches non-zero fertility outside reproductive ages."""
        rates_outside_range = pd.DataFrame(
            {
                "age": [10, 25, 55],  # 10 and 55 are outside typical range
                "race": ["White", "White", "White"],
                "fertility_rate": [0.05, 0.08, 0.05],  # Non-zero outside range
            }
        )

        config = {"rates": {"fertility": {"apply_to_ages": [15, 49]}}}

        is_valid, issues = validate_fertility_rates(rates_outside_range, config)

        assert is_valid is False
        assert any("outside age range" in issue for issue in issues)

    def test_validate_missing_combinations(self):
        """Test validation catches missing age-race combinations."""
        # Only partial combinations
        incomplete_rates = pd.DataFrame(
            {
                "age": [25, 25, 30],  # Missing 30 for Black
                "race": ["White", "Black", "White"],
                "fertility_rate": [0.08, 0.08, 0.08],
            }
        )

        is_valid, issues = validate_fertility_rates(incomplete_rates)

        assert is_valid is False
        assert any("Missing age-race combinations" in issue for issue in issues)


class TestApplyFertilityScenario:
    """Tests for apply_fertility_scenario function."""

    @pytest.fixture
    def base_fertility_rates(self):
        """Base fertility rates for scenario testing."""
        return pd.DataFrame(
            {
                "age": [20, 25, 30, 35],
                "race": ["White"] * 4,
                "fertility_rate": [0.05, 0.10, 0.10, 0.05],
            }
        )

    def test_constant_scenario(self, base_fertility_rates):
        """Test constant scenario returns unchanged rates."""
        result = apply_fertility_scenario(
            base_fertility_rates, "constant", year=2030, base_year=2025
        )

        pd.testing.assert_frame_equal(result, base_fertility_rates)

    def test_plus_10_percent_scenario(self, base_fertility_rates):
        """Test +10% scenario increases rates by 10%."""
        result = apply_fertility_scenario(
            base_fertility_rates, "+10_percent", year=2030, base_year=2025
        )

        expected_rates = base_fertility_rates["fertility_rate"] * 1.10
        pd.testing.assert_series_equal(result["fertility_rate"], expected_rates, check_names=False)

    def test_minus_10_percent_scenario(self, base_fertility_rates):
        """Test -10% scenario decreases rates by 10%."""
        result = apply_fertility_scenario(
            base_fertility_rates, "-10_percent", year=2030, base_year=2025
        )

        expected_rates = base_fertility_rates["fertility_rate"] * 0.90
        pd.testing.assert_series_equal(result["fertility_rate"], expected_rates, check_names=False)

    def test_trending_scenario(self, base_fertility_rates):
        """Test trending scenario applies annual decline."""
        years_elapsed = 5
        trend_factor = (1 - 0.005) ** years_elapsed

        result = apply_fertility_scenario(
            base_fertility_rates, "trending", year=2030, base_year=2025
        )

        expected_rates = base_fertility_rates["fertility_rate"] * trend_factor
        pd.testing.assert_series_equal(result["fertility_rate"], expected_rates, check_names=False)

    def test_unknown_scenario_uses_constant(self, base_fertility_rates):
        """Test unknown scenario falls back to constant (unchanged) rates."""
        result = apply_fertility_scenario(
            base_fertility_rates, "unknown_scenario", year=2030, base_year=2025
        )

        pd.testing.assert_series_equal(
            result["fertility_rate"],
            base_fertility_rates["fertility_rate"],
            check_names=False,
        )

    def test_scenario_ensures_non_negative(self):
        """Test that scenario ensures non-negative rates."""
        rates_with_small_values = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.001],
            }
        )

        # Apply drastic reduction
        result = apply_fertility_scenario(
            rates_with_small_values, "-10_percent", year=2030, base_year=2025
        )

        assert (result["fertility_rate"] >= 0).all()

    def test_scenario_preserves_other_columns(self, base_fertility_rates):
        """Test that scenario preserves non-rate columns."""
        result = apply_fertility_scenario(
            base_fertility_rates, "+10_percent", year=2030, base_year=2025
        )

        assert list(result["age"]) == list(base_fertility_rates["age"])
        assert list(result["race"]) == list(base_fertility_rates["race"])


class TestEdgeCases:
    """Test edge cases for fertility module."""

    def test_single_cohort(self):
        """Test with a single population cohort."""
        single_pop = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        single_rate = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.10],
            }
        )

        result = calculate_births(single_pop, single_rate, year=2025)

        assert not result.empty
        assert len(result) == 2  # Male and Female births
        total_births = result["population"].sum()
        assert abs(total_births - 100.0) < 0.001  # 1000 * 0.10 = 100

    def test_zero_fertility_rate(self):
        """Test with zero fertility rates."""
        pop = pd.DataFrame(
            {
                "age": [25, 30],
                "race": ["White", "White"],
                "population": [1000.0, 1000.0],
            }
        )
        zero_rates = pd.DataFrame(
            {
                "age": [25, 30],
                "race": ["White", "White"],
                "fertility_rate": [0.0, 0.0],
            }
        )

        result = calculate_births(pop, zero_rates, year=2025)

        total_births = result["population"].sum()
        assert total_births == 0.0

    def test_large_population(self):
        """Test with large population values."""
        large_pop = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "population": [1_000_000.0],
            }
        )
        rate = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.08],
            }
        )

        result = calculate_births(large_pop, rate, year=2025)

        total_births = result["population"].sum()
        expected_births = 1_000_000.0 * 0.08
        assert abs(total_births - expected_births) < 1.0


@pytest.mark.parametrize(
    "scenario,expected_multiplier",
    [
        ("constant", 1.0),
        ("+10_percent", 1.10),
        ("-10_percent", 0.90),
    ],
)
def test_scenario_multipliers(scenario, expected_multiplier):
    """Parametrized test for scenario multipliers."""
    rates = pd.DataFrame(
        {
            "age": [25],
            "race": ["White"],
            "fertility_rate": [0.10],
        }
    )

    result = apply_fertility_scenario(rates, scenario, year=2025, base_year=2025)

    expected_rate = 0.10 * expected_multiplier
    assert abs(result["fertility_rate"].iloc[0] - expected_rate) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
