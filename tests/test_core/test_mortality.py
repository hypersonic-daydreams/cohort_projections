"""
Unit tests for the mortality module.

Tests apply_survival_rates, apply_mortality_improvement, validate_survival_rates,
and calculate_life_expectancy functions.
"""

import pandas as pd
import pytest

from cohort_projections.core.mortality import (
    apply_mortality_improvement,
    apply_survival_rates,
    calculate_life_expectancy,
    validate_survival_rates,
)


class TestApplySurvivalRates:
    """Tests for apply_survival_rates function."""

    @pytest.fixture
    def sample_population(self):
        """Sample population data for testing."""
        data = []
        # Create population for ages 0-90 by sex and race
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    data.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": 1000.0,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_survival_rates(self):
        """Sample survival rates for testing."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    # Higher survival at younger ages, lower at older ages
                    if age < 1:
                        rate = 0.993  # Infant survival
                    elif age < 65:
                        rate = 0.999
                    elif age < 80:
                        rate = 0.97
                    else:
                        rate = 0.90
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "survival_rate": rate,
                        }
                    )
        return pd.DataFrame(data)

    def test_apply_survival_rates_valid_input(self, sample_population, sample_survival_rates):
        """Test apply_survival_rates with valid input data."""
        result = apply_survival_rates(sample_population, sample_survival_rates, year=2025)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ["year", "age", "sex", "race", "population"])

    def test_survival_ages_population_by_one_year(self, sample_population, sample_survival_rates):
        """Test that population is aged by one year after survival."""
        result = apply_survival_rates(sample_population, sample_survival_rates, year=2025)

        # Year should be 2026 (year + 1)
        assert (result["year"] == 2026).all()

        # Ages should be shifted (except for max_age group)
        # Check that age 0 population became age 1
        age_1_pop = result[
            (result["age"] == 1) & (result["sex"] == "Male") & (result["race"] == "White")
        ]["population"].sum()
        assert age_1_pop > 0

    def test_survival_reduces_population(self, sample_population, sample_survival_rates):
        """Test that total population decreases after survival (mortality)."""
        result = apply_survival_rates(sample_population, sample_survival_rates, year=2025)

        total_before = sample_population["population"].sum()
        total_after = result["population"].sum()

        assert total_after < total_before

    def test_survival_empty_population(self, sample_survival_rates):
        """Test apply_survival_rates with empty population."""
        empty_pop = pd.DataFrame(columns=["year", "age", "sex", "race", "population"])

        result = apply_survival_rates(empty_pop, sample_survival_rates, year=2025)

        assert result.empty
        assert all(col in result.columns for col in ["year", "age", "sex", "race", "population"])

    def test_survival_preserves_categories(self, sample_population, sample_survival_rates):
        """Test that sex and race categories are preserved."""
        result = apply_survival_rates(sample_population, sample_survival_rates, year=2025)

        assert set(result["sex"].unique()) == set(sample_population["sex"].unique())
        assert set(result["race"].unique()) == set(sample_population["race"].unique())

    def test_survival_missing_columns_raises_error(self, sample_survival_rates):
        """Test that missing population columns raise ValueError."""
        invalid_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                # Missing 'race' and 'population'
            }
        )

        with pytest.raises(ValueError, match="population must have columns"):
            apply_survival_rates(invalid_pop, sample_survival_rates, year=2025)

    def test_survival_missing_rate_columns_raises_error(self, sample_population):
        """Test that missing survival rate columns raise ValueError."""
        invalid_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                # Missing 'race' and 'survival_rate'
            }
        )

        with pytest.raises(ValueError, match="survival_rates must have columns"):
            apply_survival_rates(sample_population, invalid_rates, year=2025)

    def test_survival_open_age_group_handling(self, sample_population, sample_survival_rates):
        """Test that 90+ age group is handled correctly."""
        result = apply_survival_rates(sample_population, sample_survival_rates, year=2025)

        # Population at max_age (90) should include:
        # 1. Survivors from age 89 who aged to 90
        # 2. Survivors from age 90 who stayed at 90

        age_90_pop = result[result["age"] == 90]["population"].sum()
        assert age_90_pop > 0

    def test_survival_with_config(self, sample_population, sample_survival_rates):
        """Test apply_survival_rates with custom configuration."""
        config = {
            "demographics": {"age_groups": {"max_age": 90}},
            "rates": {"mortality": {"improvement_factor": 0.0}},
            "project": {"base_year": 2025},
        }

        result = apply_survival_rates(
            sample_population, sample_survival_rates, year=2025, config=config
        )

        assert not result.empty

    def test_survival_handles_missing_rates_gracefully(self, sample_population):
        """Test that missing survival rates default to 0."""
        # Only provide rates for some cohorts
        partial_rates = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "survival_rate": [0.999, 0.999],
            }
        )

        result = apply_survival_rates(sample_population, partial_rates, year=2025)

        # Should still produce output (with warnings)
        assert isinstance(result, pd.DataFrame)


class TestApplyMortalityImprovement:
    """Tests for apply_mortality_improvement function."""

    @pytest.fixture
    def base_survival_rates(self):
        """Base survival rates for testing."""
        return pd.DataFrame(
            {
                "age": [0, 25, 50, 75, 90],
                "sex": ["Male"] * 5,
                "race": ["White"] * 5,
                "survival_rate": [0.993, 0.999, 0.998, 0.95, 0.80],
            }
        )

    def test_no_improvement_when_same_year(self, base_survival_rates):
        """Test no improvement when current_year equals base_year."""
        result = apply_mortality_improvement(
            base_survival_rates, current_year=2025, base_year=2025, improvement_factor=0.005
        )

        pd.testing.assert_frame_equal(result, base_survival_rates)

    def test_no_improvement_when_factor_zero(self, base_survival_rates):
        """Test no improvement when improvement_factor is 0."""
        result = apply_mortality_improvement(
            base_survival_rates, current_year=2030, base_year=2025, improvement_factor=0.0
        )

        pd.testing.assert_frame_equal(result, base_survival_rates)

    def test_improvement_increases_survival_rates(self, base_survival_rates):
        """Test that mortality improvement increases survival rates."""
        result = apply_mortality_improvement(
            base_survival_rates, current_year=2035, base_year=2025, improvement_factor=0.005
        )

        # All survival rates should increase (or stay at max 1.0)
        for idx in range(len(base_survival_rates)):
            original = base_survival_rates.iloc[idx]["survival_rate"]
            improved = result.iloc[idx]["survival_rate"]
            assert improved >= original

    def test_improvement_capped_at_one(self):
        """Test that survival rates are capped at 1.0."""
        high_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.9999],
            }
        )

        result = apply_mortality_improvement(
            high_rates, current_year=2100, base_year=2025, improvement_factor=0.01
        )

        assert result["survival_rate"].iloc[0] <= 1.0

    def test_improvement_compounds_correctly(self, base_survival_rates):
        """Test that improvement compounds correctly over years."""
        years_elapsed = 10
        improvement_factor = 0.005

        result = apply_mortality_improvement(
            base_survival_rates,
            current_year=2035,
            base_year=2025,
            improvement_factor=improvement_factor,
        )

        # Check one specific rate
        original_survival = 0.95  # age 75
        original_death_rate = 1 - original_survival
        expected_death_rate = original_death_rate * ((1 - improvement_factor) ** years_elapsed)
        expected_survival = 1 - expected_death_rate

        age_75_result = result[result["age"] == 75]["survival_rate"].iloc[0]
        assert abs(age_75_result - expected_survival) < 0.0001

    def test_improvement_preserves_columns(self, base_survival_rates):
        """Test that improvement preserves non-rate columns."""
        result = apply_mortality_improvement(
            base_survival_rates, current_year=2030, base_year=2025, improvement_factor=0.005
        )

        assert list(result["age"]) == list(base_survival_rates["age"])
        assert list(result["sex"]) == list(base_survival_rates["sex"])
        assert list(result["race"]) == list(base_survival_rates["race"])


class TestValidateSurvivalRates:
    """Tests for validate_survival_rates function."""

    @pytest.fixture
    def valid_survival_rates(self):
        """Valid survival rates for testing."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White"]:
                for age in range(91):
                    if age < 1:
                        rate = 0.993
                    elif age < 65:
                        rate = 0.999
                    else:
                        rate = 0.95 - (age - 65) * 0.003
                    rate = max(rate, 0.5)  # Ensure reasonable rates
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "survival_rate": rate,
                        }
                    )
        return pd.DataFrame(data)

    def test_validate_valid_rates(self, valid_survival_rates):
        """Test validation passes for valid rates."""
        is_valid, issues = validate_survival_rates(valid_survival_rates)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_missing_columns(self):
        """Test validation fails when required columns are missing."""
        invalid_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                # Missing 'race' and 'survival_rate'
            }
        )

        is_valid, issues = validate_survival_rates(invalid_rates)

        assert is_valid is False
        assert any("Missing required columns" in issue for issue in issues)

    def test_validate_negative_rates(self):
        """Test validation catches negative survival rates."""
        rates_with_negative = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "survival_rate": [0.999, -0.05],  # Negative rate
            }
        )

        is_valid, issues = validate_survival_rates(rates_with_negative)

        assert is_valid is False
        assert any("Negative survival rates" in issue for issue in issues)

    def test_validate_rates_over_one(self):
        """Test validation catches survival rates > 1.0."""
        rates_over_one = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "survival_rate": [0.999, 1.05],  # Over 1.0
            }
        )

        is_valid, issues = validate_survival_rates(rates_over_one)

        assert is_valid is False
        assert any("Survival rates > 1.0" in issue for issue in issues)

    def test_validate_low_infant_survival(self):
        """Test validation catches implausibly low infant survival."""
        low_infant_rates = pd.DataFrame(
            {
                "age": [0, 25],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "survival_rate": [0.90, 0.999],  # Low infant survival
            }
        )

        is_valid, issues = validate_survival_rates(low_infant_rates)

        assert is_valid is False
        assert any("Implausibly low infant survival" in issue for issue in issues)

    def test_validate_missing_combinations(self):
        """Test validation catches missing age-sex-race combinations."""
        incomplete_rates = pd.DataFrame(
            {
                "age": [25, 25, 30],  # Missing 30 for Female
                "sex": ["Male", "Female", "Male"],
                "race": ["White", "White", "White"],
                "survival_rate": [0.999, 0.999, 0.999],
            }
        )

        is_valid, issues = validate_survival_rates(incomplete_rates)

        assert is_valid is False
        assert any("Missing age-sex-race combinations" in issue for issue in issues)


class TestCalculateLifeExpectancy:
    """Tests for calculate_life_expectancy function."""

    @pytest.fixture
    def survival_rates_for_life_exp(self):
        """Survival rates for life expectancy calculation."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White"]:
                for age in range(91):
                    # Simplified pattern
                    rate = 0.99 if age < 65 else 0.95
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "survival_rate": rate,
                        }
                    )
        return pd.DataFrame(data)

    def test_calculate_life_expectancy_returns_dataframe(self, survival_rates_for_life_exp):
        """Test that function returns a DataFrame."""
        result = calculate_life_expectancy(survival_rates_for_life_exp)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_life_expectancy_has_correct_columns(self, survival_rates_for_life_exp):
        """Test that result has correct columns."""
        result = calculate_life_expectancy(survival_rates_for_life_exp)

        expected_columns = ["sex", "race", "age", "life_expectancy"]
        assert all(col in result.columns for col in expected_columns)

    def test_life_expectancy_by_sex_and_race(self, survival_rates_for_life_exp):
        """Test that life expectancy is calculated for each sex-race group."""
        result = calculate_life_expectancy(survival_rates_for_life_exp)

        # Should have one row per sex-race combination
        assert len(result) == 2  # Male-White and Female-White

    def test_life_expectancy_positive(self, survival_rates_for_life_exp):
        """Test that life expectancy is positive."""
        result = calculate_life_expectancy(survival_rates_for_life_exp)

        assert (result["life_expectancy"] > 0).all()

    def test_life_expectancy_with_custom_start_age(self, survival_rates_for_life_exp):
        """Test life expectancy calculation from a different starting age."""
        result_from_0 = calculate_life_expectancy(survival_rates_for_life_exp, age_start=0)
        result_from_65 = calculate_life_expectancy(survival_rates_for_life_exp, age_start=65)

        # Life expectancy at 65 should be less than at birth
        le_0 = result_from_0["life_expectancy"].iloc[0]
        le_65 = result_from_65["life_expectancy"].iloc[0]

        assert le_65 < le_0

    def test_life_expectancy_empty_after_filter(self):
        """Test life expectancy with age_start beyond available data."""
        rates = pd.DataFrame(
            {
                "age": [0, 10, 20],
                "sex": ["Male"] * 3,
                "race": ["White"] * 3,
                "survival_rate": [0.99, 0.99, 0.99],
            }
        )

        result = calculate_life_expectancy(rates, age_start=100)

        # Should return empty since no ages >= 100
        assert result.empty


class TestEdgeCases:
    """Test edge cases for mortality module."""

    def test_single_cohort_survival(self):
        """Test survival with a single cohort."""
        single_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        single_rate = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.999],
            }
        )

        result = apply_survival_rates(single_pop, single_rate, year=2025)

        assert len(result) == 1
        assert result["age"].iloc[0] == 26  # Aged by 1 year
        assert abs(result["population"].iloc[0] - 999.0) < 0.1

    def test_zero_survival_rate(self):
        """Test with zero survival rate (all die)."""
        pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        zero_rate = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.0],  # Everyone dies
            }
        )

        result = apply_survival_rates(pop, zero_rate, year=2025)

        assert result["population"].iloc[0] == 0.0

    def test_perfect_survival_rate(self):
        """Test with perfect survival rate (no deaths)."""
        pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        perfect_rate = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [1.0],  # Nobody dies
            }
        )

        result = apply_survival_rates(pop, perfect_rate, year=2025)

        assert result["population"].iloc[0] == 1000.0


@pytest.mark.parametrize(
    "improvement_factor,years,expected_improvement",
    [
        (0.005, 10, 0.049),  # ~5% improvement
        (0.01, 5, 0.049),  # ~5% improvement
        (0.001, 20, 0.020),  # ~2% improvement
    ],
)
def test_mortality_improvement_parametrized(improvement_factor, years, expected_improvement):
    """Parametrized test for mortality improvement."""
    rates = pd.DataFrame(
        {
            "age": [50],
            "sex": ["Male"],
            "race": ["White"],
            "survival_rate": [0.95],  # Death rate = 0.05
        }
    )

    result = apply_mortality_improvement(
        rates,
        current_year=2025 + years,
        base_year=2025,
        improvement_factor=improvement_factor,
    )

    original_death_rate = 0.05
    result_death_rate = 1 - result["survival_rate"].iloc[0]
    actual_improvement = (original_death_rate - result_death_rate) / original_death_rate

    assert abs(actual_improvement - expected_improvement) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
