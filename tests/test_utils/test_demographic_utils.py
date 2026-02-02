"""
Unit tests for the demographic_utils module.

Tests demographic calculation functions and helper utilities.
"""

import numpy as np
import pandas as pd
import pytest

from cohort_projections.utils.demographic_utils import (
    aggregate_race_categories,
    calculate_dependency_ratio,
    calculate_growth_rate,
    calculate_median_age,
    calculate_sex_ratio,
    create_age_groups,
    interpolate_missing_ages,
    validate_cohort_sums,
)


class TestCreateAgeGroups:
    """Tests for create_age_groups function."""

    def test_single_year_age_groups(self) -> None:
        """Test creating single-year age groups."""
        result = create_age_groups(min_age=0, max_age=5, group_size=1)

        assert result == ["0", "1", "2", "3", "4", "5+"]

    def test_quinquennial_age_groups(self) -> None:
        """Test creating 5-year age groups."""
        result = create_age_groups(min_age=0, max_age=20, group_size=5)

        assert result == ["0-4", "5-9", "10-14", "15-19", "20+"]

    def test_ten_year_age_groups(self) -> None:
        """Test creating 10-year age groups."""
        result = create_age_groups(min_age=0, max_age=50, group_size=10)

        assert result == ["0-9", "10-19", "20-29", "30-39", "40-49", "50+"]

    def test_default_parameters(self) -> None:
        """Test with default parameters (0 to 90, single year)."""
        result = create_age_groups()

        assert result[0] == "0"
        assert result[-1] == "90+"
        assert len(result) == 91

    def test_open_ended_final_group(self) -> None:
        """Test that final group is always open-ended."""
        result = create_age_groups(min_age=0, max_age=85, group_size=5)

        assert result[-1].endswith("+")
        assert "85+" in result

    def test_custom_range(self) -> None:
        """Test with custom min and max ages."""
        result = create_age_groups(min_age=18, max_age=65, group_size=1)

        assert result[0] == "18"
        assert result[-1] == "65+"


class TestCalculateSexRatio:
    """Tests for calculate_sex_ratio function."""

    def test_equal_populations(self) -> None:
        """Test sex ratio with equal male and female populations."""
        pop = pd.DataFrame(
            {
                "sex": ["Male", "Female"],
                "population": [1000, 1000],
            }
        )

        result = calculate_sex_ratio(pop)

        assert result == 100.0

    def test_more_males(self) -> None:
        """Test sex ratio with more males than females."""
        pop = pd.DataFrame(
            {
                "sex": ["Male", "Female"],
                "population": [1200, 1000],
            }
        )

        result = calculate_sex_ratio(pop)

        assert result == 120.0

    def test_more_females(self) -> None:
        """Test sex ratio with more females than males."""
        pop = pd.DataFrame(
            {
                "sex": ["Male", "Female"],
                "population": [950, 1000],
            }
        )

        result = calculate_sex_ratio(pop)

        assert result == 95.0

    def test_zero_female_population(self) -> None:
        """Test sex ratio with zero female population returns NaN."""
        pop = pd.DataFrame(
            {
                "sex": ["Male", "Female"],
                "population": [1000, 0],
            }
        )

        result = calculate_sex_ratio(pop)

        assert np.isnan(result)

    def test_multiple_age_groups(self) -> None:
        """Test sex ratio calculation sums across age groups."""
        pop = pd.DataFrame(
            {
                "sex": ["Male", "Male", "Female", "Female"],
                "age": [20, 30, 20, 30],
                "population": [500, 500, 400, 600],
            }
        )

        result = calculate_sex_ratio(pop)

        # 1000 males / 1000 females * 100 = 100
        assert result == 100.0


class TestCalculateDependencyRatio:
    """Tests for calculate_dependency_ratio function."""

    def test_balanced_population(self) -> None:
        """Test dependency ratios with balanced age distribution."""
        data = []
        for age in range(91):
            data.append({"age": age, "population": 100})
        pop = pd.DataFrame(data)

        result = calculate_dependency_ratio(pop)

        assert "youth_dependency" in result
        assert "old_age_dependency" in result
        assert "total_dependency" in result

        # Youth (0-17): 18 ages * 100 = 1800
        # Working (18-64): 47 ages * 100 = 4700
        # Elderly (65+): 26 ages * 100 = 2600
        expected_youth = (1800 / 4700) * 100
        expected_elderly = (2600 / 4700) * 100
        expected_total = ((1800 + 2600) / 4700) * 100

        assert abs(result["youth_dependency"] - expected_youth) < 0.1
        assert abs(result["old_age_dependency"] - expected_elderly) < 0.1
        assert abs(result["total_dependency"] - expected_total) < 0.1

    def test_young_population(self) -> None:
        """Test dependency ratios with young population."""
        data = []
        # Mostly young people
        for age in range(18):
            data.append({"age": age, "population": 500})
        for age in range(18, 65):
            data.append({"age": age, "population": 100})
        for age in range(65, 91):
            data.append({"age": age, "population": 10})
        pop = pd.DataFrame(data)

        result = calculate_dependency_ratio(pop)

        # Youth dependency should be high
        assert result["youth_dependency"] > result["old_age_dependency"]

    def test_aging_population(self) -> None:
        """Test dependency ratios with aging population."""
        data = []
        # Mostly elderly people
        for age in range(18):
            data.append({"age": age, "population": 50})
        for age in range(18, 65):
            data.append({"age": age, "population": 100})
        for age in range(65, 91):
            data.append({"age": age, "population": 300})
        pop = pd.DataFrame(data)

        result = calculate_dependency_ratio(pop)

        # Old-age dependency should be high
        assert result["old_age_dependency"] > result["youth_dependency"]

    def test_zero_working_population(self) -> None:
        """Test dependency ratios with zero working-age population."""
        pop = pd.DataFrame(
            {
                "age": [10, 70],
                "population": [1000, 1000],
            }
        )

        result = calculate_dependency_ratio(pop)

        assert np.isnan(result["youth_dependency"])
        assert np.isnan(result["old_age_dependency"])
        assert np.isnan(result["total_dependency"])


class TestCalculateMedianAge:
    """Tests for calculate_median_age function."""

    def test_symmetric_distribution(self) -> None:
        """Test median age with symmetric distribution."""
        pop = pd.DataFrame(
            {
                "age": [20, 40, 60],
                "population": [100, 100, 100],
            }
        )

        result = calculate_median_age(pop)

        assert result == 40.0

    def test_left_skewed_distribution(self) -> None:
        """Test median age with young population."""
        pop = pd.DataFrame(
            {
                "age": [10, 20, 30, 40, 50],
                "population": [500, 300, 100, 50, 50],
            }
        )

        result = calculate_median_age(pop)

        # Median should be toward younger ages
        assert result < 25

    def test_right_skewed_distribution(self) -> None:
        """Test median age with aging population."""
        pop = pd.DataFrame(
            {
                "age": [20, 40, 60, 70, 80],
                "population": [50, 100, 200, 300, 350],
            }
        )

        result = calculate_median_age(pop)

        # Median should be toward older ages
        assert result > 60

    def test_single_age_group(self) -> None:
        """Test median with single age group."""
        pop = pd.DataFrame(
            {
                "age": [35],
                "population": [1000],
            }
        )

        result = calculate_median_age(pop)

        assert result == 35.0

    def test_empty_population(self) -> None:
        """Test median age with empty population."""
        pop = pd.DataFrame(
            {
                "age": [25, 35],
                "population": [0, 0],
            }
        )

        result = calculate_median_age(pop)

        assert np.isnan(result)

    def test_handles_open_ended_age_groups(self) -> None:
        """Test median age handles '90+' style age groups."""
        pop = pd.DataFrame(
            {
                "age": [85, "90+"],
                "population": [100, 50],
            }
        )

        result = calculate_median_age(pop)

        # Should handle the "90+" by treating it as 90
        assert result >= 85
        assert result <= 90


class TestInterpolateMissingAges:
    """Tests for interpolate_missing_ages function."""

    def test_no_missing_values(self) -> None:
        """Test interpolation with no missing values."""
        pop = pd.DataFrame(
            {
                "age": [0, 1, 2, 3, 4],
                "population": [100, 110, 120, 130, 140],
            }
        )

        result = interpolate_missing_ages(pop)

        pd.testing.assert_series_equal(result["population"], pop["population"], check_names=False)

    def test_interpolates_missing_middle_values(self) -> None:
        """Test interpolation of missing middle values."""
        pop = pd.DataFrame(
            {
                "age": [0, 1, 2, 3, 4],
                "population": [100.0, np.nan, np.nan, np.nan, 140.0],
            }
        )

        result = interpolate_missing_ages(pop)

        # Linear interpolation should fill in 110, 120, 130
        assert abs(result.iloc[1]["population"] - 110.0) < 0.1
        assert abs(result.iloc[2]["population"] - 120.0) < 0.1
        assert abs(result.iloc[3]["population"] - 130.0) < 0.1

    def test_handles_open_ended_age_groups(self) -> None:
        """Test interpolation handles '90+' style ages."""
        pop = pd.DataFrame(
            {
                "age": ["85", "86", "87", "88", "90+"],
                "population": [100.0, np.nan, np.nan, np.nan, 60.0],
            }
        )

        result = interpolate_missing_ages(pop)

        # Should not raise and should produce interpolated values
        assert not result["population"].isna().any()

    def test_preserves_original_values(self) -> None:
        """Test that original values are preserved."""
        pop = pd.DataFrame(
            {
                "age": [0, 1, 2],
                "population": [100.0, np.nan, 200.0],
            }
        )

        result = interpolate_missing_ages(pop)

        # Original values should be preserved
        assert result.iloc[0]["population"] == 100.0
        assert result.iloc[2]["population"] == 200.0


class TestAggregateRaceCategories:
    """Tests for aggregate_race_categories function."""

    def test_default_aggregation_map(self) -> None:
        """Test with default aggregation mapping."""
        pop = pd.DataFrame(
            {
                "race": [
                    "White alone, Non-Hispanic",
                    "Black alone, Non-Hispanic",
                    "Hispanic (any race)",
                ],
                "age": [25, 25, 25],
                "sex": ["Male", "Male", "Male"],
                "population": [1000, 500, 750],
            }
        )

        result = aggregate_race_categories(pop)

        assert "race_aggregated" in result.columns
        assert "White NH" in result["race_aggregated"].values

    def test_custom_aggregation_map(self) -> None:
        """Test with custom aggregation mapping."""
        pop = pd.DataFrame(
            {
                "race": ["White", "Black", "Asian"],
                "population": [1000, 500, 300],
            }
        )

        custom_map = {
            "White": "Non-minority",
            "Black": "Minority",
            "Asian": "Minority",
        }

        result = aggregate_race_categories(pop, aggregation_map=custom_map)

        # Should have two aggregated groups
        assert len(result) == 2
        # Minority should sum Black + Asian
        minority_pop = result[result["race_aggregated"] == "Minority"]["population"].iloc[0]
        assert minority_pop == 800

    def test_preserves_other_columns(self) -> None:
        """Test that other demographic columns are preserved."""
        pop = pd.DataFrame(
            {
                "race": ["White alone, Non-Hispanic", "White alone, Non-Hispanic"],
                "age": [25, 30],
                "sex": ["Male", "Male"],
                "population": [100, 200],
            }
        )

        result = aggregate_race_categories(pop)

        # Should preserve age and sex columns
        assert "age" in result.columns
        assert "sex" in result.columns


class TestCalculateGrowthRate:
    """Tests for calculate_growth_rate function."""

    def test_positive_growth(self) -> None:
        """Test positive growth rate calculation."""
        result = calculate_growth_rate(pop_start=1000, pop_end=1100, years=1)

        # 10% growth
        assert abs(result - 0.10) < 0.001

    def test_negative_growth(self) -> None:
        """Test negative growth rate calculation."""
        result = calculate_growth_rate(pop_start=1000, pop_end=900, years=1)

        # -10% growth
        assert abs(result - (-0.10)) < 0.001

    def test_multi_year_growth(self) -> None:
        """Test compound annual growth rate over multiple years."""
        # 10% compound annual growth for 5 years
        pop_end = 1000 * (1.10**5)
        result = calculate_growth_rate(pop_start=1000, pop_end=pop_end, years=5)

        assert abs(result - 0.10) < 0.001

    def test_zero_growth(self) -> None:
        """Test zero growth rate."""
        result = calculate_growth_rate(pop_start=1000, pop_end=1000, years=10)

        assert result == 0.0

    def test_zero_start_population(self) -> None:
        """Test growth rate with zero starting population."""
        result = calculate_growth_rate(pop_start=0, pop_end=1000, years=10)

        assert np.isnan(result)

    def test_zero_years(self) -> None:
        """Test growth rate with zero years."""
        result = calculate_growth_rate(pop_start=1000, pop_end=1500, years=0)

        assert np.isnan(result)

    def test_negative_start_population(self) -> None:
        """Test growth rate with negative starting population."""
        result = calculate_growth_rate(pop_start=-100, pop_end=1000, years=10)

        assert np.isnan(result)


class TestValidateCohortSums:
    """Tests for validate_cohort_sums function."""

    def test_valid_sum(self) -> None:
        """Test validation passes when sums match."""
        cohorts = pd.DataFrame(
            {
                "age": [0, 1, 2, 3, 4],
                "population": [200, 200, 200, 200, 200],
            }
        )

        is_valid, error = validate_cohort_sums(cohorts, total=1000, tolerance=0.01)

        assert is_valid == True  # noqa: E712
        assert error == 0.0

    def test_within_tolerance(self) -> None:
        """Test validation passes within tolerance."""
        cohorts = pd.DataFrame(
            {
                "age": [0, 1, 2],
                "population": [333, 333, 333],  # Sums to 999, not 1000
            }
        )

        is_valid, error = validate_cohort_sums(cohorts, total=1000, tolerance=0.01)

        assert is_valid == True  # noqa: E712
        assert error < 0.01

    def test_exceeds_tolerance(self) -> None:
        """Test validation fails when exceeding tolerance."""
        cohorts = pd.DataFrame(
            {
                "age": [0, 1, 2],
                "population": [300, 300, 300],  # Sums to 900, not 1000
            }
        )

        is_valid, error = validate_cohort_sums(cohorts, total=1000, tolerance=0.01)

        assert is_valid == False  # noqa: E712
        assert error > 0.01

    def test_exact_match(self) -> None:
        """Test validation with exact match."""
        cohorts = pd.DataFrame(
            {
                "age": [0, 1],
                "population": [500, 500],
            }
        )

        is_valid, error = validate_cohort_sums(cohorts, total=1000, tolerance=0.0)

        assert is_valid == True  # noqa: E712
        assert error == 0.0

    def test_zero_total(self) -> None:
        """Test validation with zero total returns NaN error."""
        cohorts = pd.DataFrame(
            {
                "age": [0, 1],
                "population": [100, 100],
            }
        )

        is_valid, error = validate_cohort_sums(cohorts, total=0, tolerance=0.01)

        assert np.isnan(error)

    def test_custom_tolerance(self) -> None:
        """Test validation with custom tolerance."""
        cohorts = pd.DataFrame(
            {
                "age": [0, 1],
                "population": [480, 480],  # 4% under
            }
        )

        # Should fail with 1% tolerance
        is_valid_strict, _ = validate_cohort_sums(cohorts, total=1000, tolerance=0.01)
        assert is_valid_strict == False  # noqa: E712

        # Should pass with 10% tolerance
        is_valid_loose, _ = validate_cohort_sums(cohorts, total=1000, tolerance=0.10)
        assert is_valid_loose == True  # noqa: E712


class TestEdgeCases:
    """Edge case tests for demographic utilities."""

    def test_empty_dataframe_sex_ratio(self) -> None:
        """Test sex ratio with empty DataFrame."""
        pop = pd.DataFrame(columns=["sex", "population"])

        result = calculate_sex_ratio(pop)

        assert np.isnan(result)

    def test_age_groups_edge_alignment(self) -> None:
        """Test age groups when max_age aligns with group_size."""
        result = create_age_groups(min_age=0, max_age=25, group_size=5)

        # Should include 25+ as final group
        assert "25+" in result

    def test_single_person_median_age(self) -> None:
        """Test median age with single person."""
        pop = pd.DataFrame(
            {
                "age": [42],
                "population": [1],
            }
        )

        result = calculate_median_age(pop)

        assert result == 42.0

    def test_growth_rate_doubling(self) -> None:
        """Test growth rate for population doubling."""
        # Population doubles in ~10 years at ~7% growth
        result = calculate_growth_rate(pop_start=1000, pop_end=2000, years=10)

        # Should be approximately 0.0718 (rule of 70)
        assert abs(result - 0.0718) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
