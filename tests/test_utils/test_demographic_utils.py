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
    sprague_graduate,
    validate_cohort_sums,
)


class TestCreateAgeGroups:
    """Tests for create_age_groups function."""

    @pytest.mark.parametrize(
        "min_age,max_age,group_size,expected",
        [
            (0, 5, 1, ["0", "1", "2", "3", "4", "5+"]),
            (0, 20, 5, ["0-4", "5-9", "10-14", "15-19", "20+"]),
            (0, 50, 10, ["0-9", "10-19", "20-29", "30-39", "40-49", "50+"]),
        ],
        ids=["single-year", "quinquennial", "decennial"],
    )
    def test_age_groups_parametrized(
        self, min_age: int, max_age: int, group_size: int, expected: list[str]
    ) -> None:
        """Parametrized test for different age group configurations."""
        result = create_age_groups(min_age=min_age, max_age=max_age, group_size=group_size)
        assert result == expected

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

    @pytest.mark.parametrize(
        "male_pop,female_pop,expected_ratio",
        [
            (1000, 1000, 100.0),
            (1200, 1000, 120.0),
            (950, 1000, 95.0),
            (500, 500, 100.0),
            (150, 100, 150.0),
        ],
        ids=["equal", "more-males", "more-females", "equal-small", "high-ratio"],
    )
    def test_sex_ratio_parametrized(
        self, male_pop: int, female_pop: int, expected_ratio: float
    ) -> None:
        """Parametrized test for sex ratio calculations."""
        pop = pd.DataFrame(
            {
                "sex": ["Male", "Female"],
                "population": [male_pop, female_pop],
            }
        )
        result = calculate_sex_ratio(pop)
        assert result == expected_ratio

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

    @pytest.mark.parametrize(
        "pop_start,pop_end,years,expected_rate",
        [
            (1000, 1100, 1, 0.10),  # 10% growth in 1 year
            (1000, 900, 1, -0.10),  # -10% growth in 1 year
            (1000, 1000 * (1.10**5), 5, 0.10),  # 10% compound over 5 years
            (1000, 1000, 10, 0.0),  # Zero growth
        ],
        ids=["positive-1yr", "negative-1yr", "compound-5yr", "zero-growth"],
    )
    def test_growth_rate_parametrized(
        self, pop_start: float, pop_end: float, years: int, expected_rate: float
    ) -> None:
        """Parametrized test for growth rate calculations."""
        result = calculate_growth_rate(pop_start=pop_start, pop_end=pop_end, years=years)
        assert abs(result - expected_rate) < 0.001

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


class TestSpragueGraduate:
    """Tests for sprague_graduate function (ADR-048)."""

    @pytest.fixture
    def demographic_totals(self) -> np.ndarray:
        """Realistic 18-group population totals resembling a real age distribution.

        Shaped like a typical population pyramid: larger young-adult groups,
        declining at older ages, with a smaller terminal group.
        """
        return np.array([
            5000, 4800, 4500, 5200, 5800,  # 0-4 through 20-24
            5500, 5100, 4800, 4600, 4400,  # 25-29 through 45-49
            4200, 3900, 3500, 3000, 2500,  # 50-54 through 70-74
            2000, 1500, 1200,              # 75-79, 80-84, 85+
        ])

    @pytest.fixture
    def uniform_totals(self) -> np.ndarray:
        """Uniform group totals (each group = 1000)."""
        return np.full(18, 1000.0)

    def test_output_length(self, demographic_totals: np.ndarray) -> None:
        """Sprague produces 5 * n_groups single-year values."""
        result = sprague_graduate(demographic_totals)
        assert len(result) == 18 * 5  # 90 values

    def test_group_totals_preserved(self, demographic_totals: np.ndarray) -> None:
        """Sum of 5 single-year values within each group equals the group total.

        This is the fundamental property of osculatory interpolation: the
        method produces smooth curves while preserving group totals.
        """
        result = sprague_graduate(demographic_totals)

        for i, group_total in enumerate(demographic_totals):
            single_year_sum = result[i * 5 : (i + 1) * 5].sum()
            assert abs(single_year_sum - group_total) < 0.01, (
                f"Group {i}: expected {group_total}, got {single_year_sum}"
            )

    def test_total_population_preserved(self, demographic_totals: np.ndarray) -> None:
        """Total across all single-year values equals total of all group totals."""
        result = sprague_graduate(demographic_totals)
        assert abs(result.sum() - demographic_totals.sum()) < 0.1

    def test_smoothness_no_step_functions(self, demographic_totals: np.ndarray) -> None:
        """Adjacent single-year values should not have abrupt jumps.

        The whole point of ADR-048 is to eliminate the step-function
        artifacts caused by uniform splitting. With Sprague, adjacent ages
        should have gradual transitions, not the 5-year-boundary jumps
        seen with uniform splitting.
        """
        result = sprague_graduate(demographic_totals)

        # Compute max ratio between adjacent ages (excluding age 0-1 which
        # can have infant mortality effects, and ages near 85 where terminal
        # group handling creates transitions)
        max_ratio = 1.0
        for i in range(5, 80):  # ages 5 through 79
            if result[i] > 0 and result[i - 1] > 0:
                ratio = max(result[i] / result[i - 1], result[i - 1] / result[i])
                max_ratio = max(max_ratio, ratio)

        # Sprague should produce gradual changes; max ratio should be well
        # under 1.20 (20%). Uniform splitting creates 4-5% jumps at every
        # 5-year boundary and 0% change within groups.
        assert max_ratio < 1.20, (
            f"Adjacent-age max ratio {max_ratio:.3f} exceeds 1.20 threshold"
        )

    def test_no_step_at_group_boundaries(self, uniform_totals: np.ndarray) -> None:
        """With uniform group totals, there should be no steps at boundaries.

        If all groups have the same total, uniform splitting gives identical
        values everywhere (no jumps). Sprague should similarly produce a
        smooth result with no boundary artifacts.
        """
        result = sprague_graduate(uniform_totals)

        # At group boundaries (indices 4-5, 9-10, etc.), the ratio should
        # be close to 1.0 for uniform inputs
        for boundary in [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74]:
            if boundary + 1 < len(result) and result[boundary] > 0:
                ratio = result[boundary + 1] / result[boundary]
                assert 0.85 < ratio < 1.15, (
                    f"Step at boundary ages {boundary}-{boundary + 1}: "
                    f"ratio = {ratio:.3f}"
                )

    def test_non_negative_with_clamping(self, demographic_totals: np.ndarray) -> None:
        """With clamp_negatives=True (default), all values should be >= 0."""
        result = sprague_graduate(demographic_totals, clamp_negatives=True)
        assert np.all(result >= 0), "Negative values found despite clamping"

    def test_clamping_preserves_group_totals(self) -> None:
        """When clamping occurs, group totals should still be preserved.

        Sprague can produce small negative overshoots at extreme ages for
        very small populations. After clamping to zero, the group total
        must be renormalized.
        """
        # Create a distribution with a very small terminal group that may
        # trigger negative overshoots
        totals = np.array([
            5000, 4800, 4500, 5200, 5800,
            5500, 5100, 4800, 4600, 4400,
            4200, 3900, 3500, 3000, 2500,
            2000, 1500, 50,  # Very small terminal group
        ])
        result = sprague_graduate(totals, clamp_negatives=True)

        for i, group_total in enumerate(totals):
            single_year_sum = result[i * 5 : (i + 1) * 5].sum()
            assert abs(single_year_sum - group_total) < 0.1, (
                f"Group {i}: expected {group_total}, got {single_year_sum} "
                f"after clamping"
            )

    def test_minimum_groups_required(self) -> None:
        """Sprague requires at least 5 groups."""
        with pytest.raises(ValueError, match="at least 5 groups"):
            sprague_graduate(np.array([100, 200, 300, 400]))

    def test_five_groups_minimum(self) -> None:
        """Sprague works with exactly 5 groups."""
        totals = np.array([1000, 900, 800, 700, 600])
        result = sprague_graduate(totals)
        assert len(result) == 25  # 5 groups * 5 years
        assert abs(result.sum() - totals.sum()) < 0.1

    def test_all_zeros(self) -> None:
        """Sprague handles all-zero input gracefully."""
        totals = np.zeros(18)
        result = sprague_graduate(totals)
        assert np.all(result == 0)

    def test_single_nonzero_group(self) -> None:
        """Sprague handles a single nonzero group among zeros."""
        totals = np.zeros(18)
        totals[5] = 1000.0  # Only group 5 (ages 25-29) has population
        result = sprague_graduate(totals, clamp_negatives=True)

        # Group 5 total should be preserved
        assert abs(result[25:30].sum() - 1000.0) < 0.1

        # All values should be non-negative
        assert np.all(result >= 0)

    def test_monotonically_declining_at_old_ages(
        self, demographic_totals: np.ndarray
    ) -> None:
        """For a typical age distribution, populations should generally
        decline at older ages (60+).

        This is a plausibility check: the Sprague output for ages 60-84
        should show a general downward trend, matching the declining group
        totals at those ages.
        """
        result = sprague_graduate(demographic_totals)

        # Compute 5-year averages for comparison
        avg_60_64 = result[60:65].mean()
        avg_65_69 = result[65:70].mean()
        avg_70_74 = result[70:75].mean()
        avg_75_79 = result[75:80].mean()
        avg_80_84 = result[80:85].mean()

        # Each 5-year average should be less than the previous
        assert avg_65_69 < avg_60_64
        assert avg_70_74 < avg_65_69
        assert avg_75_79 < avg_70_74
        assert avg_80_84 < avg_75_79

    def test_sprague_vs_uniform_smoothness(
        self, demographic_totals: np.ndarray
    ) -> None:
        """Sprague should produce smoother results than uniform splitting.

        Compute the sum of squared second differences (a measure of
        roughness) for both Sprague and uniform, and verify Sprague is
        smoother.
        """
        sprague_result = sprague_graduate(demographic_totals)

        # Uniform splitting for comparison
        uniform_result = np.repeat(demographic_totals / 5.0, 5)

        def roughness(arr: np.ndarray) -> float:
            """Sum of squared second differences (curvature penalty)."""
            d2 = np.diff(arr, n=2)
            return float(np.sum(d2 ** 2))

        sprague_roughness = roughness(sprague_result[5:80])
        uniform_roughness = roughness(uniform_result[5:80])

        # Sprague should be smoother (lower roughness)
        assert sprague_roughness < uniform_roughness, (
            f"Sprague roughness ({sprague_roughness:.1f}) >= "
            f"uniform roughness ({uniform_roughness:.1f})"
        )


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
