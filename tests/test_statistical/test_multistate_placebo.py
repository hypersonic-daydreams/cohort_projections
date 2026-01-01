"""
Unit tests for B2: Multi-State Placebo Analysis.

Tests the functions in module_B2_multistate_placebo/ for:
- Boom category classification
- State shift calculation
- Oil state hypothesis testing
- State ranking and comparison

Per ADR-020 Phase B6, these tests verify the correct implementation
of multi-state comparison methods to test whether North Dakota's
vintage transition patterns are unusual compared to other states.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add module path for imports
MODULE_PATH = (
    Path(__file__).parent.parent.parent
    / "sdc_2024_replication"
    / "scripts"
    / "statistical_analysis"
)
if str(MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH))

from module_B2_multistate_placebo import (
    ALL_BOOM_STATES,
    ALL_OIL_STATES,
    BAKKEN_BOOM_STATES,
    MATURE_OIL_STATES,
    OIL_STATES,
    OTHER_SHALE_STATES,
    PERMIAN_BOOM_STATES,
    SECONDARY_OIL_STATES,
    calculate_all_state_shifts,
    # Shift calculation
    calculate_state_shift,
    compare_boom_categories,
    compare_oil_vs_non_oil,
    # State classification
    get_boom_category,
    get_nd_percentile,
    get_nd_rank_among_boom_states,
    get_nd_rank_among_oil_states,
    rank_states_by_shift,
    run_bakken_specific_hypothesis_test,
    run_boom_state_hypothesis_test,
    # Hypothesis testing
    run_oil_state_hypothesis_test,
)


class TestGetBoomCategory:
    """Tests for get_boom_category() function."""

    @pytest.mark.parametrize(
        "state,expected_category",
        [
            ("North Dakota", "Bakken Boom"),
            ("Montana", "Bakken Boom"),
            ("Texas", "Permian Boom"),
            ("New Mexico", "Permian Boom"),
            ("Colorado", "Other Shale"),
            ("Oklahoma", "Other Shale"),
            ("Louisiana", "Other Shale"),
            ("California", "Mature Oil"),
            ("Alaska", "Mature Oil"),
            ("Wyoming", "Mature Oil"),
            ("Kansas", "Mature Oil"),
            ("Florida", "Non-Oil"),
            ("New York", "Non-Oil"),
            ("Minnesota", "Non-Oil"),
        ],
    )
    def test_boom_category_classification(self, state, expected_category):
        """Test correct classification of states into boom categories."""
        result = get_boom_category(state)
        assert result == expected_category

    def test_all_states_classifiable(self, sample_state_panel):
        """Test that all states in panel can be classified."""
        states = sample_state_panel["state"].unique()
        categories = [get_boom_category(state) for state in states]

        # All should return valid categories
        valid_categories = {
            "Bakken Boom",
            "Permian Boom",
            "Other Shale",
            "Mature Oil",
            "Non-Oil",
        }
        for cat in categories:
            assert cat in valid_categories


class TestOilStateClassification:
    """Tests for oil state classification lists."""

    def test_oil_states_disjoint_from_secondary(self):
        """Test primary and secondary oil states don't overlap."""
        overlap = set(OIL_STATES) & set(SECONDARY_OIL_STATES)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_oil_states_complete(self):
        """Test ALL_OIL_STATES is union of primary and secondary."""
        expected = set(OIL_STATES) | set(SECONDARY_OIL_STATES)
        actual = set(ALL_OIL_STATES)
        assert expected == actual

    def test_north_dakota_in_oil_states(self):
        """Test North Dakota is classified as an oil state."""
        assert "North Dakota" in OIL_STATES
        assert "North Dakota" in ALL_OIL_STATES

    def test_boom_states_include_all_boom_categories(self):
        """Test ALL_BOOM_STATES includes all boom category states."""
        expected = set(BAKKEN_BOOM_STATES) | set(PERMIAN_BOOM_STATES) | set(OTHER_SHALE_STATES)
        actual = set(ALL_BOOM_STATES)
        assert expected == actual

    def test_boom_categories_disjoint(self):
        """Test boom categories don't overlap."""
        all_boom = [BAKKEN_BOOM_STATES, PERMIAN_BOOM_STATES, OTHER_SHALE_STATES, MATURE_OIL_STATES]

        for i, cat1 in enumerate(all_boom):
            for j, cat2 in enumerate(all_boom):
                if i < j:
                    overlap = set(cat1) & set(cat2)
                    assert len(overlap) == 0, f"Overlap between categories: {overlap}"

    def test_north_dakota_in_bakken_boom(self):
        """Test North Dakota is in Bakken Boom category."""
        assert "North Dakota" in BAKKEN_BOOM_STATES


class TestCalculateStateShift:
    """Tests for calculate_state_shift() function."""

    def test_basic_shift_calculation(self, sample_state_panel):
        """Test shift calculation for North Dakota."""
        result = calculate_state_shift(sample_state_panel, "North Dakota")

        assert result is not None
        assert result.state == "North Dakota"
        assert result.shift_magnitude is not None

    def test_shift_result_fields(self, sample_state_panel):
        """Test StateShiftResult has all expected fields."""
        result = calculate_state_shift(sample_state_panel, "Texas")

        assert result.state == "Texas"
        assert result.state_fips is not None
        assert result.mean_2010s is not None
        assert result.mean_2020s is not None
        assert result.shift_magnitude is not None
        assert result.relative_shift is not None
        assert result.n_2010s is not None
        assert result.n_2020s is not None

    def test_shift_magnitude_calculation(self, sample_state_panel):
        """Test shift magnitude is mean_2020s - mean_2010s."""
        result = calculate_state_shift(sample_state_panel, "California")

        expected_shift = result.mean_2020s - result.mean_2010s
        assert abs(result.shift_magnitude - expected_shift) < 0.01

    def test_exclude_2020_parameter(self, sample_state_panel):
        """Test that exclude_2020=True excludes 2020 from post-period."""
        result_exclude = calculate_state_shift(
            sample_state_panel, "North Dakota", exclude_2020=True
        )
        result_include = calculate_state_shift(
            sample_state_panel, "North Dakota", exclude_2020=False
        )

        # Results should differ
        # With 2020 excluded, post-period starts at 2021
        # With 2020 included, post-period includes COVID year
        assert result_exclude.n_2020s < result_include.n_2020s

    def test_state_not_found_raises_error(self, sample_state_panel):
        """Test that non-existent state raises ValueError."""
        with pytest.raises(ValueError, match="No data found"):
            calculate_state_shift(sample_state_panel, "Not A State")

    def test_to_dict_method(self, sample_state_panel):
        """Test to_dict() produces serializable output."""
        result = calculate_state_shift(sample_state_panel, "North Dakota")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "state" in result_dict
        assert "shift_magnitude" in result_dict
        assert "relative_shift" in result_dict


class TestCalculateAllStateShifts:
    """Tests for calculate_all_state_shifts() function."""

    def test_all_states_processed(self, sample_state_panel):
        """Test all states in panel are processed."""
        result = calculate_all_state_shifts(sample_state_panel)

        n_states = sample_state_panel["state"].nunique()
        assert len(result) == n_states

    def test_output_is_dataframe(self, sample_state_panel):
        """Test output is a DataFrame."""
        result = calculate_all_state_shifts(sample_state_panel)

        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sample_state_panel):
        """Test output has expected columns."""
        result = calculate_all_state_shifts(sample_state_panel)

        expected_cols = [
            "state",
            "state_fips",
            "mean_2010s",
            "mean_2020s",
            "shift_magnitude",
            "relative_shift",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_north_dakota_in_results(self, sample_state_panel):
        """Test North Dakota is in results."""
        result = calculate_all_state_shifts(sample_state_panel)

        assert "North Dakota" in result["state"].values


class TestRankStatesByShift:
    """Tests for rank_states_by_shift() function."""

    def test_basic_ranking(self, shift_df_50_states):
        """Test basic ranking functionality."""
        result = rank_states_by_shift(shift_df_50_states)

        assert "rank" in result.columns
        assert "percentile" in result.columns

    def test_rank_values(self, shift_df_50_states):
        """Test rank values are 1 to n."""
        result = rank_states_by_shift(shift_df_50_states)

        ranks = result["rank"].values
        assert min(ranks) == 1
        assert max(ranks) == len(shift_df_50_states)

    def test_descending_order_default(self, shift_df_50_states):
        """Test default ordering is descending (largest first)."""
        result = rank_states_by_shift(shift_df_50_states, metric="shift_magnitude")

        # First row should have highest shift_magnitude
        values = result["shift_magnitude"].values
        assert values[0] >= values[-1]

    def test_ascending_order(self, shift_df_50_states):
        """Test ascending ordering option."""
        result = rank_states_by_shift(shift_df_50_states, metric="shift_magnitude", ascending=True)

        # First row should have lowest shift_magnitude
        values = result["shift_magnitude"].values
        assert values[0] <= values[-1]

    @pytest.mark.parametrize("metric", ["shift_magnitude", "relative_shift", "cohens_d"])
    def test_ranking_metrics(self, shift_df_50_states, metric):
        """Test ranking by different metrics."""
        result = rank_states_by_shift(shift_df_50_states, metric=metric)

        assert "rank" in result.columns


class TestGetNDPercentile:
    """Tests for get_nd_percentile() function."""

    def test_basic_percentile_calculation(self, shift_df_50_states):
        """Test percentile calculation for North Dakota."""
        result = get_nd_percentile(shift_df_50_states)

        assert result is not None
        assert "percentile" in result
        assert "rank" in result
        assert result["state"] == "North Dakota"

    def test_percentile_range(self, shift_df_50_states):
        """Test percentile is in valid range [0, 100]."""
        result = get_nd_percentile(shift_df_50_states)

        assert 0 <= result["percentile"] <= 100
        assert 0 <= result["percentile_from_top"] <= 100

    def test_interpretation_provided(self, shift_df_50_states):
        """Test interpretation string is provided."""
        result = get_nd_percentile(shift_df_50_states)

        assert "interpretation" in result
        assert len(result["interpretation"]) > 0

    def test_nd_not_found_error(self, sample_state_panel):
        """Test handling when ND not in data."""
        # Create data without ND
        df = sample_state_panel[sample_state_panel["state"] != "North Dakota"]
        shift_df = calculate_all_state_shifts(df)

        result = get_nd_percentile(shift_df)

        assert "error" in result


class TestOilStateHypothesis:
    """Tests for run_oil_state_hypothesis_test() function."""

    def test_basic_hypothesis_test(self, shift_df_50_states):
        """Test basic oil state hypothesis test."""
        result = run_oil_state_hypothesis_test(shift_df_50_states)

        assert result is not None
        assert "oil_mean" in result
        assert "non_oil_mean" in result
        assert "difference" in result

    def test_t_test_results(self, shift_df_50_states):
        """Test t-test results are included."""
        result = run_oil_state_hypothesis_test(shift_df_50_states)

        assert "t_test" in result
        assert "statistic" in result["t_test"]
        assert "p_value" in result["t_test"]

    def test_mann_whitney_results(self, shift_df_50_states):
        """Test Mann-Whitney results are included."""
        result = run_oil_state_hypothesis_test(shift_df_50_states)

        assert "mann_whitney" in result

    def test_custom_oil_states_list(self, shift_df_50_states):
        """Test using custom oil states list."""
        custom_oil = ["Texas", "North Dakota", "Oklahoma"]
        result = run_oil_state_hypothesis_test(shift_df_50_states, oil_states=custom_oil)

        assert result["oil_states_used"] == custom_oil
        assert result["n_oil_states"] == 3

    def test_interpretation_provided(self, shift_df_50_states):
        """Test interpretation string is provided."""
        result = run_oil_state_hypothesis_test(shift_df_50_states)

        assert "interpretation" in result


class TestCompareOilVsNonOil:
    """Tests for compare_oil_vs_non_oil() function."""

    def test_returns_dataframe(self, shift_df_50_states):
        """Test function returns a DataFrame."""
        result = compare_oil_vs_non_oil(shift_df_50_states)

        assert isinstance(result, pd.DataFrame)

    def test_oil_category_column(self, shift_df_50_states):
        """Test oil_category column is created."""
        result = compare_oil_vs_non_oil(shift_df_50_states)

        assert "oil_category" in result.columns

    def test_categories_present(self, shift_df_50_states):
        """Test expected categories are present."""
        result = compare_oil_vs_non_oil(shift_df_50_states)

        categories = result["oil_category"].values
        expected = {"Primary Oil", "Secondary Oil", "Non-Oil"}

        for cat in categories:
            assert cat in expected


class TestGetNDRankAmongOilStates:
    """Tests for get_nd_rank_among_oil_states() function."""

    def test_basic_ranking(self, shift_df_50_states):
        """Test basic ranking among oil states."""
        result = get_nd_rank_among_oil_states(shift_df_50_states)

        assert result is not None
        assert "rank_among_oil" in result
        assert "n_oil_states" in result
        assert result["state"] == "North Dakota"

    def test_rank_within_range(self, shift_df_50_states):
        """Test rank is within valid range."""
        result = get_nd_rank_among_oil_states(shift_df_50_states)

        assert 1 <= result["rank_among_oil"] <= result["n_oil_states"]

    def test_percentile_among_oil(self, shift_df_50_states):
        """Test percentile among oil states is computed."""
        result = get_nd_rank_among_oil_states(shift_df_50_states)

        assert "percentile_among_oil" in result
        assert 0 <= result["percentile_among_oil"] <= 100

    def test_all_oil_rankings_included(self, shift_df_50_states):
        """Test full rankings list is included."""
        result = get_nd_rank_among_oil_states(shift_df_50_states)

        assert "all_oil_rankings" in result
        assert isinstance(result["all_oil_rankings"], list)


class TestBoomStateHypothesis:
    """Tests for run_boom_state_hypothesis_test() function."""

    def test_basic_hypothesis_test(self, shift_df_50_states):
        """Test basic boom state hypothesis test."""
        result = run_boom_state_hypothesis_test(shift_df_50_states)

        assert result is not None
        assert "boom_mean" in result
        assert "non_oil_mean" in result
        assert "difference" in result

    def test_classification_is_boom_timing(self, shift_df_50_states):
        """Test classification method is boom_timing."""
        result = run_boom_state_hypothesis_test(shift_df_50_states)

        assert result["classification"] == "boom_timing"

    def test_boom_states_used(self, shift_df_50_states):
        """Test boom states list is included."""
        result = run_boom_state_hypothesis_test(shift_df_50_states)

        assert "boom_states_used" in result
        assert set(result["boom_states_used"]) == set(ALL_BOOM_STATES)


class TestBakkenSpecificHypothesis:
    """Tests for run_bakken_specific_hypothesis_test() function."""

    def test_basic_comparison(self, shift_df_50_states):
        """Test basic Bakken-specific hypothesis test."""
        result = run_bakken_specific_hypothesis_test(shift_df_50_states)

        assert result is not None
        assert "bakken_vs_permian" in result
        assert "group_statistics" in result

    def test_group_statistics(self, shift_df_50_states):
        """Test group statistics for all categories."""
        result = run_bakken_specific_hypothesis_test(shift_df_50_states)

        groups = result["group_statistics"]
        expected_groups = {
            "Bakken Boom",
            "Permian Boom",
            "Other Shale",
            "Mature Oil",
            "Non-Oil",
        }

        for group in expected_groups:
            assert group in groups

    def test_nd_value_included(self, shift_df_50_states):
        """Test ND-specific value is included."""
        result = run_bakken_specific_hypothesis_test(shift_df_50_states)

        assert "nd_value" in result

    def test_bakken_vs_permian_comparison(self, shift_df_50_states):
        """Test Bakken vs Permian comparison is included."""
        result = run_bakken_specific_hypothesis_test(shift_df_50_states)

        comparison = result["bakken_vs_permian"]
        assert "bakken_mean" in comparison
        assert "permian_mean" in comparison
        assert "difference" in comparison


class TestGetNDRankAmongBoomStates:
    """Tests for get_nd_rank_among_boom_states() function."""

    def test_basic_ranking(self, shift_df_50_states):
        """Test basic ranking among boom states."""
        result = get_nd_rank_among_boom_states(shift_df_50_states)

        assert result is not None
        assert "rank_among_boom" in result
        assert "n_boom_states" in result
        assert result["state"] == "North Dakota"

    def test_rank_within_range(self, shift_df_50_states):
        """Test rank is within valid range."""
        result = get_nd_rank_among_boom_states(shift_df_50_states)

        assert 1 <= result["rank_among_boom"] <= result["n_boom_states"]

    def test_all_boom_rankings_included(self, shift_df_50_states):
        """Test full boom rankings list is included."""
        result = get_nd_rank_among_boom_states(shift_df_50_states)

        assert "all_boom_rankings" in result
        assert isinstance(result["all_boom_rankings"], list)

    def test_boom_category_in_rankings(self, shift_df_50_states):
        """Test boom category is in rankings list entries."""
        result = get_nd_rank_among_boom_states(shift_df_50_states)

        for entry in result["all_boom_rankings"]:
            assert "boom_category" in entry


class TestCompareBoomCategories:
    """Tests for compare_boom_categories() function."""

    def test_returns_dataframe(self, shift_df_50_states):
        """Test function returns a DataFrame."""
        result = compare_boom_categories(shift_df_50_states)

        assert isinstance(result, pd.DataFrame)

    def test_all_categories_present(self, shift_df_50_states):
        """Test all boom categories are present."""
        result = compare_boom_categories(shift_df_50_states)

        expected = {
            "Bakken Boom",
            "Permian Boom",
            "Other Shale",
            "Mature Oil",
            "Non-Oil",
        }
        actual = set(result["boom_category"].values)

        assert expected == actual

    def test_summary_statistics(self, shift_df_50_states):
        """Test summary statistics are computed."""
        result = compare_boom_categories(shift_df_50_states)

        expected_cols = [
            "n_states",
            "mean_shift_magnitude",
            "mean_relative_shift",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_category_ordering(self, shift_df_50_states):
        """Test categories are in expected order."""
        result = compare_boom_categories(shift_df_50_states)

        categories = result["boom_category"].tolist()

        # First should be Bakken Boom (most relevant for ND analysis)
        assert categories[0] == "Bakken Boom"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_oil_states_list(self, shift_df_50_states):
        """Test with empty oil states list."""
        result = run_oil_state_hypothesis_test(shift_df_50_states, oil_states=[])

        assert result["n_oil_states"] == 0

    def test_single_state_group(self, shift_df_50_states):
        """Test with single state in a group."""
        single_state = ["North Dakota"]
        result = run_oil_state_hypothesis_test(shift_df_50_states, oil_states=single_state)

        assert result["n_oil_states"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
