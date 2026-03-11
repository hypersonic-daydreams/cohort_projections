"""Tests for Module 1: Forecast Accuracy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.forecast_accuracy import (
    ForecastAccuracyModule,
    _resolve_county_group,
    _validate_dataframe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COUNTY_GROUPS: dict[str, list[str]] = {
    "bakken": ["38105", "38053"],
    "urban_college": ["38017", "38035"],
    "reservation": ["38005"],
}


@pytest.fixture()
def county_groups() -> dict[str, list[str]]:
    """Standard county-group mapping for tests."""
    return COUNTY_GROUPS


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small synthetic projection-result DataFrame.

    Contains 3 counties, 2 horizons, total-population rows plus one
    age-group slice for age_group="0-4".
    """
    records = []
    counties = [
        ("38017", "county"),  # urban_college
        ("38105", "county"),  # bakken
        ("38099", "county"),  # rural (not in any group)
    ]
    for fips, geo_type in counties:
        for horizon in [5, 10]:
            base = 10000.0 + int(fips[-3:]) * 10
            actual = base * (1 + 0.01 * horizon)
            # Introduce a known offset: projected over-estimates by 2%
            projected = actual * 1.02
            records.append(
                {
                    "run_id": "run1",
                    "geography": fips,
                    "geography_type": geo_type,
                    "year": 2020 + horizon,
                    "horizon": horizon,
                    "sex": "total",
                    "age_group": "total",
                    "target": "population",
                    "projected_value": projected,
                    "actual_value": actual,
                    "base_value": base,
                }
            )
            # Age-group row for 0-4
            age_actual = actual * 0.06
            age_projected = age_actual * 1.03
            records.append(
                {
                    "run_id": "run1",
                    "geography": fips,
                    "geography_type": geo_type,
                    "year": 2020 + horizon,
                    "horizon": horizon,
                    "sex": "total",
                    "age_group": "0-4",
                    "target": "population",
                    "projected_value": age_projected,
                    "actual_value": age_actual,
                    "base_value": base * 0.06,
                }
            )
    # Add a state-level total row
    for horizon in [5, 10]:
        state_actual = sum(
            (10000.0 + int(f[-3:]) * 10) * (1 + 0.01 * horizon)
            for f, _ in counties
        )
        records.append(
            {
                "run_id": "run1",
                "geography": "state",
                "geography_type": "state",
                "year": 2020 + horizon,
                "horizon": horizon,
                "sex": "total",
                "age_group": "total",
                "target": "population",
                "projected_value": state_actual * 1.015,
                "actual_value": state_actual,
                "base_value": state_actual / (1 + 0.01 * horizon),
            }
        )
    return pd.DataFrame(records)


@pytest.fixture()
def module(county_groups: dict[str, list[str]]) -> ForecastAccuracyModule:
    """Instantiate the module with test county groups."""
    return ForecastAccuracyModule(county_groups=county_groups)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for input validation helpers."""

    def test_validate_dataframe_passes_with_all_columns(
        self, sample_df: pd.DataFrame
    ) -> None:
        _validate_dataframe(sample_df)  # should not raise

    def test_validate_dataframe_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"run_id": [1], "geography": ["x"]})
        with pytest.raises(ValueError, match="missing columns"):
            _validate_dataframe(df)

    def test_resolve_county_group_known(self) -> None:
        assert _resolve_county_group("38017", COUNTY_GROUPS) == "urban_college"

    def test_resolve_county_group_rural_fallback(self) -> None:
        assert _resolve_county_group("39999", COUNTY_GROUPS) == "rural"


# ---------------------------------------------------------------------------
# accuracy_by_geography_horizon
# ---------------------------------------------------------------------------


class TestAccuracyByGeographyHorizon:
    """Tests for county/state accuracy by horizon."""

    def test_returns_dataframe(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.accuracy_by_geography_horizon(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_diagnostic_columns(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.accuracy_by_geography_horizon(sample_df)
        expected_cols = {
            "run_id",
            "metric_name",
            "metric_group",
            "geography",
            "geography_group",
            "target",
            "horizon",
            "value",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_mape_approximately_2_percent(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        """Projected = actual * 1.02, so MAPE should be ~2%."""
        result = module.accuracy_by_geography_horizon(sample_df)
        county_mape = result[
            (result["metric_name"] == "mape")
            & (result["geography_type"] != "state" if "geography_type" in result.columns else True)
            & (result["geography"] != "state")
        ]
        assert all(np.isclose(county_mape["value"], 2.0, atol=0.1))

    def test_includes_state_rows(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.accuracy_by_geography_horizon(sample_df)
        state_rows = result[result["geography"] == "state"]
        assert len(state_rows) > 0
        assert all(state_rows["geography_group"] == "state")

    def test_county_group_assignment(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.accuracy_by_geography_horizon(sample_df)
        bakken = result[result["geography"] == "38105"]
        assert all(bakken["geography_group"] == "bakken")
        rural = result[result["geography"] == "38099"]
        assert all(rural["geography_group"] == "rural")


# ---------------------------------------------------------------------------
# accuracy_by_age_group
# ---------------------------------------------------------------------------


class TestAccuracyByAgeGroup:
    """Tests for age-stratified accuracy."""

    def test_returns_age_labelled_metrics(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.accuracy_by_age_group(sample_df)
        assert len(result) > 0
        # Metric names should be suffixed with the age group
        assert all("__age_" in mn for mn in result["metric_name"])

    def test_mape_for_age_group_approximately_3_percent(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        """Age 0-4 projected = actual * 1.03, so MAPE ~ 3%."""
        result = module.accuracy_by_age_group(sample_df)
        age_mape = result[result["metric_name"] == "mape__age_0-4"]
        assert len(age_mape) > 0
        assert all(np.isclose(age_mape["value"], 3.0, atol=0.2))


# ---------------------------------------------------------------------------
# bias_summary
# ---------------------------------------------------------------------------


class TestBiasSummary:
    """Tests for signed-error bias metrics."""

    def test_positive_bias(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        """Projected > actual everywhere, so MSE should be positive."""
        result = module.bias_summary(sample_df)
        mse_rows = result[result["metric_name"] == "mean_signed_error"]
        assert all(mse_rows["value"] > 0)

    def test_mspe_approximately_2_percent(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.bias_summary(sample_df)
        county_mspe = result[
            (result["metric_name"] == "mean_signed_percentage_error")
            & (result["geography"] != "state")
        ]
        # +2% overprojection
        assert all(np.isclose(county_mspe["value"], 2.0, atol=0.1))


# ---------------------------------------------------------------------------
# rank_direction_tests
# ---------------------------------------------------------------------------


class TestRankDirectionTests:
    """Tests for Spearman, directional accuracy, decile capture."""

    def test_directional_accuracy_perfect_when_all_growing(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        """All counties grow in both projected and actual, so DA = 1.0."""
        result = module.rank_direction_tests(sample_df)
        da = result[result["metric_name"] == "directional_accuracy"]
        assert len(da) > 0
        assert all(np.isclose(da["value"], 1.0))

    def test_contains_all_rank_metrics(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.rank_direction_tests(sample_df)
        names = set(result["metric_name"])
        assert "spearman_rank_correlation" in names
        assert "directional_accuracy" in names
        assert "top_decile_capture" in names
        assert "bottom_decile_capture" in names


# ---------------------------------------------------------------------------
# weighted_vs_unweighted_comparison
# ---------------------------------------------------------------------------


class TestWeightedVsUnweighted:
    """Tests for WAPE vs MAPE comparison."""

    def test_returns_three_metrics_per_horizon(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.weighted_vs_unweighted_comparison(sample_df)
        horizons = result["horizon"].unique()
        for h in horizons:
            h_rows = result[result["horizon"] == h]
            names = set(h_rows["metric_name"])
            assert names == {"mape", "wape", "wape_mape_ratio"}

    def test_ratio_close_to_one_for_uniform_error(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        """When all counties have the same 2% error, ratio ~ 1."""
        result = module.weighted_vs_unweighted_comparison(sample_df)
        ratios = result[result["metric_name"] == "wape_mape_ratio"]
        assert all(np.isclose(ratios["value"], 1.0, atol=0.05))


# ---------------------------------------------------------------------------
# compute_all_metrics (integration)
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    """Integration test for the top-level entry point."""

    def test_returns_nonempty_dataframe(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.compute_all_metrics(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_all_rows_have_run_id(
        self, module: ForecastAccuracyModule, sample_df: pd.DataFrame
    ) -> None:
        result = module.compute_all_metrics(sample_df)
        assert all(result["run_id"] == "run1")
