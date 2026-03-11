"""Tests for benchmark runner implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.benchmark_runners import (
    average_growth,
    build_component_swap,
    carry_forward,
    linear_trend,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result_df(
    run_id: str = "test_run",
    geographies: list[str] | None = None,
    years: list[int] | None = None,
    actual_values: list[float] | None = None,
    projected_values: list[float] | None = None,
    target: str = "population",
) -> pd.DataFrame:
    """Build a synthetic projection-result DataFrame.

    Creates one row per (geography, year) combination.
    """
    if geographies is None:
        geographies = ["38017"]
    if years is None:
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

    n_rows = len(geographies) * len(years)
    if actual_values is None:
        actual_values = [100.0] * n_rows
    if projected_values is None:
        projected_values = actual_values.copy()

    rows = []
    idx = 0
    for geo in geographies:
        for yr in years:
            rows.append(
                {
                    "run_id": run_id,
                    "geography": geo,
                    "geography_type": "county",
                    "year": yr,
                    "horizon": yr - 2020,
                    "sex": "total",
                    "age_group": "total",
                    "target": target,
                    "projected_value": projected_values[idx],
                    "actual_value": actual_values[idx],
                    "base_value": actual_values[0],
                }
            )
            idx += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# carry_forward tests
# ---------------------------------------------------------------------------


class TestCarryForward:
    """Tests for the carry_forward benchmark."""

    def test_constant_projection(self):
        """Carry-forward should produce the origin-year value for all horizons."""
        # Actual values increase linearly: 100, 110, 120, ...
        years = [2018, 2019, 2020, 2021, 2022, 2023]
        actuals = [80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        df = _make_result_df(years=years, actual_values=actuals)

        result = carry_forward(df, origin_year=2020)

        # All projected values should be 100.0 (the 2020 actual)
        assert (result["projected_value"] == 100.0).all()

    def test_run_id(self):
        """Should set run_id to benchmark_carry_forward."""
        df = _make_result_df()
        result = carry_forward(df, origin_year=2020)
        assert (result["run_id"] == "benchmark_carry_forward").all()

    def test_preserves_other_columns(self):
        """Should preserve all columns except projected_value and run_id."""
        df = _make_result_df()
        result = carry_forward(df, origin_year=2020)

        assert list(result["geography"]) == list(df["geography"])
        assert list(result["year"]) == list(df["year"])
        assert list(result["actual_value"]) == list(df["actual_value"])

    def test_multiple_geographies(self):
        """Each geography should get its own base value."""
        years = [2019, 2020, 2021]
        # geo1: actual=100 at 2020, geo2: actual=200 at 2020
        actuals = [90.0, 100.0, 110.0, 190.0, 200.0, 210.0]
        df = _make_result_df(
            geographies=["38017", "38015"],
            years=years,
            actual_values=actuals,
        )

        result = carry_forward(df, origin_year=2020)

        geo1 = result[result["geography"] == "38017"]
        geo2 = result[result["geography"] == "38015"]
        assert (geo1["projected_value"] == 100.0).all()
        assert (geo2["projected_value"] == 200.0).all()

    def test_missing_columns_raises(self):
        """Should raise ValueError when required columns are missing."""
        df = pd.DataFrame({"geography": ["38017"], "year": [2020]})
        with pytest.raises(ValueError, match="missing required columns"):
            carry_forward(df, origin_year=2020)


# ---------------------------------------------------------------------------
# linear_trend tests
# ---------------------------------------------------------------------------


class TestLinearTrend:
    """Tests for the linear_trend benchmark."""

    def test_extrapolates_correctly(self):
        """Given a perfect linear history, trend should extrapolate exactly."""
        # History: 100, 110, 120, 130, 140 for years 2016-2020
        # Slope = 10/year, so 2021 -> 150, 2022 -> 160
        years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]
        actuals = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
        df = _make_result_df(years=years, actual_values=actuals)

        result = linear_trend(df, origin_year=2020, lookback=5)

        # Check the future projections
        future = result[result["year"] > 2020]
        np.testing.assert_allclose(
            future["projected_value"].values, [150.0, 160.0], atol=0.01
        )

    def test_run_id(self):
        """Should set run_id to benchmark_linear_trend."""
        df = _make_result_df()
        result = linear_trend(df, origin_year=2020)
        assert (result["run_id"] == "benchmark_linear_trend").all()

    def test_constant_history_zero_slope(self):
        """Constant history should produce zero slope (carry-forward)."""
        years = [2018, 2019, 2020, 2021, 2022]
        actuals = [100.0, 100.0, 100.0, 100.0, 100.0]
        df = _make_result_df(years=years, actual_values=actuals)

        result = linear_trend(df, origin_year=2020, lookback=3)

        np.testing.assert_allclose(
            result["projected_value"].values, [100.0] * 5, atol=0.01
        )

    def test_lookback_parameter(self):
        """Should only use the specified lookback window."""
        # Early years have different slope than recent
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
        actuals = [500.0, 400.0, 300.0, 100.0, 110.0, 120.0, 130.0]
        df = _make_result_df(years=years, actual_values=actuals)

        # With lookback=3, only 2018-2020 used (slope ~ +10/yr)
        result = linear_trend(df, origin_year=2020, lookback=3)
        proj_2021 = result[result["year"] == 2021]["projected_value"].iloc[0]

        # Should be close to 130 (120 + 10)
        np.testing.assert_allclose(proj_2021, 130.0, atol=0.5)


# ---------------------------------------------------------------------------
# average_growth tests
# ---------------------------------------------------------------------------


class TestAverageGrowth:
    """Tests for the average_growth benchmark."""

    def test_applies_mean_rate(self):
        """Given constant 10% growth, should compound at 10%."""
        years = [2018, 2019, 2020, 2021, 2022]
        # 10% growth each year
        actuals = [100.0, 110.0, 121.0, 133.1, 146.41]
        df = _make_result_df(years=years, actual_values=actuals)

        result = average_growth(df, origin_year=2020, lookback=3)

        # Mean growth rate from 2018-2020: (10/100 + 11/110) / 2 = 0.10
        # 2021: 121 * 1.10 = 133.1, 2022: 121 * 1.10^2 = 146.41
        proj_2021 = result[result["year"] == 2021]["projected_value"].iloc[0]
        proj_2022 = result[result["year"] == 2022]["projected_value"].iloc[0]

        np.testing.assert_allclose(proj_2021, 121.0 * 1.10, rtol=0.01)
        np.testing.assert_allclose(proj_2022, 121.0 * 1.10**2, rtol=0.01)

    def test_run_id(self):
        """Should set run_id to benchmark_average_growth."""
        df = _make_result_df()
        result = average_growth(df, origin_year=2020)
        assert (result["run_id"] == "benchmark_average_growth").all()

    def test_zero_growth(self):
        """Constant values should produce zero growth rate."""
        years = [2018, 2019, 2020, 2021, 2022]
        actuals = [100.0, 100.0, 100.0, 100.0, 100.0]
        df = _make_result_df(years=years, actual_values=actuals)

        result = average_growth(df, origin_year=2020, lookback=3)

        future = result[result["year"] > 2020]
        np.testing.assert_allclose(future["projected_value"].values, [100.0, 100.0])

    def test_handles_zero_base_value(self):
        """Should handle zero values without division errors."""
        years = [2018, 2019, 2020, 2021]
        actuals = [0.0, 0.0, 50.0, 60.0]
        df = _make_result_df(years=years, actual_values=actuals)

        # Should not raise
        result = average_growth(df, origin_year=2020, lookback=3)
        assert not result["projected_value"].isna().any()


# ---------------------------------------------------------------------------
# build_component_swap tests
# ---------------------------------------------------------------------------


class TestBuildComponentSwap:
    """Tests for the build_component_swap helper."""

    def test_swaps_specified_components(self):
        """Specified targets should come from method_b, rest from method_a."""
        years = [2020, 2021]

        # method_a has all targets
        rows_a = []
        for target, val in [("population", 1000.0), ("births", 50.0), ("deaths", 30.0)]:
            for yr in years:
                rows_a.append(
                    {
                        "run_id": "a",
                        "geography": "38017",
                        "geography_type": "county",
                        "year": yr,
                        "horizon": yr - 2020,
                        "sex": "total",
                        "age_group": "total",
                        "target": target,
                        "projected_value": val,
                        "actual_value": val,
                        "base_value": val,
                    }
                )
        method_a = pd.DataFrame(rows_a)

        # method_b has different values
        rows_b = []
        for target, val in [("population", 2000.0), ("births", 80.0), ("deaths", 60.0)]:
            for yr in years:
                rows_b.append(
                    {
                        "run_id": "b",
                        "geography": "38017",
                        "geography_type": "county",
                        "year": yr,
                        "horizon": yr - 2020,
                        "sex": "total",
                        "age_group": "total",
                        "target": target,
                        "projected_value": val,
                        "actual_value": val,
                        "base_value": val,
                    }
                )
        method_b = pd.DataFrame(rows_b)

        result = build_component_swap(
            method_a, method_b, components=["births", "deaths"]
        )

        # Population should come from method_a
        pop = result[result["target"] == "population"]
        assert (pop["projected_value"] == 1000.0).all()

        # Births and deaths should come from method_b
        births = result[result["target"] == "births"]
        deaths = result[result["target"] == "deaths"]
        assert (births["projected_value"] == 80.0).all()
        assert (deaths["projected_value"] == 60.0).all()

    def test_composite_run_id(self):
        """run_id should encode the swap label."""
        method_a = _make_result_df(run_id="a", target="population")
        method_b = _make_result_df(run_id="b", target="births")

        result = build_component_swap(
            method_a,
            method_b,
            components=["births"],
            label_a="m2024",
            label_b="m2026",
        )

        expected_id = "m2024_swap_births_from_m2026"
        assert (result["run_id"] == expected_id).all()

    def test_no_swap_returns_method_a(self):
        """With no matching components, result should equal method_a."""
        method_a = _make_result_df(run_id="a", target="population")
        method_b = _make_result_df(run_id="b", target="population")

        result = build_component_swap(
            method_a, method_b, components=["births"]
        )

        # No births in either, so all rows come from method_a
        assert len(result) == len(method_a)
        assert (result["target"] == "population").all()

    def test_multiple_components_sorted_in_label(self):
        """Component names in run_id should be sorted."""
        method_a = _make_result_df(run_id="a", target="population")
        method_b = _make_result_df(run_id="b", target="population")

        result = build_component_swap(
            method_a, method_b, components=["deaths", "births"]
        )

        # Should be alphabetically sorted
        assert "births_deaths" in result["run_id"].iloc[0]
