"""
Unit tests for the housing-unit method place projection module (ADR-060).

Tests cover data loading, trend fitting, PPH projection, population
computation, orchestration, edge cases, cross-validation, and config
parsing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.process.place_housing_unit_projection import (
    cross_validate_with_share_trending,
    load_housing_data,
    project_population_from_hu,
    project_pph,
    run_housing_unit_projections,
    trend_housing_units,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_csv(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    """Write a synthetic housing CSV and return its path."""
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "housing.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _config_for(csv_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a minimal config dict pointing at *csv_path*."""
    hu: dict[str, Any] = {
        "enabled": True,
        "housing_data_path": str(csv_path),
        "projection_years": [2025, 2030],
        "trend_method": "log_linear",
        "pph_method": "hold_last",
        "min_history_years": 3,
    }
    hu.update(overrides)
    return {"housing_unit_method": hu}


def _history_df(years: list[int], hu_values: list[float]) -> pd.DataFrame:
    """Create a simple (year, housing_units) DataFrame."""
    return pd.DataFrame({"year": years, "housing_units": hu_values})


def _pph_df(years: list[int], pph_values: list[float]) -> pd.DataFrame:
    """Create a simple (year, avg_hh_size) DataFrame."""
    return pd.DataFrame({"year": years, "avg_hh_size": pph_values})


# ---------------------------------------------------------------------------
# 1. Housing data loading
# ---------------------------------------------------------------------------


class TestLoadHousingData:
    """Tests for load_housing_data()."""

    def test_valid_csv_loads_correctly(self, tmp_path: Path) -> None:
        """Happy-path: valid CSV loads with correct dtypes."""
        csv = _synthetic_csv(tmp_path, [
            {"place_fips": "3825700", "place_name": "Fargo", "year": 2020,
             "housing_units": 55000, "avg_hh_size": 2.30},
            {"place_fips": "3825700", "place_name": "Fargo", "year": 2021,
             "housing_units": 56000, "avg_hh_size": 2.28},
        ])
        config = _config_for(csv)
        df = load_housing_data(config)
        assert len(df) == 2
        assert np.issubdtype(df["housing_units"].dtype, np.number)
        assert df["year"].dtype == int

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        """CSV missing required columns raises ValueError."""
        csv = _synthetic_csv(tmp_path, [
            {"place_fips": "3825700", "year": 2020, "housing_units": 100},
        ])
        config = _config_for(csv)
        with pytest.raises(ValueError, match="missing required columns"):
            load_housing_data(config)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        config = _config_for(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            load_housing_data(config)

    def test_empty_data_loads_without_error(self, tmp_path: Path) -> None:
        """CSV with headers but no data rows loads as empty DataFrame."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("place_fips,place_name,year,housing_units,avg_hh_size\n")
        config = _config_for(csv_path)
        df = load_housing_data(config)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# 2. Linear trend fitting
# ---------------------------------------------------------------------------


class TestTrendLinear:
    """Tests for trend_housing_units() with method='linear'."""

    def test_linear_recovers_known_slope(self) -> None:
        """Linear trend with exact slope = 100 HU/year is recovered."""
        years = [2010, 2011, 2012, 2013, 2014]
        hu = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        df = _history_df(years, hu)
        proj = trend_housing_units(df, method="linear", projection_years=[2015, 2020])
        # slope is exactly 100/yr
        np.testing.assert_allclose(proj["hu_projected"].values, [1500.0, 2000.0], atol=1.0)

    def test_linear_floors_at_zero(self) -> None:
        """Linear trend does not produce negative housing units."""
        years = [2010, 2015]
        hu = [100.0, 10.0]
        df = _history_df(years, hu)
        proj = trend_housing_units(df, method="linear", projection_years=[2030])
        assert proj["hu_projected"].iloc[0] >= 0.0


# ---------------------------------------------------------------------------
# 3. Log-linear trend fitting
# ---------------------------------------------------------------------------


class TestTrendLogLinear:
    """Tests for trend_housing_units() with method='log_linear'."""

    def test_log_linear_exponential_growth(self) -> None:
        """Log-linear trend recovers an exponential growth pattern."""
        years = [2010, 2015, 2020]
        # 2% annual growth: HU(t) = 1000 * exp(0.02 * (t - 2010))
        hu = [1000 * np.exp(0.02 * (y - 2010)) for y in years]
        df = _history_df(years, hu)
        proj = trend_housing_units(df, method="log_linear", projection_years=[2025])
        expected = 1000 * np.exp(0.02 * 15)
        np.testing.assert_allclose(proj["hu_projected"].iloc[0], expected, rtol=0.01)

    def test_log_linear_handles_zero_hu(self) -> None:
        """Zero HU values are guarded (replaced with 1.0 for log)."""
        df = _history_df([2010, 2015], [0.0, 100.0])
        proj = trend_housing_units(df, method="log_linear", projection_years=[2020])
        assert proj["hu_projected"].iloc[0] > 0

    def test_unknown_method_raises(self) -> None:
        """Unknown trend method raises ValueError."""
        df = _history_df([2010, 2015], [100.0, 200.0])
        with pytest.raises(ValueError, match="Unknown trend method"):
            trend_housing_units(df, method="cubic_spline", projection_years=[2020])


# ---------------------------------------------------------------------------
# 4. PPH projection
# ---------------------------------------------------------------------------


class TestProjectPPH:
    """Tests for project_pph()."""

    def test_hold_last_uses_most_recent(self) -> None:
        """hold_last method repeats the final observed PPH."""
        df = _pph_df([2015, 2020], [2.50, 2.40])
        proj = project_pph(df, method="hold_last", projection_years=[2025, 2030])
        assert proj["pph_projected"].iloc[0] == pytest.approx(2.40)
        assert proj["pph_projected"].iloc[1] == pytest.approx(2.40)

    def test_linear_trend_extrapolates(self) -> None:
        """linear_trend method extrapolates the observed PPH slope."""
        df = _pph_df([2010, 2015, 2020], [2.60, 2.50, 2.40])
        proj = project_pph(df, method="linear_trend", projection_years=[2025])
        # Slope = -0.02/yr, so 2025 => 2.30
        assert proj["pph_projected"].iloc[0] == pytest.approx(2.30, abs=0.05)

    def test_linear_trend_floors_at_one(self) -> None:
        """PPH never drops below 1.0 (a household has at least 1 person)."""
        df = _pph_df([2010, 2020], [2.0, 1.2])
        proj = project_pph(df, method="linear_trend", projection_years=[2050])
        assert proj["pph_projected"].iloc[0] >= 1.0

    def test_empty_history_raises(self) -> None:
        """Empty PPH history raises ValueError."""
        df = pd.DataFrame(columns=["year", "avg_hh_size"])
        with pytest.raises(ValueError, match="empty history"):
            project_pph(df, method="hold_last", projection_years=[2025])

    def test_unknown_pph_method_raises(self) -> None:
        """Unknown PPH method raises ValueError."""
        df = _pph_df([2015, 2020], [2.5, 2.4])
        with pytest.raises(ValueError, match="Unknown PPH method"):
            project_pph(df, method="bayesian", projection_years=[2025])


# ---------------------------------------------------------------------------
# 5. Population computation
# ---------------------------------------------------------------------------


class TestProjectPopulationFromHU:
    """Tests for project_population_from_hu()."""

    def test_multiplication_is_correct(self) -> None:
        """HU * PPH = population, merged by year."""
        hu = pd.DataFrame({"year": [2025, 2030], "hu_projected": [1000.0, 1200.0]})
        pph = pd.DataFrame({"year": [2025, 2030], "pph_projected": [2.5, 2.5]})
        pop = project_population_from_hu(hu, pph)
        np.testing.assert_allclose(pop["population_hu"].values, [2500.0, 3000.0])

    def test_mismatched_years_use_inner_join(self) -> None:
        """Only years present in BOTH inputs produce rows."""
        hu = pd.DataFrame({"year": [2025, 2030, 2035], "hu_projected": [100.0, 200.0, 300.0]})
        pph = pd.DataFrame({"year": [2025, 2030], "pph_projected": [2.0, 2.0]})
        pop = project_population_from_hu(hu, pph)
        assert len(pop) == 2
        assert set(pop["year"]) == {2025, 2030}


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------


class TestRunHousingUnitProjections:
    """Tests for run_housing_unit_projections()."""

    def test_multiple_places(self, tmp_path: Path) -> None:
        """Orchestrator produces projections for multiple places."""
        rows = []
        for place in ["3825700", "3807200"]:
            for year in range(2010, 2024):
                rows.append({
                    "place_fips": place,
                    "place_name": "TestPlace",
                    "year": year,
                    "housing_units": 1000 + (year - 2010) * 50,
                    "avg_hh_size": 2.40,
                })
        csv = _synthetic_csv(tmp_path, rows)
        config = _config_for(csv, projection_years=[2025, 2030])
        result = run_housing_unit_projections(config)
        assert result["place_fips"].nunique() == 2
        assert set(result["year"]) == {2025, 2030}

    def test_config_driven_method_selection(self, tmp_path: Path) -> None:
        """Orchestrator honours trend_method from config."""
        rows = [
            {"place_fips": "3825700", "place_name": "Fargo", "year": y,
             "housing_units": 1000 + (y - 2010) * 50, "avg_hh_size": 2.4}
            for y in range(2010, 2024)
        ]
        csv = _synthetic_csv(tmp_path, rows)
        config = _config_for(csv, trend_method="linear", projection_years=[2025])
        result = run_housing_unit_projections(config)
        assert result["method"].iloc[0] == "hu_linear"

    def test_insufficient_history_skipped(self, tmp_path: Path) -> None:
        """Places with fewer than min_history_years vintages are skipped."""
        rows = [
            {"place_fips": "3825700", "place_name": "Fargo", "year": 2020,
             "housing_units": 1000, "avg_hh_size": 2.4},
            {"place_fips": "3825700", "place_name": "Fargo", "year": 2021,
             "housing_units": 1050, "avg_hh_size": 2.4},
        ]
        csv = _synthetic_csv(tmp_path, rows)
        config = _config_for(csv, min_history_years=3)
        result = run_housing_unit_projections(config)
        assert result.empty


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for trend and PPH functions."""

    def test_single_data_point_holds_constant(self) -> None:
        """Single observation results in constant projection."""
        df = _history_df([2020], [500.0])
        proj = trend_housing_units(df, method="log_linear", projection_years=[2025])
        assert proj["hu_projected"].iloc[0] == pytest.approx(500.0)

    def test_empty_history_raises(self) -> None:
        """Empty housing-unit history raises ValueError."""
        df = pd.DataFrame(columns=["year", "housing_units"])
        with pytest.raises(ValueError, match="empty"):
            trend_housing_units(df, method="linear", projection_years=[2025])

    def test_default_projection_years(self) -> None:
        """When projection_years is None, defaults are generated."""
        df = _history_df([2015, 2020], [1000.0, 1100.0])
        proj = trend_housing_units(df, method="linear", projection_years=None)
        # Should produce 2 default years (last + 5, last + 10)
        assert len(proj) == 2
        assert 2025 in proj["year"].values
        assert 2030 in proj["year"].values


# ---------------------------------------------------------------------------
# 8. Cross-validation
# ---------------------------------------------------------------------------


class TestCrossValidation:
    """Tests for cross_validate_with_share_trending()."""

    def test_divergence_computation(self) -> None:
        """Divergence metrics are correctly computed."""
        hu = pd.DataFrame({
            "place_fips": ["3825700", "3825700"],
            "year": [2025, 2030],
            "population_hu": [110.0, 130.0],
        })
        st = pd.DataFrame({
            "place_fips": ["3825700", "3825700"],
            "year": [2025, 2030],
            "population": [100.0, 120.0],
        })
        result = cross_validate_with_share_trending(hu, st)
        assert len(result) == 2
        # 2025: (110 - 100) / 100 * 100 = 10%
        row_2025 = result[result["year"] == 2025].iloc[0]
        assert row_2025["pct_diff"] == pytest.approx(10.0)
        assert row_2025["abs_diff"] == pytest.approx(10.0)

    def test_total_population_column_accepted(self) -> None:
        """share-trending df with 'total_population' column is accepted."""
        hu = pd.DataFrame({
            "place_fips": ["3825700"],
            "year": [2025],
            "population_hu": [100.0],
        })
        st = pd.DataFrame({
            "place_fips": ["3825700"],
            "year": [2025],
            "total_population": [90.0],
        })
        result = cross_validate_with_share_trending(hu, st)
        assert len(result) == 1

    def test_missing_pop_column_raises(self) -> None:
        """share-trending df without population column raises ValueError."""
        hu = pd.DataFrame({
            "place_fips": ["3825700"],
            "year": [2025],
            "population_hu": [100.0],
        })
        st = pd.DataFrame({
            "place_fips": ["3825700"],
            "year": [2025],
            "other_col": [90.0],
        })
        with pytest.raises(ValueError, match="population"):
            cross_validate_with_share_trending(hu, st)


# ---------------------------------------------------------------------------
# 9. Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Tests for config-driven behaviour."""

    def test_disabled_config_returns_gracefully(self, tmp_path: Path) -> None:
        """When enabled=false, pipeline stage should skip gracefully.

        The module-level function does not check enabled (that is the
        pipeline stage's job), so we test that the stage helper would
        skip.  Here we verify the config is parsed without error.
        """
        rows = [
            {"place_fips": "3825700", "place_name": "Fargo", "year": y,
             "housing_units": 1000, "avg_hh_size": 2.4}
            for y in range(2010, 2024)
        ]
        csv = _synthetic_csv(tmp_path, rows)
        config = _config_for(csv, enabled=False)
        # The module still runs when called directly; enabled is for the stage
        result = run_housing_unit_projections(config)
        assert isinstance(result, pd.DataFrame)

    def test_place_fips_filter(self, tmp_path: Path) -> None:
        """place_fips_list filters to only requested places."""
        rows = []
        for place in ["3825700", "3807200", "3833900"]:
            for year in range(2010, 2024):
                rows.append({
                    "place_fips": place,
                    "place_name": "Test",
                    "year": year,
                    "housing_units": 1000,
                    "avg_hh_size": 2.4,
                })
        csv = _synthetic_csv(tmp_path, rows)
        config = _config_for(csv)
        result = run_housing_unit_projections(config, place_fips_list=["3825700"])
        assert result["place_fips"].nunique() == 1
        assert result["place_fips"].iloc[0] == "3825700"
