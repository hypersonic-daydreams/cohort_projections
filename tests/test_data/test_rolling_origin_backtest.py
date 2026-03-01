"""
Tests for PP-005 WS-A rolling-origin cross-validation backtesting utilities.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.process.rolling_origin_backtest import (
    aggregate_rolling_metrics,
    build_per_window_summary,
    generate_rolling_windows,
    run_rolling_origin_backtest,
    select_rolling_winner,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_share_history(start: int = 2000, end: int = 2024) -> pd.DataFrame:
    """Create synthetic county share history for rolling-origin tests."""
    rows: list[dict[str, object]] = []
    for year in range(start, end + 1):
        county_pop = 1000.0 + (year - start) * 10.0
        share_a = 0.40 + (year - start) * 0.002
        share_b = 0.30 - (year - start) * 0.001
        for place_fips, share in [("3825700", share_a), ("3884780", share_b)]:
            rows.append(
                {
                    "county_fips": "38017",
                    "place_fips": place_fips,
                    "place_name": f"Synthetic {place_fips}",
                    "year": year,
                    "row_type": "place",
                    "share_raw": share,
                    "county_population": county_pop,
                    "population": share * county_pop,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_county_pop(start: int = 2000, end: int = 2024) -> pd.DataFrame:
    """Create matching county population for rolling-origin tests."""
    rows = []
    for year in range(start, end + 1):
        rows.append(
            {
                "county_fips": "38017",
                "year": year,
                "county_population": 1000.0 + (year - start) * 10.0,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_tier_assignments() -> pd.DataFrame:
    """Tier assignments for the two synthetic places."""
    return pd.DataFrame(
        [
            {"place_fips": "3825700", "confidence_tier": "HIGH", "population_2024": 500.0},
            {"place_fips": "3884780", "confidence_tier": "MODERATE", "population_2024": 300.0},
        ]
    )


def _make_mock_results(
    variant_ids: list[str],
    windows: list[tuple[int, int, int, int]],
    scores: dict[str, list[float]] | None = None,
) -> list[dict[str, object]]:
    """Build mock per-window-per-variant result dicts for aggregation tests."""
    results: list[dict[str, object]] = []
    for window in windows:
        train_start, train_end, test_start, test_end = window
        window_label = f"{train_start}-{train_end}/{test_start}-{test_end}"
        for vi, variant_id in enumerate(variant_ids):
            if scores and variant_id in scores:
                score = scores[variant_id][len([r for r in results if r["variant_id"] == variant_id])]
            else:
                score = 5.0 + vi * 0.5
            results.append(
                {
                    "window": window_label,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "variant_id": variant_id,
                    "fitting_method": "ols" if variant_id.startswith("A") else "wls",
                    "constraint_method": (
                        "proportional" if variant_id.endswith("I") else "cap_and_redistribute"
                    ),
                    "score": score,
                    "tier_aggregates": pd.DataFrame(),
                    "place_metrics": pd.DataFrame(),
                }
            )
    return results


# ---------------------------------------------------------------------------
# generate_rolling_windows
# ---------------------------------------------------------------------------


class TestGenerateRollingWindows:
    """Tests for window generation logic."""

    def test_default_25_year_history_produces_four_windows(self) -> None:
        """2000-2024 with min_train=5, horizon=5 yields 4 expanding windows."""
        windows = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        assert len(windows) == 4

    def test_first_window_starts_at_history_start(self) -> None:
        """First window training always starts at history_start."""
        windows = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        assert windows[0][0] == 2000

    def test_first_window_train_length_equals_min_train_years(self) -> None:
        """First window has exactly min_train_years of training data."""
        windows = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        train_start, train_end, _, _ = windows[0]
        assert train_end - train_start + 1 == 5

    def test_expected_windows_for_2000_2024(self) -> None:
        """Exact windows for the ND 2000-2024 history range."""
        windows = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        expected = [
            (2000, 2004, 2005, 2009),
            (2000, 2009, 2010, 2014),
            (2000, 2014, 2015, 2019),
            (2000, 2019, 2020, 2024),
        ]
        assert windows == expected

    def test_windows_are_non_overlapping(self) -> None:
        """Test periods never overlap with each other."""
        windows = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        test_ranges = [(w[2], w[3]) for w in windows]
        for i in range(len(test_ranges) - 1):
            assert test_ranges[i][1] < test_ranges[i + 1][0]

    def test_training_windows_are_expanding(self) -> None:
        """Each subsequent training window is strictly longer."""
        windows = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        train_lengths = [w[1] - w[0] + 1 for w in windows]
        for i in range(len(train_lengths) - 1):
            assert train_lengths[i + 1] > train_lengths[i]

    def test_insufficient_history_returns_empty(self) -> None:
        """If history is too short for even one window, return empty list."""
        windows = generate_rolling_windows(2020, 2024, min_train_years=5, test_horizon=5)
        assert windows == []

    def test_min_train_years_one_maximises_windows(self) -> None:
        """min_train_years=1 creates the most possible windows."""
        windows = generate_rolling_windows(2000, 2010, min_train_years=1, test_horizon=5)
        assert len(windows) == 2
        assert windows[0] == (2000, 2000, 2001, 2005)
        assert windows[1] == (2000, 2005, 2006, 2010)

    def test_large_horizon_may_reduce_window_count(self) -> None:
        """Larger test_horizon means fewer windows fit."""
        windows_5 = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=5)
        windows_10 = generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=10)
        assert len(windows_10) < len(windows_5)

    def test_invalid_min_train_raises_valueerror(self) -> None:
        """min_train_years < 1 is rejected."""
        with pytest.raises(ValueError, match="min_train_years"):
            generate_rolling_windows(2000, 2024, min_train_years=0, test_horizon=5)

    def test_invalid_test_horizon_raises_valueerror(self) -> None:
        """test_horizon < 1 is rejected."""
        with pytest.raises(ValueError, match="test_horizon"):
            generate_rolling_windows(2000, 2024, min_train_years=5, test_horizon=0)

    def test_start_after_end_raises_valueerror(self) -> None:
        """history_start > history_end is rejected."""
        with pytest.raises(ValueError, match="history_start"):
            generate_rolling_windows(2024, 2000, min_train_years=5, test_horizon=5)

    def test_exact_fit_yields_one_window(self) -> None:
        """History exactly min_train + horizon yields one window."""
        windows = generate_rolling_windows(2000, 2009, min_train_years=5, test_horizon=5)
        assert len(windows) == 1
        assert windows[0] == (2000, 2004, 2005, 2009)


# ---------------------------------------------------------------------------
# aggregate_rolling_metrics
# ---------------------------------------------------------------------------


class TestAggregateRollingMetrics:
    """Tests for cross-window score aggregation."""

    def test_mean_score_computed_correctly(self) -> None:
        """Mean score across windows matches manual calculation."""
        results = _make_mock_results(
            ["A-I"],
            [(2000, 2004, 2005, 2009), (2000, 2009, 2010, 2014)],
            scores={"A-I": [4.0, 6.0]},
        )
        agg = aggregate_rolling_metrics(results)
        assert len(agg) == 1
        np.testing.assert_allclose(agg.iloc[0]["mean_score"], 5.0)

    def test_median_score_computed_correctly(self) -> None:
        """Median score handles odd window counts."""
        results = _make_mock_results(
            ["B-II"],
            [
                (2000, 2004, 2005, 2009),
                (2000, 2009, 2010, 2014),
                (2000, 2014, 2015, 2019),
            ],
            scores={"B-II": [1.0, 10.0, 3.0]},
        )
        agg = aggregate_rolling_metrics(results)
        np.testing.assert_allclose(agg.iloc[0]["median_score"], 3.0)

    def test_std_score_nan_for_single_window(self) -> None:
        """Standard deviation is NaN when only one window exists."""
        results = _make_mock_results(["A-I"], [(2000, 2004, 2005, 2009)])
        agg = aggregate_rolling_metrics(results)
        assert np.isnan(agg.iloc[0]["std_score"])

    def test_multiple_variants_produce_multiple_rows(self) -> None:
        """Each variant gets its own aggregation row."""
        results = _make_mock_results(
            ["A-I", "B-II"],
            [(2000, 2004, 2005, 2009), (2000, 2009, 2010, 2014)],
        )
        agg = aggregate_rolling_metrics(results)
        assert len(agg) == 2
        assert set(agg["variant_id"]) == {"A-I", "B-II"}

    def test_n_windows_matches_actual_count(self) -> None:
        """n_windows column reflects actual window count per variant."""
        windows = [
            (2000, 2004, 2005, 2009),
            (2000, 2009, 2010, 2014),
            (2000, 2014, 2015, 2019),
        ]
        results = _make_mock_results(["A-I"], windows)
        agg = aggregate_rolling_metrics(results)
        assert int(agg.iloc[0]["n_windows"]) == 3

    def test_empty_results_raises_valueerror(self) -> None:
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            aggregate_rolling_metrics([])


# ---------------------------------------------------------------------------
# build_per_window_summary
# ---------------------------------------------------------------------------


class TestBuildPerWindowSummary:
    """Tests for per-window summary table."""

    def test_correct_row_count(self) -> None:
        """One row per window-variant combination."""
        results = _make_mock_results(
            ["A-I", "B-II"],
            [(2000, 2004, 2005, 2009), (2000, 2009, 2010, 2014)],
        )
        summary = build_per_window_summary(results)
        assert len(summary) == 4  # 2 variants x 2 windows

    def test_required_columns_present(self) -> None:
        """Summary has all expected columns."""
        results = _make_mock_results(["A-I"], [(2000, 2004, 2005, 2009)])
        summary = build_per_window_summary(results)
        expected = {
            "window",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "variant_id",
            "fitting_method",
            "constraint_method",
            "score",
        }
        assert expected.issubset(set(summary.columns))

    def test_empty_results_raises_valueerror(self) -> None:
        """Empty input is rejected."""
        with pytest.raises(ValueError, match="empty"):
            build_per_window_summary([])


# ---------------------------------------------------------------------------
# select_rolling_winner
# ---------------------------------------------------------------------------


class TestSelectRollingWinner:
    """Tests for winner selection from aggregated rolling scores."""

    def test_lowest_mean_score_wins(self) -> None:
        """Variant with the lowest mean_score is selected."""
        agg = pd.DataFrame(
            [
                {
                    "variant_id": "A-I",
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                    "mean_score": 5.0,
                    "median_score": 5.0,
                    "std_score": 1.0,
                    "min_score": 4.0,
                    "max_score": 6.0,
                    "n_windows": 3,
                },
                {
                    "variant_id": "B-II",
                    "fitting_method": "wls",
                    "constraint_method": "cap_and_redistribute",
                    "mean_score": 3.0,
                    "median_score": 3.0,
                    "std_score": 0.5,
                    "min_score": 2.5,
                    "max_score": 3.5,
                    "n_windows": 3,
                },
            ]
        )
        info = select_rolling_winner(agg, acceptance_criteria="mean_score")
        assert info["winner_variant_id"] == "B-II"

    def test_median_score_criteria(self) -> None:
        """median_score acceptance_criteria uses median column."""
        agg = pd.DataFrame(
            [
                {
                    "variant_id": "A-I",
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                    "mean_score": 5.0,
                    "median_score": 3.0,
                    "std_score": 1.0,
                    "min_score": 3.0,
                    "max_score": 7.0,
                    "n_windows": 3,
                },
                {
                    "variant_id": "B-II",
                    "fitting_method": "wls",
                    "constraint_method": "cap_and_redistribute",
                    "mean_score": 4.0,
                    "median_score": 4.0,
                    "std_score": 0.5,
                    "min_score": 3.5,
                    "max_score": 4.5,
                    "n_windows": 3,
                },
            ]
        )
        info = select_rolling_winner(agg, acceptance_criteria="median_score")
        assert info["winner_variant_id"] == "A-I"

    def test_tie_breaking_prefers_a_over_b(self) -> None:
        """On tied scores, A-family variants are preferred over B-family."""
        agg = pd.DataFrame(
            [
                {
                    "variant_id": "A-I",
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                    "mean_score": 3.0,
                    "median_score": 3.0,
                    "std_score": 0.0,
                    "min_score": 3.0,
                    "max_score": 3.0,
                    "n_windows": 2,
                },
                {
                    "variant_id": "B-I",
                    "fitting_method": "wls",
                    "constraint_method": "proportional",
                    "mean_score": 3.0,
                    "median_score": 3.0,
                    "std_score": 0.0,
                    "min_score": 3.0,
                    "max_score": 3.0,
                    "n_windows": 2,
                },
            ]
        )
        info = select_rolling_winner(agg, acceptance_criteria="mean_score")
        assert info["winner_variant_id"] == "A-I"

    def test_all_same_score_picks_simplest(self) -> None:
        """When all four variants tie, A-I wins as simplest."""
        rows = []
        for vid, fm, cm in [
            ("A-I", "ols", "proportional"),
            ("A-II", "ols", "cap_and_redistribute"),
            ("B-I", "wls", "proportional"),
            ("B-II", "wls", "cap_and_redistribute"),
        ]:
            rows.append(
                {
                    "variant_id": vid,
                    "fitting_method": fm,
                    "constraint_method": cm,
                    "mean_score": 5.0,
                    "median_score": 5.0,
                    "std_score": 0.0,
                    "min_score": 5.0,
                    "max_score": 5.0,
                    "n_windows": 2,
                }
            )
        agg = pd.DataFrame(rows)
        info = select_rolling_winner(agg, acceptance_criteria="mean_score")
        assert info["winner_variant_id"] == "A-I"

    def test_invalid_acceptance_criteria_raises_valueerror(self) -> None:
        """Unrecognised criteria value is rejected."""
        agg = pd.DataFrame(
            [
                {
                    "variant_id": "A-I",
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                    "mean_score": 3.0,
                    "median_score": 3.0,
                    "std_score": 0.0,
                    "min_score": 3.0,
                    "max_score": 3.0,
                    "n_windows": 2,
                }
            ]
        )
        with pytest.raises(ValueError, match="acceptance_criteria"):
            select_rolling_winner(agg, acceptance_criteria="invalid")

    def test_empty_scores_raises_valueerror(self) -> None:
        """Empty aggregated scores table is rejected."""
        with pytest.raises(ValueError, match="empty"):
            select_rolling_winner(pd.DataFrame(), acceptance_criteria="mean_score")

    def test_winner_payload_has_expected_keys(self) -> None:
        """Winner dict includes all documented keys."""
        agg = pd.DataFrame(
            [
                {
                    "variant_id": "B-II",
                    "fitting_method": "wls",
                    "constraint_method": "cap_and_redistribute",
                    "mean_score": 3.0,
                    "median_score": 3.0,
                    "std_score": 0.5,
                    "min_score": 2.5,
                    "max_score": 3.5,
                    "n_windows": 4,
                }
            ]
        )
        info = select_rolling_winner(agg)
        expected_keys = {
            "winner_variant_id",
            "acceptance_criteria",
            "winner_score",
            "winner_mean_score",
            "winner_median_score",
            "winner_n_windows",
            "scores",
        }
        assert expected_keys == set(info.keys())


# ---------------------------------------------------------------------------
# run_rolling_origin_backtest (integration with mocks)
# ---------------------------------------------------------------------------


class TestRunRollingOriginBacktest:
    """Integration tests using mocked run_single_variant."""

    def _mock_run_single_variant(
        self,
        variant_id: str,
        fitting_method: str,
        constraint_method: str,
        train_years: list[int],
        test_years: list[int],
        share_history: pd.DataFrame,
        county_pop: pd.DataFrame,
        config: dict[str, object],
    ) -> dict[str, pd.DataFrame]:
        """Mock that returns simple projected/actual DataFrames."""
        test_start, test_end = test_years
        years = list(range(test_start, test_end + 1))
        projected_rows = []
        actual_rows = []
        for year in years:
            for pfips in ["3825700", "3884780"]:
                projected_rows.append(
                    {
                        "variant_id": variant_id,
                        "county_fips": "38017",
                        "place_fips": pfips,
                        "year": year,
                        "projected_population": 500.0,
                        "projected_share": 0.5,
                    }
                )
                actual_rows.append(
                    {
                        "place_fips": pfips,
                        "year": year,
                        "actual_population": 500.0 + (10.0 if pfips == "3825700" else -5.0),
                    }
                )
        return {
            "projected": pd.DataFrame(projected_rows),
            "actual": pd.DataFrame(actual_rows),
        }

    @patch("cohort_projections.data.process.rolling_origin_backtest.run_single_variant")
    def test_returns_correct_result_count(self, mock_rsv: MagicMock) -> None:
        """Number of results = windows x variants."""
        mock_rsv.side_effect = self._mock_run_single_variant

        windows = [(2000, 2004, 2005, 2009), (2000, 2009, 2010, 2014)]
        variants = {
            "A-I": {"fitting_method": "ols", "constraint_method": "proportional"},
            "B-II": {"fitting_method": "wls", "constraint_method": "cap_and_redistribute"},
        }
        config: dict[str, object] = {"place_projections": {"model": {"epsilon": 0.001}}}

        results = run_rolling_origin_backtest(
            share_history=_synthetic_share_history(),
            county_pop=_synthetic_county_pop(),
            tier_assignments=_synthetic_tier_assignments(),
            variants=variants,
            config=config,
            windows=windows,
        )
        assert len(results) == 4  # 2 windows x 2 variants

    @patch("cohort_projections.data.process.rolling_origin_backtest.run_single_variant")
    def test_each_result_has_score(self, mock_rsv: MagicMock) -> None:
        """Each result dict contains a numeric score."""
        mock_rsv.side_effect = self._mock_run_single_variant

        windows = [(2000, 2004, 2005, 2009)]
        variants = {"A-I": {"fitting_method": "ols", "constraint_method": "proportional"}}
        config: dict[str, object] = {"place_projections": {"model": {"epsilon": 0.001}}}

        results = run_rolling_origin_backtest(
            share_history=_synthetic_share_history(),
            county_pop=_synthetic_county_pop(),
            tier_assignments=_synthetic_tier_assignments(),
            variants=variants,
            config=config,
            windows=windows,
        )
        for r in results:
            assert isinstance(r["score"], float)
            assert np.isfinite(r["score"])

    def test_empty_windows_raises_valueerror(self) -> None:
        """Empty window list raises ValueError."""
        variants = {"A-I": {"fitting_method": "ols", "constraint_method": "proportional"}}
        config: dict[str, object] = {
            "place_projections": {
                "model": {"history_start": 2020, "history_end": 2024},
                "rolling_origin_backtest": {"min_train_years": 10, "test_horizon": 10},
            }
        }
        with pytest.raises(ValueError, match="No valid rolling-origin windows"):
            run_rolling_origin_backtest(
                share_history=_synthetic_share_history(),
                county_pop=_synthetic_county_pop(),
                tier_assignments=_synthetic_tier_assignments(),
                variants=variants,
                config=config,
                windows=[],
            )

    @patch("cohort_projections.data.process.rolling_origin_backtest.run_single_variant")
    def test_windows_auto_generated_from_config(self, mock_rsv: MagicMock) -> None:
        """When windows=None, windows are derived from config."""
        mock_rsv.side_effect = self._mock_run_single_variant

        config: dict[str, object] = {
            "place_projections": {
                "model": {"history_start": 2000, "history_end": 2014, "epsilon": 0.001},
                "rolling_origin_backtest": {"min_train_years": 5, "test_horizon": 5},
            }
        }
        variants = {"A-I": {"fitting_method": "ols", "constraint_method": "proportional"}}

        results = run_rolling_origin_backtest(
            share_history=_synthetic_share_history(2000, 2014),
            county_pop=_synthetic_county_pop(2000, 2014),
            tier_assignments=_synthetic_tier_assignments(),
            variants=variants,
            config=config,
            windows=None,
        )
        # 2000-2014 with min_train=5, horizon=5 => 2 windows
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestConfigParsing:
    """Tests for config-driven window generation."""

    def test_config_driven_window_generation(self) -> None:
        """Config values produce expected windows."""
        config = {
            "place_projections": {
                "model": {"history_start": 2000, "history_end": 2024},
                "rolling_origin_backtest": {
                    "enabled": True,
                    "min_train_years": 5,
                    "test_horizon": 5,
                    "acceptance_criteria": "mean_score",
                },
            }
        }
        ro_cfg = config["place_projections"]["rolling_origin_backtest"]
        model_cfg = config["place_projections"]["model"]

        windows = generate_rolling_windows(
            history_start=int(model_cfg["history_start"]),
            history_end=int(model_cfg["history_end"]),
            min_train_years=int(ro_cfg["min_train_years"]),
            test_horizon=int(ro_cfg["test_horizon"]),
        )
        assert len(windows) == 4
        assert windows[-1] == (2000, 2019, 2020, 2024)
