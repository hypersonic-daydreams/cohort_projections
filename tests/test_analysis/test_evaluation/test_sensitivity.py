"""Tests for the Sensitivity and Robustness module (Module 3).

Uses a mock projection runner that returns synthetic DataFrames to
exercise the SensitivityModule without requiring real projection
infrastructure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.sensitivity import (
    SensitivityModule,
    _REQUIRED_COLS,
)
from cohort_projections.analysis.evaluation.utils import validate_dataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GEOGRAPHIES = ["38017", "38035", "38101"]
_HORIZONS = [1, 5, 10]

DEFAULT_CONFIG: dict[str, Any] = {
    "perturbation_pct": {
        "base_population": [0.5, 1.0],
        "births": [1.0],
    },
    "parameter_sweep_levels": 3,
    "monte_carlo_iterations": 20,
    "near_term_max_horizon": 5,
    "long_term_min_horizon": 10,
}


def _make_results_df(
    geographies: list[str] | None = None,
    horizons: list[int] | None = None,
    base: float = 10000.0,
    noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a synthetic projection-results DataFrame."""
    geographies = geographies or _GEOGRAPHIES
    horizons = horizons or _HORIZONS
    if rng is None:
        rng = np.random.default_rng(42)

    rows = []
    for geo in geographies:
        for hor in horizons:
            projected = base + hor * 100 + rng.normal(0, noise_scale)
            actual = base + hor * 100
            rows.append({
                "geography": geo,
                "year": 2025 + hor,
                "horizon": hor,
                "projected_value": projected,
                "actual_value": actual,
            })
    return pd.DataFrame(rows)


def _make_mock_runner(
    noise_scale: float = 0.0,
    unstable_geo: str | None = None,
    unstable_amplification: float = 10.0,
) -> Any:
    """Return a mock projection runner callable.

    If *unstable_geo* is given, that geography will have exaggerated
    responses to perturbations (to test instability detection).
    """

    def runner(overrides: dict[str, Any]) -> pd.DataFrame:
        rng = np.random.default_rng(hash(str(overrides)) % 2**31)
        df = _make_results_df(noise_scale=noise_scale, rng=rng)

        # Apply perturbation effects
        perturbation = overrides.get("__perturbation__")
        if perturbation is not None:
            # Single-component perturbation
            if "pct" in perturbation:
                pct = perturbation["pct"]
                df["projected_value"] *= 1 + pct / 100.0
                if unstable_geo:
                    mask = df["geography"] == unstable_geo
                    df.loc[mask, "projected_value"] *= (
                        1 + pct / 100.0 * unstable_amplification
                    )
            # Multi-component (MC) perturbation
            elif "components" in perturbation:
                total_pct = sum(perturbation["components"].values())
                df["projected_value"] *= 1 + total_pct / 100.0
                if unstable_geo:
                    mask = df["geography"] == unstable_geo
                    df.loc[mask, "projected_value"] *= (
                        1 + total_pct / 100.0 * unstable_amplification
                    )

        return df

    return runner


@pytest.fixture()
def baseline_results() -> pd.DataFrame:
    return _make_results_df()


@pytest.fixture()
def config() -> dict[str, Any]:
    return DEFAULT_CONFIG.copy()


@pytest.fixture()
def module(config: dict[str, Any]) -> SensitivityModule:
    return SensitivityModule(_make_mock_runner(), config)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


class TestValidateResultsDf:
    """Tests for validate_dataframe (sensitivity required columns)."""

    def test_valid_df_passes(self, baseline_results: pd.DataFrame) -> None:
        validate_dataframe(baseline_results, _REQUIRED_COLS)  # should not raise

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"geography": ["A"], "year": [2025]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, _REQUIRED_COLS)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------


class TestParameterSweep:
    """Tests for SensitivityModule.parameter_sweep."""

    def test_output_shape(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.parameter_sweep("blend_factor", [0.3, 0.5, 0.7], baseline_results)
        assert len(result) == 3
        assert set(result.columns) >= {
            "param_name",
            "param_value",
            "near_term_error",
            "long_term_error",
            "bias",
            "realism_score",
            "stability_score",
        }

    def test_param_values_preserved(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        values = [0.1, 0.5, 0.9]
        result = module.parameter_sweep("alpha", values, baseline_results)
        assert list(result["param_value"]) == values
        assert (result["param_name"] == "alpha").all()

    def test_realism_score_bounded(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.parameter_sweep("x", [1.0], baseline_results)
        assert result["realism_score"].iloc[0] >= 0.0
        assert result["realism_score"].iloc[0] <= 1.0


# ---------------------------------------------------------------------------
# Interaction sweep
# ---------------------------------------------------------------------------


class TestInteractionSweep:
    """Tests for SensitivityModule.interaction_sweep."""

    def test_factorial_count(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        pairs = [("alpha", "beta")]
        values = {"alpha": [1, 2], "beta": [10, 20, 30]}
        result = module.interaction_sweep(pairs, values, baseline_results)
        # 2 * 3 = 6 combinations
        assert len(result) == 6

    def test_columns_present(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.interaction_sweep(
            [("a", "b")], {"a": [1], "b": [2]}, baseline_results
        )
        assert set(result.columns) >= {
            "param_a",
            "param_b",
            "value_a",
            "value_b",
            "near_term_error",
            "long_term_error",
            "bias",
        }


# ---------------------------------------------------------------------------
# Perturbation test
# ---------------------------------------------------------------------------


class TestPerturbationTest:
    """Tests for SensitivityModule.perturbation_test."""

    def test_output_has_both_directions(
        self,
        baseline_results: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        mod = SensitivityModule(_make_mock_runner(), config)
        result = mod.perturbation_test(
            baseline_inputs={},
            perturbation_pcts={"base_population": [1.0]},
            baseline_results=baseline_results,
        )
        assert set(result["direction"].unique()) == {"positive", "negative"}

    def test_sensitivity_index_computed(
        self,
        baseline_results: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        mod = SensitivityModule(_make_mock_runner(), config)
        result = mod.perturbation_test(
            baseline_inputs={},
            perturbation_pcts={"births": [1.0]},
            baseline_results=baseline_results,
        )
        assert "sensitivity_index" in result.columns
        # Sensitivity index should be finite for non-zero baseline
        finite_mask = np.isfinite(result["sensitivity_index"])
        assert finite_mask.any()

    def test_unstable_county_detected(
        self,
        baseline_results: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        """An unstable county should have a higher sensitivity index."""
        unstable = "38017"
        runner = _make_mock_runner(unstable_geo=unstable, unstable_amplification=10.0)
        mod = SensitivityModule(runner, config)
        result = mod.perturbation_test(
            baseline_inputs={},
            perturbation_pcts={"base_population": [1.0]},
            baseline_results=baseline_results,
        )
        # The unstable county should have higher mean sensitivity
        mean_sens = result.groupby("geography")["sensitivity_index"].mean()
        assert mean_sens[unstable] > mean_sens.drop(unstable).mean()


# ---------------------------------------------------------------------------
# Monte Carlo propagation
# ---------------------------------------------------------------------------


class TestMonteCarloPropaagation:
    """Tests for SensitivityModule.monte_carlo_propagation."""

    def test_produces_intervals(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.monte_carlo_propagation(
            baseline_inputs={},
            n_iterations=10,
            baseline_results=baseline_results,
            seed=42,
        )
        assert "interval_width" in result.columns
        assert "p05" in result.columns
        assert "p95" in result.columns
        # Interval width should be non-negative
        assert (result["interval_width"] >= 0).all()

    def test_one_row_per_geo_horizon(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.monte_carlo_propagation(
            baseline_inputs={},
            n_iterations=10,
            baseline_results=baseline_results,
            seed=0,
        )
        n_expected = len(_GEOGRAPHIES) * len(_HORIZONS)
        assert len(result) == n_expected

    def test_uncertainty_amplification_present(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.monte_carlo_propagation(
            baseline_inputs={},
            n_iterations=10,
            baseline_results=baseline_results,
            seed=1,
        )
        assert "uncertainty_amplification" in result.columns


# ---------------------------------------------------------------------------
# Base-year sensitivity
# ---------------------------------------------------------------------------


class TestBaseYearSensitivity:
    """Tests for SensitivityModule.base_year_sensitivity."""

    def test_output_shape(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        result = module.base_year_sensitivity([2015, 2020], baseline_results)
        n_expected = 2 * len(_GEOGRAPHIES) * len(_HORIZONS)
        assert len(result) == n_expected
        assert "deviation_pct" in result.columns


# ---------------------------------------------------------------------------
# History-window and shock-year sensitivity
# ---------------------------------------------------------------------------


class TestHistoryWindowSensitivity:
    """Tests for SensitivityModule.history_window_sensitivity."""

    def test_returns_one_row_per_window(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        windows = [
            {"label": "full", "start_year": 2000, "end_year": 2024},
            {"label": "recent", "start_year": 2015, "end_year": 2024},
        ]
        result = module.history_window_sensitivity(windows, baseline_results)
        assert len(result) == 2
        assert list(result["window_label"]) == ["full", "recent"]


class TestShockYearSensitivity:
    """Tests for SensitivityModule.shock_year_sensitivity."""

    def test_returns_one_row_per_config(
        self,
        module: SensitivityModule,
        baseline_results: pd.DataFrame,
    ) -> None:
        configs = [
            {"label": "all_years"},
            {"label": "no_pandemic", "exclude_years": [2020, 2021]},
        ]
        result = module.shock_year_sensitivity(configs, baseline_results)
        assert len(result) == 2
        assert "bias" in result.columns


# ---------------------------------------------------------------------------
# Stability index
# ---------------------------------------------------------------------------


class TestComputeStabilityIndex:
    """Tests for SensitivityModule.compute_stability_index."""

    def test_ranks_unstable_county_last(
        self,
        config: dict[str, Any],
    ) -> None:
        """The unstable county should have the highest (worst) rank."""
        unstable = "38017"
        runner = _make_mock_runner(unstable_geo=unstable, unstable_amplification=10.0)
        mod = SensitivityModule(runner, config)
        baseline = _make_results_df()
        pert_results = mod.perturbation_test(
            baseline_inputs={},
            perturbation_pcts={"base_population": [1.0]},
            baseline_results=baseline,
        )
        stability = mod.compute_stability_index(pert_results)

        # Unstable county should be ranked last (highest number)
        unstable_row = stability[stability["geography"] == unstable]
        assert len(unstable_row) == 1
        assert unstable_row["stability_rank"].iloc[0] == len(_GEOGRAPHIES)
        assert bool(unstable_row["disproportionate_flag"].iloc[0]) is True

    def test_missing_column_raises(
        self,
        module: SensitivityModule,
    ) -> None:
        df = pd.DataFrame({"geography": ["A"], "other": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            module.compute_stability_index(df)

    def test_disproportionate_flag(
        self,
        module: SensitivityModule,
    ) -> None:
        """Counties with sensitivity > threshold get flagged."""
        df = pd.DataFrame({
            "geography": ["A", "A", "B", "B"],
            "sensitivity_index": [1.0, 1.5, 3.0, 4.0],
        })
        result = module.compute_stability_index(df)
        a_row = result[result["geography"] == "A"]
        b_row = result[result["geography"] == "B"]
        assert bool(a_row["disproportionate_flag"].iloc[0]) is False
        assert bool(b_row["disproportionate_flag"].iloc[0]) is True
