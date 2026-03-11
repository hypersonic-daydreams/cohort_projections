"""Tests for the EvaluationRunner orchestrator.

Covers full evaluation (with and without projection runner),
accuracy-only, comparison, report generation, and sensitivity
integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.data_structures import ScorecardEntry
from cohort_projections.analysis.evaluation.runner import EvaluationRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEOGRAPHIES = ["38017", "38035", "38101"]
_HORIZONS = [1, 5, 10]


def _make_results_df(
    geographies: list[str] | None = None,
    horizons: list[int] | None = None,
    base: float = 10000.0,
    noise_scale: float = 0.0,
    run_id: str = "run1",
    model_name: str = "m2026",
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
                "run_id": run_id,
                "model_name": model_name,
                "geography": geo,
                "geography_type": "county",
                "year": 2025 + hor,
                "horizon": hor,
                "sex": "total",
                "age_group": "total",
                "target": "population",
                "projected_value": projected,
                "actual_value": actual,
                "base_value": base,
            })
    return pd.DataFrame(rows)


def _make_mock_runner(noise_scale: float = 0.0) -> Any:
    """Return a mock projection runner callable.

    Uses the same pattern as test_sensitivity.py -- the callable accepts
    a dict of overrides and returns a results DataFrame.
    """

    def runner(overrides: dict[str, Any]) -> pd.DataFrame:
        rng = np.random.default_rng(hash(str(overrides)) % 2**31)
        df = _make_results_df(noise_scale=noise_scale, rng=rng)

        # Apply perturbation effects so sensitivity metrics are non-trivial
        perturbation = overrides.get("__perturbation__")
        if perturbation is not None:
            if "pct" in perturbation:
                pct = perturbation["pct"]
                df["projected_value"] *= 1 + pct / 100.0
            elif "components" in perturbation:
                total_pct = sum(perturbation["components"].values())
                df["projected_value"] *= 1 + total_pct / 100.0

        return df

    return runner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def eval_config_path() -> Path:
    """Path to the project evaluation config."""
    return Path(__file__).resolve().parents[3] / "config" / "evaluation_config.yaml"


@pytest.fixture()
def runner(eval_config_path: Path) -> EvaluationRunner:
    """EvaluationRunner loaded from project config."""
    return EvaluationRunner(config_path=eval_config_path)


@pytest.fixture()
def results_df() -> pd.DataFrame:
    """Baseline results DataFrame."""
    return _make_results_df()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Verify configuration is loaded and parsed correctly."""

    def test_horizons_loaded(self, runner: EvaluationRunner) -> None:
        assert isinstance(runner.horizons, list)
        assert len(runner.horizons) > 0
        assert all(isinstance(h, int) for h in runner.horizons)

    def test_county_groups_loaded(self, runner: EvaluationRunner) -> None:
        assert isinstance(runner.county_groups, dict)
        assert "bakken" in runner.county_groups

    def test_accuracy_metrics_loaded(self, runner: EvaluationRunner) -> None:
        assert "mape" in runner.accuracy_metrics
        assert "mae" in runner.accuracy_metrics

    def test_near_long_term_thresholds(self, runner: EvaluationRunner) -> None:
        assert runner.near_term_max == 5
        assert runner.long_term_min == 10


# ---------------------------------------------------------------------------
# run_accuracy_only
# ---------------------------------------------------------------------------


class TestRunAccuracyOnly:
    """Tests for EvaluationRunner.run_accuracy_only."""

    def test_returns_dataframe(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        diag = runner.run_accuracy_only(results_df)
        assert isinstance(diag, pd.DataFrame)
        assert len(diag) > 0

    def test_contains_expected_columns(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        diag = runner.run_accuracy_only(results_df)
        expected = {"metric_name", "horizon", "geography", "value"}
        assert expected.issubset(set(diag.columns))

    def test_metrics_match_config(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        diag = runner.run_accuracy_only(results_df)
        metric_names = set(diag["metric_name"].unique())
        for m in runner.accuracy_metrics:
            assert m in metric_names


# ---------------------------------------------------------------------------
# run_full_evaluation without projection_runner_fn
# ---------------------------------------------------------------------------


class TestRunFullEvaluationNoRunner:
    """Full evaluation without a projection runner (accuracy + realism only)."""

    def test_returns_expected_keys(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        assert "accuracy_diagnostics" in out
        assert "realism_diagnostics" in out
        assert "component_diagnostics" in out
        assert "comparison" in out
        assert "sensitivity" in out
        assert "scorecard" in out
        assert "figures" in out

    def test_sensitivity_is_none_without_runner(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        assert out["sensitivity"] is None

    def test_accuracy_diagnostics_populated(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        acc = out["accuracy_diagnostics"]
        assert isinstance(acc, pd.DataFrame)
        assert len(acc) > 0

    def test_scorecard_is_entry(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        assert isinstance(out["scorecard"], ScorecardEntry)

    def test_comparison_none_without_methods(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        assert out["comparison"] is None


# ---------------------------------------------------------------------------
# run_full_evaluation with projection_runner_fn
# ---------------------------------------------------------------------------


class TestRunFullEvaluationWithRunner:
    """Full evaluation with a mock projection runner (includes sensitivity)."""

    def test_sensitivity_populated(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        out = runner.run_full_evaluation(
            results_df, projection_runner_fn=mock_runner
        )
        sens = out["sensitivity"]
        assert sens is not None
        assert isinstance(sens, dict)

    def test_sensitivity_has_perturbation(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        out = runner.run_full_evaluation(
            results_df, projection_runner_fn=mock_runner
        )
        sens = out["sensitivity"]
        assert "perturbation" in sens
        pert = sens["perturbation"]
        assert isinstance(pert, pd.DataFrame)
        assert len(pert) > 0

    def test_sensitivity_has_stability_index(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        out = runner.run_full_evaluation(
            results_df, projection_runner_fn=mock_runner
        )
        sens = out["sensitivity"]
        assert "stability_index" in sens
        stab = sens["stability_index"]
        assert isinstance(stab, pd.DataFrame)
        assert len(stab) > 0

    def test_accuracy_still_computed_with_runner(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        out = runner.run_full_evaluation(
            results_df, projection_runner_fn=mock_runner
        )
        assert isinstance(out["accuracy_diagnostics"], pd.DataFrame)
        assert len(out["accuracy_diagnostics"]) > 0


# ---------------------------------------------------------------------------
# run_sensitivity_only
# ---------------------------------------------------------------------------


class TestRunSensitivityOnly:
    """Tests for EvaluationRunner.run_sensitivity_only."""

    def test_returns_dict_with_perturbation(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        result = runner.run_sensitivity_only(mock_runner, results_df)
        assert isinstance(result, dict)
        assert "perturbation" in result
        assert "stability_index" in result

    def test_perturbation_df_has_expected_columns(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        result = runner.run_sensitivity_only(mock_runner, results_df)
        pert = result["perturbation"]
        expected = {"component", "perturbation_pct", "direction", "sensitivity_index"}
        assert expected.issubset(set(pert.columns))

    def test_stability_index_has_geography(
        self, runner: EvaluationRunner, results_df: pd.DataFrame
    ) -> None:
        mock_runner = _make_mock_runner()
        result = runner.run_sensitivity_only(mock_runner, results_df)
        stab = result["stability_index"]
        assert "geography" in stab.columns
        assert "stability_rank" in stab.columns


# ---------------------------------------------------------------------------
# run_comparison
# ---------------------------------------------------------------------------


class TestRunComparison:
    """Tests for EvaluationRunner.run_comparison."""

    def test_returns_combined_diagnostics(
        self, runner: EvaluationRunner
    ) -> None:
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        method_a = _make_results_df(model_name="method_a", rng=rng1)
        method_b = _make_results_df(
            noise_scale=50.0, model_name="method_b", rng=rng2
        )
        result = runner.run_comparison({"method_a": method_a, "method_b": method_b})
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_both_methods_present(
        self, runner: EvaluationRunner
    ) -> None:
        method_a = _make_results_df(model_name="method_a")
        method_b = _make_results_df(model_name="method_b")
        result = runner.run_comparison({"method_a": method_a, "method_b": method_b})
        assert set(result["model_name"].unique()) == {"method_a", "method_b"}

    def test_delta_vs_baseline_column(
        self, runner: EvaluationRunner
    ) -> None:
        method_a = _make_results_df(model_name="method_a")
        method_b = _make_results_df(noise_scale=10.0, model_name="method_b")
        result = runner.run_comparison({"method_a": method_a, "method_b": method_b})
        assert "delta_vs_baseline" in result.columns
        # Baseline deltas should be zero for method_a
        baseline_rows = result[result["model_name"] == "method_a"]
        assert all(np.isclose(baseline_rows["delta_vs_baseline"], 0.0))

    def test_empty_dict_returns_empty(
        self, runner: EvaluationRunner
    ) -> None:
        result = runner.run_comparison({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """Tests for EvaluationRunner.generate_report."""

    def test_creates_output_directory(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        report_dir = tmp_path / "report"
        runner.generate_report(out, report_dir)
        assert report_dir.exists()

    def test_writes_accuracy_csv(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        report_dir = tmp_path / "report"
        runner.generate_report(out, report_dir)
        assert (report_dir / "accuracy_diagnostics.csv").exists()

    def test_writes_scorecard_summary(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        report_dir = tmp_path / "report"
        runner.generate_report(out, report_dir)
        assert (report_dir / "scorecard_summary.txt").exists()

    def test_writes_sensitivity_files_when_present(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        mock_runner = _make_mock_runner()
        out = runner.run_full_evaluation(
            results_df, projection_runner_fn=mock_runner
        )
        report_dir = tmp_path / "report"
        runner.generate_report(out, report_dir)
        assert (report_dir / "sensitivity_perturbation.csv").exists()
        assert (report_dir / "sensitivity_stability_index.csv").exists()

    def test_no_sensitivity_files_without_runner(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        report_dir = tmp_path / "report"
        runner.generate_report(out, report_dir)
        assert not (report_dir / "sensitivity_perturbation.csv").exists()
        assert not (report_dir / "sensitivity_stability_index.csv").exists()

    def test_output_dir_via_run_full_evaluation(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Passing output_dir to run_full_evaluation saves report."""
        report_dir = tmp_path / "auto_report"
        runner.run_full_evaluation(results_df, output_dir=report_dir)
        assert (report_dir / "accuracy_diagnostics.csv").exists()

    def test_returns_path(
        self, runner: EvaluationRunner, results_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = runner.run_full_evaluation(results_df)
        report_dir = tmp_path / "report"
        result = runner.generate_report(out, report_dir)
        assert result == report_dir
