"""Tests for evaluation visualization module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.visualization import (
    MATPLOTLIB_AVAILABLE,
    plot_age_divergence,
    plot_bias_map,
    plot_component_blame,
    plot_county_horizon_heatmap,
    plot_horizon_profile,
    plot_parameter_response,
    plot_stability_scatter,
    save_evaluation_report,
)

pytestmark = pytest.mark.skipif(
    not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed"
)


@pytest.fixture()
def diagnostics_df() -> pd.DataFrame:
    """Synthetic diagnostics DataFrame covering two models and multiple horizons."""
    rows = []
    for model in ["m2026", "m2026r1"]:
        for county in ["38017", "38035", "38101"]:
            for h in [1, 3, 5, 10, 15]:
                rows.append(
                    {
                        "run_id": f"run_{model}",
                        "model_name": model,
                        "metric_name": "mape",
                        "metric_group": "accuracy",
                        "geography": county,
                        "geography_group": "urban_college",
                        "target": "population",
                        "horizon": h,
                        "value": 1.0 + h * 0.3 + (0.5 if model == "m2026" else 0.0),
                    }
                )
                rows.append(
                    {
                        "run_id": f"run_{model}",
                        "model_name": model,
                        "metric_name": "mean_signed_percentage_error",
                        "metric_group": "accuracy",
                        "geography": county,
                        "geography_group": "urban_college",
                        "target": "population",
                        "horizon": h,
                        "value": 0.3 if county == "38017" else -0.2,
                    }
                )
                rows.append(
                    {
                        "run_id": f"run_{model}",
                        "model_name": model,
                        "metric_name": "jsd",
                        "metric_group": "realism",
                        "geography": county,
                        "geography_group": "urban_college",
                        "target": "age_distribution",
                        "horizon": h,
                        "value": 0.01 + h * 0.002,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def component_df() -> pd.DataFrame:
    rows = []
    for h in [1, 3, 5, 10]:
        for comp in ["births", "deaths", "net_migration"]:
            rows.append(
                {
                    "horizon": h,
                    "component": comp,
                    "projected_component_value": 1000 + h * 10,
                    "actual_component_value": 1000 + h * 8,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def sweep_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "blend_factor": [0.3, 0.5, 0.7, 0.9],
            "mape": [4.2, 3.8, 3.1, 3.0],
        }
    )


class TestHorizonProfile:
    def test_returns_figure(self, diagnostics_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_horizon_profile(diagnostics_df, "mape")
        assert isinstance(fig, Figure)

    def test_subset_methods(self, diagnostics_df: pd.DataFrame) -> None:
        fig = plot_horizon_profile(diagnostics_df, "mape", methods=["m2026"])
        axes = fig.get_axes()
        assert len(axes) == 1


class TestCountyHorizonHeatmap:
    def test_returns_figure(self, diagnostics_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_county_horizon_heatmap(diagnostics_df, "mape")
        assert isinstance(fig, Figure)


class TestBiasMap:
    def test_returns_figure(self, diagnostics_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_bias_map(diagnostics_df)
        assert isinstance(fig, Figure)


class TestComponentBlame:
    def test_returns_figure(self, component_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_component_blame(component_df)
        assert isinstance(fig, Figure)


class TestAgeDivergence:
    def test_returns_figure(self, diagnostics_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_age_divergence(diagnostics_df)
        assert isinstance(fig, Figure)


class TestParameterResponse:
    def test_returns_figure(self, sweep_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_parameter_response(sweep_df, "blend_factor", "mape")
        assert isinstance(fig, Figure)


class TestStabilityScatter:
    def test_returns_figure(self, diagnostics_df: pd.DataFrame) -> None:
        from matplotlib.figure import Figure

        fig = plot_stability_scatter(diagnostics_df)
        assert isinstance(fig, Figure)


class TestSaveReport:
    def test_saves_files(
        self, tmp_path: Path, diagnostics_df: pd.DataFrame
    ) -> None:
        fig = plot_horizon_profile(diagnostics_df, "mape")
        out = save_evaluation_report(
            tmp_path / "report",
            diagnostics_df,
            {"horizon_profile": fig},
        )
        assert (out / "diagnostics_summary.csv").exists()
        assert (out / "horizon_profile.png").exists()
