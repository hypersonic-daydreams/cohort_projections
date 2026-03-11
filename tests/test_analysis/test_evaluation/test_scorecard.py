"""Tests for the model scorecard module."""

from __future__ import annotations

import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.data_structures import ScorecardEntry
from cohort_projections.analysis.evaluation.scorecard import ModelScorecard


@pytest.fixture()
def default_config() -> dict:
    return {
        "scorecard_weights": {
            "near_term_accuracy": 0.25,
            "long_term_accuracy": 0.25,
            "bias_calibration": 0.15,
            "age_structure_realism": 0.15,
            "robustness_stability": 0.10,
            "interpretability": 0.10,
        },
        "near_term_max_horizon": 5,
        "long_term_min_horizon": 10,
    }


@pytest.fixture()
def accuracy_df() -> pd.DataFrame:
    """Diagnostics with MAPE and MSPE across several horizons."""
    rows = []
    for h in [1, 3, 5, 10, 15, 20]:
        rows.append(
            {"metric_name": "mape", "horizon": h, "value": 1.0 + h * 0.5}
        )
        rows.append(
            {
                "metric_name": "mean_signed_percentage_error",
                "horizon": h,
                "value": 0.2,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def realism_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric_name": "jsd", "horizon": 5, "value": 0.02},
            {"metric_name": "jsd", "horizon": 10, "value": 0.05},
        ]
    )


class TestModelScorecard:
    def test_build_scorecard_returns_entry(
        self, default_config: dict, accuracy_df: pd.DataFrame
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(
            accuracy_df, run_id="run1", model_name="m2026"
        )
        assert isinstance(entry, ScorecardEntry)
        assert entry.run_id == "run1"
        assert entry.model_name == "m2026"

    def test_near_term_accuracy_uses_correct_horizons(
        self, default_config: dict, accuracy_df: pd.DataFrame
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(accuracy_df)
        # Near-term horizons: 1, 3, 5 -> MAPE values: 1.5, 2.5, 3.5 -> mean 2.5
        assert entry.near_term_accuracy == pytest.approx(2.5, abs=0.01)

    def test_long_term_accuracy_uses_correct_horizons(
        self, default_config: dict, accuracy_df: pd.DataFrame
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(accuracy_df)
        # Long-term horizons: 10, 15, 20 -> MAPE values: 6.0, 8.5, 11.0 -> mean 8.5
        assert entry.long_term_accuracy == pytest.approx(8.5, abs=0.01)

    def test_bias_calibration(
        self, default_config: dict, accuracy_df: pd.DataFrame
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(accuracy_df)
        assert entry.bias_calibration == pytest.approx(0.2, abs=0.01)

    def test_age_realism_with_jsd(
        self,
        default_config: dict,
        accuracy_df: pd.DataFrame,
        realism_df: pd.DataFrame,
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(accuracy_df, realism_diagnostics=realism_df)
        # mean JSD = 0.035, realism = 1 - 0.035 = 0.965
        assert entry.age_structure_realism == pytest.approx(0.965, abs=0.01)

    def test_age_realism_default_when_none(
        self, default_config: dict, accuracy_df: pd.DataFrame
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(accuracy_df, realism_diagnostics=None)
        assert entry.age_structure_realism == pytest.approx(0.5)

    def test_composite_score_calculated(
        self, default_config: dict, accuracy_df: pd.DataFrame
    ) -> None:
        sc = ModelScorecard(default_config)
        entry = sc.build_scorecard(accuracy_df)
        assert entry.composite_score > 0
        # Verify it matches recomputation
        recomputed = sc.compute_composite(entry)
        assert entry.composite_score == pytest.approx(recomputed)

    def test_composite_lower_is_better(self, default_config: dict) -> None:
        sc = ModelScorecard(default_config)
        good = ScorecardEntry(
            run_id="good",
            model_name="good",
            near_term_accuracy=1.0,
            long_term_accuracy=2.0,
            bias_calibration=0.1,
            age_structure_realism=0.95,
            robustness_stability=0.9,
            interpretability=0.8,
        )
        bad = ScorecardEntry(
            run_id="bad",
            model_name="bad",
            near_term_accuracy=5.0,
            long_term_accuracy=10.0,
            bias_calibration=2.0,
            age_structure_realism=0.3,
            robustness_stability=0.3,
            interpretability=0.3,
        )
        assert sc.compute_composite(good) < sc.compute_composite(bad)

    def test_compare_scorecards(self, default_config: dict) -> None:
        sc = ModelScorecard(default_config)
        entries = [
            ScorecardEntry("r1", "m1", 1.0, 2.0, 0.1, 0.9, 0.8, 0.7, 0.0),
            ScorecardEntry("r2", "m2", 2.0, 4.0, 0.5, 0.7, 0.6, 0.5, 0.0),
        ]
        df = sc.compare_scorecards(entries)
        assert len(df) == 2
        assert list(df.columns) == [
            "run_id",
            "model_name",
            "near_term_accuracy",
            "long_term_accuracy",
            "bias_calibration",
            "age_structure_realism",
            "robustness_stability",
            "interpretability",
            "composite_score",
        ]

    def test_render_summary_contains_model_names(
        self, default_config: dict
    ) -> None:
        sc = ModelScorecard(default_config)
        entries = [
            ScorecardEntry("r1", "model_alpha", 1.0, 2.0, 0.0, 0.9, 0.8, 0.7),
            ScorecardEntry("r2", "model_beta", 3.0, 5.0, 0.5, 0.6, 0.5, 0.4),
        ]
        txt = sc.render_summary(entries)
        assert "model_alpha" in txt
        assert "model_beta" in txt
        assert "Best composite" in txt
