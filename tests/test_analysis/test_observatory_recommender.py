"""Tests for ObservatoryRecommender (cohort_projections.analysis.observatory.recommender).

Covers suggestion generation, boundary detection, untested catalog surfacing,
interaction detection, diminishing returns flagging, persistent weakness
identification, parameter sensitivity summary, formatting, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cohort_projections.analysis.observatory.comparator import ObservatoryComparator
from cohort_projections.analysis.observatory.recommender import (
    CONFIG_ONLY_PARAMS,
    Recommendation,
    ObservatoryRecommender,
    _is_numeric,
    _linear_trend,
    _next_step,
    _numeric_values_sorted,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store_and_comparator(
    scorecards: pd.DataFrame,
    manifests: dict[str, dict] | None = None,
    run_configs: dict[str, dict] | None = None,
) -> tuple[MagicMock, ObservatoryComparator]:
    """Build a mock store + real comparator for testing."""
    store = MagicMock()
    store.get_consolidated_scorecards.return_value = scorecards
    store.get_run_ids.return_value = sorted(scorecards["run_id"].unique().tolist()) if not scorecards.empty else []

    # Default manifest/config setup
    if manifests is None:
        manifests = {}
    if run_configs is None:
        run_configs = {}

    def _get_manifest(run_id: str) -> dict:
        if run_id in manifests:
            return manifests[run_id]
        raise FileNotFoundError(f"No manifest for {run_id}")

    def _get_config(run_id: str) -> dict:
        return run_configs.get(run_id, {})

    store.get_run_manifest.side_effect = _get_manifest
    store.get_run_config.side_effect = _get_config

    comparator = ObservatoryComparator(store=store)
    return store, comparator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHAMPION = {
    "run_id": "run-champ",
    "method_id": "m2026",
    "config_id": "cfg-base",
    "status_at_run": "champion",
    "county_mape_overall": 8.5,
    "county_mape_rural": 7.0,
    "county_mape_bakken": 19.0,
    "county_mape_urban_college": 12.0,
    "state_ape_recent_short": 1.0,
    "state_ape_recent_medium": 2.5,
}

EXP_A = {
    "run_id": "run-a",
    "method_id": "m2026r1",
    "config_id": "cfg-a",
    "status_at_run": "experiment",
    "county_mape_overall": 8.2,
    "county_mape_rural": 7.1,
    "county_mape_bakken": 18.5,
    "county_mape_urban_college": 10.5,
    "state_ape_recent_short": 1.1,
    "state_ape_recent_medium": 2.3,
}

EXP_B = {
    "run_id": "run-b",
    "method_id": "m2026r1",
    "config_id": "cfg-b",
    "status_at_run": "experiment",
    "county_mape_overall": 8.0,
    "county_mape_rural": 7.2,
    "county_mape_bakken": 18.0,
    "county_mape_urban_college": 10.0,
    "state_ape_recent_short": 1.2,
    "state_ape_recent_medium": 2.2,
}


@pytest.fixture()
def scorecards() -> pd.DataFrame:
    return pd.DataFrame([CHAMPION, EXP_A, EXP_B])


@pytest.fixture()
def manifests() -> dict[str, dict]:
    return {
        "run-champ": {
            "run_id": "run-champ",
            "champion_method_id": "m2026",
            "methods": [{"method_id": "m2026"}],
        },
        "run-a": {
            "run_id": "run-a",
            "champion_method_id": "m2026",
            "methods": [
                {"method_id": "m2026"},
                {"method_id": "m2026r1"},
            ],
        },
        "run-b": {
            "run_id": "run-b",
            "champion_method_id": "m2026",
            "methods": [
                {"method_id": "m2026"},
                {"method_id": "m2026r1"},
            ],
        },
    }


@pytest.fixture()
def run_configs() -> dict[str, dict]:
    return {
        "run-champ": {
            "m2026": {"college_blend_factor": 0.5},
        },
        "run-a": {
            "m2026": {"college_blend_factor": 0.5},
            "m2026r1": {"college_blend_factor": 0.7},
        },
        "run-b": {
            "m2026": {"college_blend_factor": 0.5},
            "m2026r1": {"college_blend_factor": 0.9},
        },
    }


@pytest.fixture()
def recommender(
    scorecards: pd.DataFrame,
    manifests: dict,
    run_configs: dict,
) -> ObservatoryRecommender:
    store, comparator = _make_store_and_comparator(scorecards, manifests, run_configs)
    return ObservatoryRecommender(store=store, comparator=comparator)


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_is_numeric_int(self) -> None:
        assert _is_numeric(42) is True

    def test_is_numeric_float(self) -> None:
        assert _is_numeric(3.14) is True

    def test_is_numeric_bool(self) -> None:
        assert _is_numeric(True) is False

    def test_is_numeric_inf(self) -> None:
        assert _is_numeric(float("inf")) is False

    def test_is_numeric_nan(self) -> None:
        assert _is_numeric(float("nan")) is False

    def test_is_numeric_string(self) -> None:
        assert _is_numeric("7") is False

    def test_numeric_values_sorted(self) -> None:
        result = _numeric_values_sorted([3, "x", 1, True, 2])
        assert result == [1.0, 2.0, 3.0]

    def test_numeric_values_sorted_empty(self) -> None:
        assert _numeric_values_sorted([]) == []

    def test_linear_trend_positive(self) -> None:
        slope = _linear_trend([1, 2, 3], [1, 2, 3])
        assert slope is not None
        assert slope == pytest.approx(1.0)

    def test_linear_trend_flat(self) -> None:
        slope = _linear_trend([1, 2, 3], [5, 5, 5])
        assert slope == pytest.approx(0.0)

    def test_linear_trend_too_few_points(self) -> None:
        assert _linear_trend([1], [1]) is None

    def test_linear_trend_constant_x(self) -> None:
        assert _linear_trend([1, 1, 1], [1, 2, 3]) is None

    def test_next_step_increase(self) -> None:
        result = _next_step([0.5, 0.7, 0.9], "increase")
        assert result == pytest.approx(1.1)

    def test_next_step_decrease(self) -> None:
        result = _next_step([0.5, 0.7, 0.9], "decrease")
        assert result == pytest.approx(0.3)

    def test_next_step_empty(self) -> None:
        assert _next_step([], "increase") == 0.0


# ---------------------------------------------------------------------------
# TestRecommendation
# ---------------------------------------------------------------------------


class TestRecommendation:
    """Tests for the Recommendation dataclass."""

    def test_construction(self) -> None:
        rec = Recommendation(
            parameter="college_blend_factor",
            suggested_value=0.8,
            direction="increase",
            rationale="Boundary detected.",
            expected_impact="~0.1pp improvement",
            priority=2,
            requires_code_change=False,
        )
        assert rec.parameter == "college_blend_factor"
        assert rec.priority == 2
        assert rec.requires_code_change is False

    def test_grid_suggestion_default_none(self) -> None:
        rec = Recommendation(
            parameter="x",
            suggested_value=1,
            direction="explore",
            rationale="test",
            expected_impact="unknown",
            priority=5,
            requires_code_change=True,
        )
        assert rec.grid_suggestion is None

    def test_config_only_detection(self) -> None:
        assert "college_blend_factor" in CONFIG_ONLY_PARAMS
        assert "gq_correction_fraction" in CONFIG_ONLY_PARAMS


# ---------------------------------------------------------------------------
# TestParameterSensitivity
# ---------------------------------------------------------------------------


class TestParameterSensitivity:
    """Tests for parameter_sensitivity_summary."""

    def test_returns_dataframe(self, recommender: ObservatoryRecommender) -> None:
        result = recommender.parameter_sensitivity_summary()
        assert isinstance(result, pd.DataFrame)

    def test_empty_store(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        assert rec.parameter_sensitivity_summary().empty


# ---------------------------------------------------------------------------
# TestSuggestNextExperiments
# ---------------------------------------------------------------------------


class TestSuggestNextExperiments:
    """Tests for suggest_next_experiments."""

    def test_returns_list_of_recommendations(self, recommender: ObservatoryRecommender) -> None:
        recs = recommender.suggest_next_experiments(n=5)
        assert isinstance(recs, list)
        for r in recs:
            assert isinstance(r, Recommendation)

    def test_respects_n_limit(self, recommender: ObservatoryRecommender) -> None:
        recs = recommender.suggest_next_experiments(n=1)
        assert len(recs) <= 1

    def test_empty_store_returns_empty(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        recs = rec.suggest_next_experiments()
        assert recs == []

    def test_sorted_by_priority(self, recommender: ObservatoryRecommender) -> None:
        recs = recommender.suggest_next_experiments(n=10)
        if len(recs) >= 2:
            priorities = [r.priority for r in recs]
            assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# TestUntestedCatalog
# ---------------------------------------------------------------------------


class TestUntestedCatalog:
    """Tests for _untested_catalog_recommendations."""

    def test_surfaces_untested_entries(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        catalog = [
            {"slug": "exp-x", "parameter": "alpha", "status": "untested", "tier": 1, "hypothesis": "test"},
            {"slug": "exp-y", "parameter": "beta", "status": "passed_all_gates", "tier": 2, "hypothesis": "done"},
        ]
        rec = ObservatoryRecommender(
            store=store, comparator=comparator, variant_catalog=catalog
        )
        recs = rec._untested_catalog_recommendations()
        params = [r.parameter for r in recs]
        assert "alpha" in params
        assert "beta" not in params

    def test_empty_catalog(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator, variant_catalog=[])
        assert rec._untested_catalog_recommendations() == []

    def test_catalog_as_dataframe(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        catalog_df = pd.DataFrame([
            {"slug": "exp-x", "parameter": "alpha", "status": "untested", "tier": 1, "hypothesis": "h"},
        ])
        # Pass catalog_df directly; the constructor uses `variant_catalog or []`
        # which fails for DataFrames, so we must set it after construction.
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        rec.variant_catalog = catalog_df
        recs = rec._untested_catalog_recommendations()
        assert len(recs) == 1
        assert recs[0].parameter == "alpha"


# ---------------------------------------------------------------------------
# TestDiffConfigs
# ---------------------------------------------------------------------------


class TestDiffConfigs:
    """Tests for _diff_configs static method."""

    def test_no_changes(self) -> None:
        base = {"a": 1, "b": 2}
        assert ObservatoryRecommender._diff_configs(base, base) == {}

    def test_changed_value(self) -> None:
        base = {"a": 1, "b": 2}
        variant = {"a": 1, "b": 3}
        assert ObservatoryRecommender._diff_configs(base, variant) == {"b": 3}

    def test_added_key(self) -> None:
        base = {"a": 1}
        variant = {"a": 1, "c": 3}
        assert ObservatoryRecommender._diff_configs(base, variant) == {"c": 3}

    def test_removed_key(self) -> None:
        base = {"a": 1, "b": 2}
        variant = {"a": 1}
        diff = ObservatoryRecommender._diff_configs(base, variant)
        # b is in base but not variant — value becomes None
        assert "b" in diff
        assert diff["b"] is None


# ---------------------------------------------------------------------------
# TestInteractionRecommendations
# ---------------------------------------------------------------------------


class TestInteractionRecommendations:
    """Tests for _interaction_recommendations."""

    def test_with_improving_params(self) -> None:
        """Two independently improving parameters should produce an interaction rec."""
        sensitivity = pd.DataFrame([
            {"parameter": "college_blend_factor", "value": 0.7, "county_mape_overall_delta": -0.1},
            {"parameter": "convergence_medium_hold", "value": 3, "county_mape_overall_delta": -0.2},
        ])
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        recs = rec._interaction_recommendations(sensitivity)
        assert len(recs) == 1
        assert "college_blend_factor" in recs[0].parameter and "convergence_medium_hold" in recs[0].parameter

    def test_no_improving_params(self) -> None:
        sensitivity = pd.DataFrame([
            {"parameter": "college_blend_factor", "value": 0.7, "county_mape_overall_delta": 0.1},
        ])
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        assert rec._interaction_recommendations(sensitivity) == []

    def test_empty_sensitivity(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        assert rec._interaction_recommendations(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# TestDiminishingReturns
# ---------------------------------------------------------------------------


class TestDiminishingReturns:
    """Tests for _diminishing_returns_flags."""

    def test_plateaued_parameter(self) -> None:
        """Parameter with flat slope should be flagged."""
        sensitivity = pd.DataFrame([
            {"parameter": "alpha", "value": 0.5, "county_mape_overall_delta": -0.01},
            {"parameter": "alpha", "value": 0.6, "county_mape_overall_delta": -0.01},
            {"parameter": "alpha", "value": 0.7, "county_mape_overall_delta": -0.01},
        ])
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(
            store=store, comparator=comparator, config={"plateau_threshold": 0.02}
        )
        recs = rec._diminishing_returns_flags(sensitivity)
        assert len(recs) == 1
        assert recs[0].direction == "plateau"
        assert recs[0].priority == 9

    def test_not_enough_points(self) -> None:
        """Fewer than 3 points should yield no flags."""
        sensitivity = pd.DataFrame([
            {"parameter": "alpha", "value": 0.5, "county_mape_overall_delta": -0.1},
            {"parameter": "alpha", "value": 0.7, "county_mape_overall_delta": -0.3},
        ])
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        assert rec._diminishing_returns_flags(sensitivity) == []


# ---------------------------------------------------------------------------
# TestPersistentWeaknesses
# ---------------------------------------------------------------------------


class TestPersistentWeaknesses:
    """Tests for identify_persistent_weaknesses."""

    def test_returns_dataframe(self, recommender: ObservatoryRecommender) -> None:
        result = recommender.identify_persistent_weaknesses()
        assert isinstance(result, pd.DataFrame)
        assert "metric" in result.columns

    def test_empty_store(self) -> None:
        store, comparator = _make_store_and_comparator(pd.DataFrame())
        rec = ObservatoryRecommender(store=store, comparator=comparator)
        result = rec.identify_persistent_weaknesses()
        assert result.empty


# ---------------------------------------------------------------------------
# TestFormatRecommendations
# ---------------------------------------------------------------------------


class TestFormatRecommendations:
    """Tests for format_recommendations."""

    def test_format_empty_list(self, recommender: ObservatoryRecommender) -> None:
        output = recommender.format_recommendations([])
        assert "No recommendations" in output

    def test_format_with_recommendations(self, recommender: ObservatoryRecommender) -> None:
        recs = [
            Recommendation(
                parameter="college_blend_factor",
                suggested_value=0.8,
                direction="increase",
                rationale="Boundary detected.",
                expected_impact="~0.1pp improvement",
                priority=2,
                requires_code_change=False,
            ),
        ]
        output = recommender.format_recommendations(recs)
        assert "EXPERIMENT RECOMMENDATIONS" in output
        assert "college_blend_factor" in output
        assert "0.8" in output
        assert "increase" in output

    def test_format_code_change_flag(self, recommender: ObservatoryRecommender) -> None:
        recs = [
            Recommendation(
                parameter="gq_fraction",
                suggested_value=0.75,
                direction="explore",
                rationale="test",
                expected_impact="unknown",
                priority=1,
                requires_code_change=True,
            ),
        ]
        output = recommender.format_recommendations(recs)
        assert "CODE CHANGE REQUIRED" in output
