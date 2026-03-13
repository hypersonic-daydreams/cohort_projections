"""Tests for Observatory dashboard helper logic.

These tests cover the run-metadata layer that powers readable selectors,
bundle presets, comparison-row selection, and queue-health summaries in the
Panel dashboard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pandas as pd

import scripts.analysis.observatory_dashboard as dashboard_launcher
from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
    build_comparison_rows,
    build_run_metadata_frame,
    select_run_preset,
)
from cohort_projections.analysis.observatory.dashboard.tab_command_center import (
    _command_center_summary,
    _queue_health_snapshot,
)


def _build_fixture_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return synthetic index, scorecard, and experiment-log frames."""
    index = pd.DataFrame(
        [
            {
                "run_id": "br-champion",
                "run_date": "20260309",
                "benchmark_label": "college_fix_head_to_head",
                "decision_status": "pending",
            },
            {
                "run_id": "br-review",
                "run_date": "20260309",
                "benchmark_label": "convergence-medium-hold-5",
                "decision_status": "pending",
            },
            {
                "run_id": "br-passed",
                "run_date": "20260310",
                "benchmark_label": "college-blend-70",
                "decision_status": "approved",
            },
        ]
    )

    scorecards = pd.DataFrame(
        [
            {
                "run_id": "br-champion",
                "method_id": "m2026",
                "config_id": "cfg-20260223-baseline-v1",
                "status_at_run": "champion",
                "county_mape_overall": 8.86,
                "state_ape_recent_short": 0.95,
            },
            {
                "run_id": "br-champion",
                "method_id": "m2026r1",
                "config_id": "cfg-20260309-college-fix-v1",
                "status_at_run": "candidate",
                "county_mape_overall": 8.70,
                "state_ape_recent_short": 0.87,
            },
            {
                "run_id": "br-review",
                "method_id": "m2026",
                "config_id": "cfg-20260223-baseline-v1",
                "status_at_run": "champion",
                "county_mape_overall": 8.86,
                "state_ape_recent_short": 0.95,
            },
            {
                "run_id": "br-review",
                "method_id": "m2026r1",
                "config_id": "cfg-20260309-medium-hold-5-v1",
                "status_at_run": "experiment",
                "county_mape_overall": 8.84,
                "state_ape_recent_short": 0.90,
            },
            {
                "run_id": "br-passed",
                "method_id": "m2026",
                "config_id": "cfg-20260223-baseline-v1",
                "status_at_run": "champion",
                "county_mape_overall": 8.86,
                "state_ape_recent_short": 0.95,
            },
            {
                "run_id": "br-passed",
                "method_id": "m2026r1",
                "config_id": "cfg-20260310-college-blend-70-v1",
                "status_at_run": "experiment",
                "county_mape_overall": 8.50,
                "state_ape_recent_short": 0.82,
            },
        ]
    )

    experiment_log = pd.DataFrame(
        [
            {
                "run_id": "br-review",
                "experiment_id": "exp-20260309-convergence-medium-hold-5",
                "outcome": "needs_human_review",
                "next_action": "flag_for_review",
            },
            {
                "run_id": "br-passed",
                "experiment_id": "exp-20260310-college-blend-70",
                "outcome": "passed_all_gates",
                "next_action": "promote",
            },
        ]
    )

    return index, scorecards, experiment_log


def test_build_comparison_rows_prefers_variant_rows_except_for_champion_bundle() -> None:
    """Comparison rows should use the challenger for non-champion bundles."""
    _, scorecards, _ = _build_fixture_frames()

    result = build_comparison_rows(scorecards, champion_id="br-champion")

    by_run = {row["run_id"]: row for _, row in result.iterrows()}
    assert by_run["br-champion"]["method_id"] == "m2026"
    assert by_run["br-review"]["method_id"] == "m2026r1"
    assert by_run["br-passed"]["method_id"] == "m2026r1"


def test_build_run_metadata_frame_creates_readable_labels_and_statuses() -> None:
    """Run metadata should expose readable labels for selectors and legends."""
    index, scorecards, experiment_log = _build_fixture_frames()

    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )

    by_run = metadata.set_index("run_id")
    assert by_run.loc["br-champion", "status_code"] == "champion"
    assert by_run.loc["br-review", "status_code"] == "needs_human_review"
    assert by_run.loc["br-passed", "status_code"] == "passed_all_gates"
    assert "[Champion]" in by_run.loc["br-champion", "selector_label"]
    assert "College Blend 70" in by_run.loc["br-passed", "selector_label"]
    assert by_run.loc["br-champion", "legend_label"] == "Champion"


def test_select_run_preset_returns_expected_groups() -> None:
    """Presets should surface top challengers, review queues, and latest runs."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )

    assert select_run_preset(metadata, "Champion vs top challengers", "br-champion", limit=2) == [
        "br-passed",
        "br-review",
    ]
    assert select_run_preset(metadata, "Needs review", "br-champion") == ["br-review"]
    assert select_run_preset(metadata, "Passed only", "br-champion") == ["br-passed"]
    assert select_run_preset(metadata, "Latest 3", "br-champion", limit=1) == ["br-passed"]


def test_queue_health_snapshot_reports_runnable_and_blocked_work() -> None:
    """Queue health should surface runnable variants, blocked grids, and review load."""
    catalog = MagicMock()
    catalog.get_inventory_summary.return_value = {
        "untested_runnable": 3,
        "untested_requires_code_change": 2,
        "grid_blocked": 1,
        "grid_blocked_ids": ["dampening-sweep"],
    }
    recommender = MagicMock()
    recommender.suggest_next_experiments.return_value = [
        SimpleNamespace(requires_code_change=False),
        SimpleNamespace(requires_code_change=True),
        SimpleNamespace(requires_code_change=False),
    ]
    dm = SimpleNamespace(
        catalog=catalog,
        recommender=recommender,
        run_metadata=pd.DataFrame({"status_code": ["needs_human_review", "passed_all_gates"]}),
    )

    snapshot = _queue_health_snapshot(cast(DashboardDataManager, dm))
    assert snapshot["untested_runnable"] == 3
    assert snapshot["untested_requires_code_change"] == 2
    assert snapshot["grid_blocked"] == 1
    assert snapshot["grid_blocked_ids"] == ["dampening-sweep"]
    assert snapshot["review_queue"] == 1
    assert snapshot["runnable_recommendations"] == 2


def test_command_center_summary_surfaces_current_decision_context() -> None:
    """The first-run summary should explain champion, challenger, review load, and next step."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )
    recommender = MagicMock()
    recommender.suggest_next_experiments.return_value = [
        SimpleNamespace(
            parameter="college_blend_factor",
            suggested_value=1.0,
        )
    ]
    dm = SimpleNamespace(
        champion_id="br-champion",
        recommender=recommender,
        run_metadata=metadata,
    )

    summary = _command_center_summary(cast(DashboardDataManager, dm))

    assert "Current champion:" in summary
    assert "Best tested challenger: College Blend 70" in summary
    assert "1 run(s) currently need human review." in summary
    assert "college_blend_factor -> 1.0" in summary


def test_build_dashboard_returns_fresh_instances(monkeypatch) -> None:
    """Dashboard builder should return a fresh app instance for each session."""
    created_dms: list[object] = []

    class DummyDM:
        pass

    def fake_create_app(dm: object) -> object:
        created_dms.append(dm)
        return {"dm_id": id(dm)}

    monkeypatch.setattr(dashboard_launcher, "_configure_panel_runtime", lambda: None)

    import cohort_projections.analysis.observatory.dashboard.app as app_module
    import cohort_projections.analysis.observatory.dashboard.data_manager as dm_module

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(dm_module, "DashboardDataManager", DummyDM)

    first = dashboard_launcher.build_dashboard()
    second = dashboard_launcher.build_dashboard()

    assert first is not second
    assert len(created_dms) == 2
    assert created_dms[0] is not created_dms[1]
