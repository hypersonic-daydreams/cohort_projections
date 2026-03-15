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
import pytest

import scripts.analysis.observatory_dashboard as dashboard_launcher
from cohort_projections.analysis.observatory.dashboard.data_manager import (
    DashboardDataManager,
    _search_sidecar_path,
    build_comparison_rows,
    build_longitudinal_run_history,
    build_metric_delta_history,
    build_run_metadata_frame,
    build_search_session_frame,
    build_status_timeline,
    select_run_preset,
)
from cohort_projections.analysis.observatory.dashboard.tab_command_center import (
    _command_center_summary,
    _queue_health_snapshot,
    _search_progress_html,
)
from cohort_projections.analysis.observatory.dashboard.tab_history import (
    _history_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_horizon_bias import (
    _horizon_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_projection_ensemble import (
    _projection_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_scorecard import (
    _scorecard_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_sensitivity import (
    _sensitivity_takeaway_text,
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


def _run_label_factory(metadata: pd.DataFrame) -> dict[str, str]:
    """Return short run labels keyed by run ID."""
    return {
        str(row["run_id"]): str(row["legend_label"])
        for _, row in metadata.iterrows()
    }


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


def test_build_longitudinal_run_history_adds_delta_columns() -> None:
    """Run-history view should include selected-vs-reference headline deltas."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )

    history = build_longitudinal_run_history(metadata)
    by_run = history.set_index("run_id")

    assert "delta_county_mape_overall" in history.columns
    assert by_run.loc["br-passed", "delta_county_mape_overall"] == pytest.approx(-0.36)
    assert bool(by_run.loc["br-review", "review_flag"]) is True


def test_build_metric_delta_history_captures_long_metrics() -> None:
    """Long metric history should contain one row per run/metric delta."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )

    metric_history = build_metric_delta_history(scorecards, metadata, "br-champion")

    review_rows = metric_history[metric_history["run_id"] == "br-review"]
    overall = review_rows[review_rows["metric"] == "county_mape_overall"].iloc[0]
    assert overall["delta_value"] == pytest.approx(-0.02)
    assert "state_ape_recent_short" in metric_history["metric"].tolist()


def test_build_status_timeline_aggregates_outcomes_by_date() -> None:
    """Status timeline should count benchmark bundles by date and resolved status."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )

    timeline = build_status_timeline(metadata)
    counts = {
        (row["run_date_display"], row["status_code"]): row["run_count"]
        for _, row in timeline.iterrows()
    }

    assert counts[("2026-03-09", "champion")] == 1
    assert counts[("2026-03-09", "needs_human_review")] == 1
    assert counts[("2026-03-10", "passed_all_gates")] == 1


def test_history_takeaway_summarizes_longitudinal_state() -> None:
    """History summary should describe coverage, accepted challengers, and queue state."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )
    dm = SimpleNamespace(longitudinal_run_history=build_longitudinal_run_history(metadata))

    summary = _history_takeaway_text(cast(DashboardDataManager, dm))

    assert "**History coverage:** 3 benchmark bundles" in summary
    assert "**Best accepted challenger:** College Blend 70" in summary
    assert "**Champion baseline drift:**" in summary


def test_build_search_session_frame_summarizes_progress(tmp_path) -> None:
    """Autonomous-search session summaries should expose progress and artifacts."""
    session_dir = tmp_path / "search-one"
    session_dir.mkdir(parents=True)
    (session_dir / "session.yaml").write_text(
        """
search_id: search-one
created_at: 2026-03-15T10:00:00+00:00
updated_at: 2026-03-15T10:15:00+00:00
status: running
base_revision: HEAD
resolved_base_revision: abc123
summary:
  total: 10
  planned: 4
  running: 1
  completed: 4
  failed: 1
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "candidate_summary.csv").write_text("candidate_id,status\ncand-1,completed\n", encoding="utf-8")
    (session_dir / "search_report.md").write_text("# report\n", encoding="utf-8")

    frame = build_search_session_frame(tmp_path)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["search_id"] == "search-one"
    assert row["progress_count"] == 5
    assert row["progress_pct"] == 50.0
    assert row["candidate_summary_csv"].endswith("candidate_summary.csv")
    assert row["search_report_markdown"].endswith("search_report.md")
    assert row["process_status"] == "stopped"


def test_build_search_session_frame_includes_launch_sidecars(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dashboard sidecars should surface launching sessions before session.yaml exists."""
    search_id = "search-launching"
    meta_path = _search_sidecar_path(tmp_path, search_id, "dashboard_meta.yaml")
    meta_path.write_text(
        "search_id: search-launching\nlaunched_at: 2026-03-15T10:20:00+00:00\n",
        encoding="utf-8",
    )
    pid_path = _search_sidecar_path(tmp_path, search_id, "dashboard.pid")
    pid_path.write_text("43210\n", encoding="utf-8")
    log_path = _search_sidecar_path(tmp_path, search_id, "dashboard.log")
    log_path.write_text("launch output\n", encoding="utf-8")

    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager._is_search_process_running",
        lambda pid, sid: pid == 43210 and sid == search_id,
    )

    frame = build_search_session_frame(tmp_path)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["search_id"] == search_id
    assert row["status"] == "launching"
    assert bool(row["dashboard_process_running"]) is True
    assert row["dashboard_launch_log"].endswith("dashboard.log")


def test_search_progress_html_shows_selected_session_metrics() -> None:
    """Search progress HTML should surface counts and progress values."""
    row = pd.Series(
        {
            "search_id": "search-one",
            "status": "running",
            "progress_pct": 60.0,
            "total": 10,
            "planned": 3,
            "running": 1,
            "completed": 5,
            "failed": 1,
        }
    )

    html = _search_progress_html(row)

    assert "search-one" in html
    assert "Progress: 6/10" in html
    assert "width:60.0%" in html


def test_stop_search_session_terminates_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stopping a search session should target the recorded process group."""
    dm = SimpleNamespace(
        search_session_row=lambda search_id: pd.Series(
            {
                "dashboard_pid": 43210,
            }
        )
    )
    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager._is_search_process_running",
        lambda pid, sid: pid == 43210 and sid == "search-one",
    )
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.os.killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )

    message = DashboardDataManager.stop_search_session(
        cast(DashboardDataManager, dm),
        "search-one",
    )

    assert "Sent SIGTERM" in message
    assert killed


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


def test_scorecard_takeaway_prefers_best_selected_challenger() -> None:
    """Scorecard summary should highlight the strongest selected challenger."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )
    labels = _run_label_factory(metadata)
    dm = SimpleNamespace(
        champion_id="br-champion",
        comparison_rows=build_comparison_rows(scorecards, "br-champion"),
        run_metadata=metadata,
        ordered_run_ids=list(labels),
        run_label=lambda run_id, short=False: labels[str(run_id)],
    )

    summary = _scorecard_takeaway_text(
        ["br-review", "br-passed"],
        cast(DashboardDataManager, dm),
    )

    assert "**Best selected challenger:** College Blend 70" in summary
    assert "Recommended next step" in summary


def test_projection_takeaway_reports_endpoint_difference_vs_champion() -> None:
    """Projection summary should translate the curve comparison into a final-year delta."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )
    labels = _run_label_factory(metadata)
    projection_curves = pd.DataFrame(
        [
            {
                "origin_year": 2020,
                "method": "m2026",
                "year": 2055,
                "projected_state": 900000.0,
                "run_id": "br-champion",
            },
            {
                "origin_year": 2020,
                "method": "m2026r1",
                "year": 2055,
                "projected_state": 920000.0,
                "run_id": "br-passed",
            },
        ]
    )
    dm = SimpleNamespace(
        champion_id="br-champion",
        champion_method_id="m2026",
        projection_curves=projection_curves,
        run_metadata=metadata,
        run_label=lambda run_id, short=False: labels[str(run_id)],
    )

    summary = _projection_takeaway_text(
        "2020",
        ["br-passed"],
        True,
        cast(DashboardDataManager, dm),
    )

    assert "**Spotlight run:** College Blend 70" in summary
    assert "20,000 people higher than the champion" in summary


def test_horizon_takeaway_summarizes_long_horizon_error_and_bias() -> None:
    """Horizon summary should explain long-horizon error growth and bias direction."""
    annual_horizon_summary = pd.DataFrame(
        [
            {
                "horizon": 5,
                "method": "m2026",
                "mean_county_mape": 4.5,
                "mean_county_mpe": -0.2,
                "run_id": "br-champion",
            },
            {
                "horizon": 20,
                "method": "m2026",
                "mean_county_mape": 12.3,
                "mean_county_mpe": -1.4,
                "run_id": "br-champion",
            },
        ]
    )
    metadata = pd.DataFrame(
        [
            {
                "run_id": "br-champion",
                "selected_method_id": "m2026",
                "reference_method_id": "m2026",
                "legend_label": "Champion",
            }
        ]
    )
    dm = SimpleNamespace(
        annual_horizon_summary=annual_horizon_summary,
        champion_id="br-champion",
        champion_method_id="m2026",
        run_ids=["br-champion"],
        run_metadata=metadata,
        run_label=lambda run_id, short=False: "Champion",
    )

    summary = _horizon_takeaway_text(
        cast(DashboardDataManager, dm),
        "br-champion",
        "All",
    )

    assert "Average county error rises to 12.3% by horizon 20." in summary
    assert "under-projects on average" in summary


def test_sensitivity_takeaway_surfaces_recommendation_and_weakness_state() -> None:
    """Sensitivity summary should highlight the top recommendation and weakness status."""
    recommender = MagicMock()
    recommender.suggest_next_experiments.return_value = [
        SimpleNamespace(
            parameter="college_blend_factor",
            suggested_value=1.0,
            rationale="Largest tested improvement in college counties.",
        )
    ]
    recommender.identify_persistent_weaknesses.return_value = pd.DataFrame(
        [
            {
                "metric": "county_mape_rural",
                "best_challenger_delta": 0.12,
            }
        ]
    )
    dm = SimpleNamespace(
        recommender=recommender,
        sensitivity_summary=pd.DataFrame(
            [
                {"parameter": "college_blend_factor", "mape_swing": 0.8},
                {"parameter": "convergence_medium_hold", "mape_swing": 0.2},
            ]
        ),
    )

    summary = _sensitivity_takeaway_text(cast(DashboardDataManager, dm))

    assert "**Top recommended next experiment:** college_blend_factor -> 1.0." in summary
    assert "1 tracked metric(s) still have no improving challenger." in summary
    assert "Most influential tested parameter so far: `college_blend_factor`." in summary
