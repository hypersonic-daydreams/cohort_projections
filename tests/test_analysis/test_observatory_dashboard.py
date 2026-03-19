"""Tests for Observatory dashboard helper logic.

These tests cover the run-metadata layer that powers readable selectors,
bundle presets, comparison-row selection, and queue-health summaries in the
Panel dashboard.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import panel as pn
import pytest

import scripts.analysis.observatory_dashboard as dashboard_launcher
from cohort_projections.analysis.observatory.dashboard.app import (
    _resolve_stepper_state,
)
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
    _build_onboarding_card,
    _command_center_summary,
    _enter_guided_review,
    _queue_health_snapshot,
    _search_progress_html,
)
from cohort_projections.analysis.observatory.dashboard.tab_decision_brief import (
    _decision_brief_markdown,
    _verdict_strip_html,
    build_decision_brief_tab,
)
from cohort_projections.analysis.observatory.dashboard.tab_history import (
    _history_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_horizon_bias import (
    _horizon_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_projection_ensemble import (
    _build_projection_focus_summary,
    _projection_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_scorecard import (
    _build_delta_bar_chart,
    _scorecard_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.tab_sensitivity import (
    _build_recommendation_cards,
    _sensitivity_takeaway_text,
)
from cohort_projections.analysis.observatory.dashboard.theme import (
    build_tabs_stylesheet,
    resolve_layout_mode,
)
from cohort_projections.analysis.observatory.decision_support import (
    build_benchmark_decision_brief,
    build_candidate_decision_summary,
    build_search_candidate_rows,
    build_search_session_summary,
    finalize_candidate_recommendations,
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
    return {str(row["run_id"]): str(row["legend_label"]) for _, row in metadata.iterrows()}


class _FakeResultsStore:
    """Minimal ResultsStore test double for DashboardDataManager tests."""

    def __init__(
        self,
        *,
        history_dir,
        config: dict | None = None,
        index: pd.DataFrame | None = None,
        scorecards: pd.DataFrame | None = None,
        experiment_log: pd.DataFrame | None = None,
        run_ids: list[str] | None = None,
    ) -> None:
        self._history_dir = history_dir
        self._config = config or {"recommender": {}}
        self._index = index if index is not None else pd.DataFrame()
        self._scorecards = scorecards if scorecards is not None else pd.DataFrame()
        self._experiment_log = experiment_log if experiment_log is not None else pd.DataFrame()
        self._run_ids = run_ids or []
        self.refreshed = False

    def get_index(self) -> pd.DataFrame:
        return self._index

    def get_experiment_log(self) -> pd.DataFrame:
        return self._experiment_log

    def get_consolidated_scorecards(self) -> pd.DataFrame:
        return self._scorecards

    def get_consolidated_county_metrics(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_consolidated_state_metrics(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_consolidated_projection_curves(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_consolidated_sensitivity_summary(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_run_ids(self) -> list[str]:
        return list(self._run_ids)

    def refresh(self) -> None:
        self.refreshed = True


def _make_dashboard_manager(
    monkeypatch: pytest.MonkeyPatch,
    *,
    history_dir,
    session_root,
    index: pd.DataFrame | None = None,
    scorecards: pd.DataFrame | None = None,
    experiment_log: pd.DataFrame | None = None,
    run_ids: list[str] | None = None,
    catalog: object | None = None,
    recommendations: list[object] | None = None,
) -> DashboardDataManager:
    """Build a DashboardDataManager with isolated fake dependencies."""

    class DummyComparator:
        def __init__(self, store, config=None) -> None:
            self.store = store

        def _resolve_champion(self, scorecards: pd.DataFrame) -> str | None:
            if scorecards.empty or "run_id" not in scorecards.columns:
                return None
            champion = scorecards[
                scorecards.get("status_at_run", pd.Series(dtype=object)).astype(str).str.lower()
                == "champion"
            ]
            if champion.empty:
                return None
            return str(champion.iloc[0]["run_id"])

        def best_variant_per_group(self) -> dict[str, dict[str, object]]:
            return {}

        def county_group_impact(self) -> pd.DataFrame:
            return pd.DataFrame()

    class DummyRecommender:
        def __init__(self, store, comparator=None, config=None, bounds_catalog=None) -> None:
            self._recommendations = list(recommendations or [])

        def suggest_next_experiments(self, limit: int = 5):
            return self._recommendations[:limit]

        def identify_persistent_weaknesses(self) -> pd.DataFrame:
            return pd.DataFrame()

        def parameter_sensitivity_summary(self) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.ObservatoryComparator",
        DummyComparator,
    )
    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.ObservatoryRecommender",
        DummyRecommender,
    )
    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.load_search_policy",
        lambda: SimpleNamespace(
            project_root=session_root.parent,
            session_root=session_root,
            default_run_budget=8,
            default_max_pending=3,
            default_max_recommended=2,
            include_recipe_catalog=True,
        ),
    )
    monkeypatch.setattr(
        DashboardDataManager,
        "_try_build_catalog",
        lambda self: catalog,
    )

    store = _FakeResultsStore(
        history_dir=history_dir,
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        run_ids=run_ids,
    )
    return DashboardDataManager(store=cast(Any, store), config=store._config)


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


def test_command_center_summary_uses_search_session_brief_when_benchmark_data_missing() -> None:
    """Command Center summary should use the session brief when benchmark review is unavailable."""
    dm = SimpleNamespace(
        decision_brief={
            "source": "search_session",
            "session_headline": "Search finished, but none of the candidates produced a usable benchmark.",
            "session_recommendation": "Restore the missing input data and rerun the blocked candidates.",
            "session_blocker_summary": "Benchmark history index is missing.",
        }
    )

    summary = _command_center_summary(cast(DashboardDataManager, dm))

    assert "none of the candidates produced a usable benchmark" in summary
    assert "Blocker: Benchmark history index is missing." in summary


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


def test_resolve_layout_mode_detects_portrait_orientation() -> None:
    """Portrait layout should activate for tall desktop viewports, not only narrow ones."""
    assert resolve_layout_mode(1440, 2560) == "portrait"
    assert resolve_layout_mode(2560, 1440) == "standard"


def test_build_tabs_stylesheet_deemphasizes_non_guided_tabs_in_review_mode() -> None:
    """Guided review mode should visually prioritize the review-sequence tabs."""
    stylesheet = build_tabs_stylesheet(review_mode=True)

    assert "opacity: 0.62" in stylesheet
    assert ".bk-tab:nth-child(2)" in stylesheet
    assert ".bk-tab:nth-child(7)" in stylesheet


def test_onboarding_card_explains_first_action_path() -> None:
    """First-run Command Center state should explain the primary exploration flow."""
    dm = SimpleNamespace(run_ids=[], benchmark_history_snapshot={"index_present": False})

    card = _build_onboarding_card(cast(DashboardDataManager, dm))
    body = "\n".join(str(getattr(obj, "object", "")) for obj in card.objects)

    assert "Projection Observatory compares projection variants" in body
    assert "What Start Exploring produces" in body
    assert "Where blocked results go" in body


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
    (session_dir / "candidate_summary.csv").write_text(
        "candidate_id,status\ncand-1,completed\n", encoding="utf-8"
    )
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


def test_build_search_session_frame_includes_launch_sidecars(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_build_search_session_frame_ignores_invalid_yaml_payloads(tmp_path) -> None:
    """Invalid YAML shapes should not crash session discovery."""
    session_dir = tmp_path / "search-invalid"
    session_dir.mkdir(parents=True)
    (session_dir / "session.yaml").write_text("- not-a-mapping\n", encoding="utf-8")

    frame = build_search_session_frame(tmp_path)

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["search_id"] == "search-invalid"
    assert row["status"] == "unknown"
    assert row["total"] == 0


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


def test_search_progress_html_marks_stopped_failed_session() -> None:
    """Stopped sessions with only failures should render the STOPPED state."""
    row = pd.Series(
        {
            "search_id": "search-failed",
            "status": "running",
            "progress_pct": 100.0,
            "total": 4,
            "planned": 0,
            "running": 0,
            "completed": 0,
            "failed": 4,
            "dashboard_process_running": False,
        }
    )

    html = _search_progress_html(row)

    assert "STOPPED" in html
    assert "Failed: 4" in html


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


def test_stop_search_session_reports_missing_pid() -> None:
    """Missing dashboard PIDs should produce a clear stop message."""
    dm = SimpleNamespace(search_session_row=lambda search_id: pd.Series({"dashboard_pid": None}))

    message = DashboardDataManager.stop_search_session(
        cast(DashboardDataManager, dm),
        "search-one",
    )

    assert "No dashboard-managed process PID" in message


def test_build_search_candidate_rows_classifies_missing_input_as_blocked(
    tmp_path,
) -> None:
    """Candidate summaries should surface missing shared-data inputs as blockers."""
    log_dir = tmp_path / "data" / "analysis" / "experiments" / "search_runs" / "search-one" / "logs"
    log_dir.mkdir(parents=True)
    log_rel = "data/analysis/experiments/search_runs/search-one/logs/cand-1.stdout.log"
    (tmp_path / log_rel).write_text(
        "\n".join(
            [
                "Benchmark suite failed (rc=1)",
                "FileNotFoundError: PEP 2020-2024 file not found: /shared/cc-est2024-agesex-all.parquet",
                "{",
                '  "outcome": "inconclusive",',
                '  "run_id": null,',
                '  "classification_details": {',
                '    "classification": "inconclusive",',
                '    "reasons": [',
                '      "Benchmark suite failed",',
                '      "FileNotFoundError: PEP 2020-2024 file not found: /shared/cc-est2024-agesex-all.parquet"',
                "    ]",
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    session = {
        "candidates": [
            {
                "candidate_id": "cand-1",
                "source": "recipe_catalog",
                "source_id": "RX-1",
                "execution_mode": "code_recipe",
                "status": "completed",
                "result": {
                    "outcome": "inconclusive",
                    "run_id": "not_run",
                    "stdout_log": log_rel,
                },
            }
        ]
    }

    rows = build_search_candidate_rows(session, project_root=tmp_path)

    assert rows[0]["decision_state"] == "blocked_by_data_or_runtime"
    assert rows[0]["blocker_type"] == "missing_input_data"
    assert rows[0]["blocker_detail"] == "/shared/cc-est2024-agesex-all.parquet"
    assert "Restore the missing input data" in rows[0]["recommended_action"]
    assert rows[0]["user_status_label"] == "Blocked"
    assert rows[0]["next_action_label"] == "Resolve Blocker"
    assert rows[0]["blocker_category_label"] == "Missing input data"


def test_build_search_candidate_rows_promotes_best_ready_candidate_to_recommended() -> None:
    """Completed passed-all-gates candidates should yield one recommended winner."""
    session = {
        "candidates": [
            {
                "candidate_id": "cand-b",
                "status": "completed",
                "result": {
                    "outcome": "passed_all_gates",
                    "run_id": "br-b",
                    "benchmark_summary": {
                        "primary_metric": "county_mape_overall",
                        "deltas": {"county_mape_overall": -0.10},
                    },
                },
            },
            {
                "candidate_id": "cand-a",
                "status": "completed",
                "result": {
                    "outcome": "passed_all_gates",
                    "run_id": "br-a",
                    "benchmark_summary": {
                        "primary_metric": "county_mape_overall",
                        "deltas": {"county_mape_overall": -0.30},
                    },
                },
            },
        ]
    }

    rows = build_search_candidate_rows(session)

    recommended = [row for row in rows if row["decision_state"] == "recommended"]
    assert len(recommended) == 1
    assert recommended[0]["candidate_id"] == "cand-a"


def test_build_search_session_summary_surfaces_blocked_search_state() -> None:
    """Session summary should clearly distinguish blocked searches from no-winner cases."""
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "cand-1",
                "decision_state": "blocked_by_data_or_runtime",
                "benchmark_completeness": "none",
                "outcome": "inconclusive",
                "blocker_detail": "/shared/cc-est2024-agesex-all.parquet",
                "recommended_action": "Restore the missing input data and rerun.",
            },
            {
                "candidate_id": "cand-2",
                "decision_state": "blocked_by_data_or_runtime",
                "benchmark_completeness": "none",
                "outcome": "inconclusive",
                "blocker_detail": "/shared/cc-est2024-agesex-all.parquet",
                "recommended_action": "Restore the missing input data and rerun.",
            },
        ]
    )

    summary = build_search_session_summary(
        candidates,
        search_id="search-one",
        status="finished",
        history_index_present=False,
        incomplete_bundle_count=2,
    )

    assert summary["session_decision_state"] == "blocked_by_data_or_runtime"
    assert summary["successful_benchmark_count"] == 0
    assert summary["inconclusive_count"] == 2
    assert "none of the 2 candidate(s) produced a usable benchmark" in summary["session_headline"]
    assert "Benchmark history index is missing" in summary["session_blocker_summary"]
    assert summary["user_status_label"] == "Blocked"
    assert summary["next_action_label"] == "Resolve Blocker"
    assert summary["safe_to_recommend"] is False


def test_build_search_session_summary_handles_empty_candidates() -> None:
    """Empty sessions should render a specific not-executed brief."""
    summary = build_search_session_summary(pd.DataFrame(), search_id="search-empty")

    assert summary["session_decision_state"] == "not_executed"
    assert "No search candidates" in summary["session_headline"]


def test_build_search_session_summary_surfaces_failed_hard_gate_state() -> None:
    """Hard-gate-only sessions should explain that no candidate can advance."""
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "cand-1",
                "decision_state": "failed_hard_gate",
                "benchmark_completeness": "full",
                "outcome": "failed_hard_gate",
            }
        ]
    )

    summary = build_search_session_summary(candidates, search_id="search-hard-gate")

    assert summary["session_decision_state"] == "failed_hard_gate"
    assert "failed hard gates" in summary["session_headline"]


def test_build_benchmark_decision_brief_prefers_best_passed_challenger() -> None:
    """Benchmark-backed briefs should name the strongest fully benchmarked challenger."""
    index, scorecards, experiment_log = _build_fixture_frames()
    metadata = build_run_metadata_frame(
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        champion_id="br-champion",
    )

    brief = build_benchmark_decision_brief(metadata, champion_id="br-champion")

    assert brief["source"] == "benchmark"
    assert brief["decision_state"] == "recommended"
    assert "College Blend 70" in brief["headline"]
    assert "improves county error" in brief["explanation"]
    assert brief["user_status_label"] == "Best candidate so far"
    assert brief["next_action_label"] == "Review Results"


def test_build_benchmark_decision_brief_handles_champion_only_archive() -> None:
    """Champion-only benchmark archives should stay review-ready, not recommended."""
    metadata = pd.DataFrame(
        [
            {
                "run_id": "br-champion",
                "display_name": "Champion",
                "selected_county_mape_overall": 8.86,
                "reference_county_mape_overall": 8.86,
                "status_code": "champion",
                "run_date_sort": pd.Timestamp("2026-03-09"),
            }
        ]
    )

    brief = build_benchmark_decision_brief(metadata, champion_id="br-champion")

    assert brief["decision_state"] == "ready_for_review"
    assert "only benchmark-backed option" in brief["headline"]


def test_build_benchmark_decision_brief_surfaces_failed_hard_gate_front_runner() -> None:
    """Failed hard-gate challengers should not be presented as review-ready."""
    metadata = pd.DataFrame(
        [
            {
                "run_id": "br-champion",
                "display_name": "Champion",
                "selected_county_mape_overall": 8.86,
                "reference_county_mape_overall": 8.86,
                "status_code": "champion",
                "run_date_sort": pd.Timestamp("2026-03-09"),
            },
            {
                "run_id": "br-failed",
                "display_name": "Risky Challenger",
                "selected_county_mape_overall": 8.20,
                "reference_county_mape_overall": 8.86,
                "status_code": "failed_hard_gate",
                "run_date_sort": pd.Timestamp("2026-03-10"),
            },
        ]
    )

    brief = build_benchmark_decision_brief(metadata, champion_id="br-champion")

    assert brief["decision_state"] == "failed_hard_gate"
    assert "failed a hard gate" in brief["headline"]


def test_build_candidate_decision_summary_extracts_registration_blocker_from_log(
    tmp_path,
) -> None:
    """Trailing result JSON in logs should populate blocker classification."""
    log_path = tmp_path / "logs" / "candidate.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text(
        "prefix\n{\n"
        '  "outcome": "inconclusive",\n'
        '  "classification_details": {\n'
        '    "reasons": ["code changes required", "method not in METHOD_DISPATCH"]\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    row = {
        "candidate_id": "cand-registration",
        "status": "completed",
        "outcome": "inconclusive",
        "run_id": "not_run",
        "stdout_log": "logs/candidate.log",
    }

    summary = build_candidate_decision_summary(row, project_root=tmp_path)

    assert summary["decision_state"] == "blocked_by_data_or_runtime"
    assert summary["blocker_type"] == "code_registration_required"
    assert "needs code registration" in summary["explanation"]


def test_finalize_candidate_recommendations_promotes_lowest_delta() -> None:
    """The strongest ready-for-review candidate should be promoted to recommended."""
    rows = [
        {
            "candidate_id": "cand-b",
            "decision_state": "ready_for_review",
            "decision_label": "Ready for review",
            "confidence": "high",
            "best_metric_delta": -0.10,
            "explanation": "Candidate B explanation.",
        },
        {
            "candidate_id": "cand-a",
            "decision_state": "ready_for_review",
            "decision_label": "Ready for review",
            "confidence": "high",
            "best_metric_delta": -0.25,
            "explanation": "Candidate A explanation.",
        },
    ]

    result = finalize_candidate_recommendations(rows)
    recommended = [row for row in result if row["decision_state"] == "recommended"]

    assert len(recommended) == 1
    assert recommended[0]["candidate_id"] == "cand-a"
    assert "best fully benchmarked candidate" in recommended[0]["explanation"]


def test_decision_brief_markdown_surfaces_state_headline_and_next_step() -> None:
    """Decision brief markdown should include the state, headline, and next action."""
    dm = SimpleNamespace(
        decision_brief={
            "source": "search_session",
            "session_decision_state": "blocked_by_data_or_runtime",
            "session_headline": "Search finished, but none of the candidates produced a usable benchmark.",
            "session_blocker_summary": "Required input data is missing at `/shared/cc-est2024-agesex-all.parquet`.",
            "session_recommendation": "Restore the missing input data and rerun the blocked candidates.",
            "recommendation_candidate_id": "",
        }
    )

    body = _decision_brief_markdown(cast(DashboardDataManager, dm))

    assert "**Outcome:** Blocked" in body
    assert "Required input data is missing" in body
    assert "Restore the missing input data and rerun the blocked candidates." in body


def test_decision_brief_markdown_includes_current_best_option_for_benchmark_briefs() -> None:
    """Benchmark-backed briefs should show the current best option when present."""
    dm = SimpleNamespace(
        decision_brief={
            "source": "benchmark",
            "decision_state": "recommended",
            "headline": "Best benchmark-backed candidate: College Blend 70.",
            "explanation": "It improves county error.",
            "recommended_action": "Review it in Scorecards.",
            "recommendation_candidate_id": "br-passed",
        }
    )

    body = _decision_brief_markdown(cast(DashboardDataManager, dm))

    assert "**Current focus:** br-passed" in body
    assert "search-session evidence" not in body


def test_verdict_strip_uses_user_facing_labels_instead_of_internal_codes() -> None:
    """The verdict strip should show plain-language labels, not raw decision codes."""
    dm = SimpleNamespace(
        decision_brief={
            "source": "search_session",
            "session_decision_state": "blocked_by_data_or_runtime",
            "user_status_label": "Blocked",
            "confidence_label": "Low confidence",
            "primary_subject_label": "College Blend 70",
            "main_reason": "Benchmark history index is missing.",
            "recommended_next_step": "Resolve the missing benchmark inputs.",
            "safe_to_recommend_label": "Not yet — collect more evidence first.",
            "safe_to_recommend": False,
            "escalation_guidance": "Bring to a senior analyst now",
        }
    )

    verdict_html = _verdict_strip_html(cast(DashboardDataManager, dm))

    assert "Blocked" in verdict_html
    assert "Safe to recommend?" in verdict_html
    assert "blocked_by_data_or_runtime" not in verdict_html


def test_build_decision_brief_tab_renders_cards() -> None:
    """Decision Brief tab should render the narrative cards without requiring benchmark data."""
    dm = SimpleNamespace(
        decision_brief={
            "source": "search_session",
            "session_decision_state": "blocked_by_data_or_runtime",
            "session_headline": "Search finished, but none of the candidates produced a usable benchmark.",
            "session_blocker_summary": "Required input data is missing at `/shared/cc-est2024-agesex-all.parquet`.",
            "session_recommendation": "Restore the missing input data and rerun the blocked candidates.",
            "recommendation_candidate_id": "",
        },
        benchmark_history_snapshot={
            "index_present": False,
            "bundle_count": 9,
            "complete_bundle_count": 0,
            "incomplete_bundle_count": 9,
        },
        session_review_data={"candidates": pd.DataFrame()},
        selection_state=SimpleNamespace(review_mode=False, review_step=0),
    )

    tab = build_decision_brief_tab(cast(DashboardDataManager, dm), tabs=None)
    header = cast(Any, tab[0])
    verdict = cast(Any, tab[2])
    flex_box = cast(Any, tab[3])

    assert "Decision Brief" in header.object
    assert "Safe to recommend?" in verdict.object
    assert len(flex_box.objects) >= 3


def test_build_decision_brief_tab_includes_candidate_snapshot_when_available() -> None:
    """Session-backed briefs should render a candidate table when session rows exist."""
    dm = SimpleNamespace(
        decision_brief={
            "source": "search_session",
            "session_decision_state": "ready_for_review",
            "session_headline": "Search finished with one usable benchmark.",
            "session_recommendation": "Review cand-1 next.",
            "recommendation_candidate_id": "cand-1",
        },
        benchmark_history_snapshot={
            "index_present": True,
            "bundle_count": 1,
            "complete_bundle_count": 1,
            "incomplete_bundle_count": 0,
        },
        session_review_data={
            "candidates": pd.DataFrame(
                [
                    {
                        "candidate_id": "cand-1",
                        "decision_label": "Ready for review",
                        "outcome": "passed_all_gates",
                        "best_metric_name": "county_mape_overall",
                        "best_metric_delta": -0.2,
                        "headline": "cand-1 produced a usable benchmark.",
                    }
                ]
            )
        },
        selection_state=SimpleNamespace(review_mode=False, review_step=0),
    )

    tab = build_decision_brief_tab(cast(DashboardDataManager, dm), tabs=None)
    flex_box = cast(Any, tab[3])
    assert len(flex_box.objects) == 4


def test_dashboard_data_manager_benchmark_snapshot_and_active_search_prefer_live_session(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manager should count incomplete bundles and prefer truly active search sessions."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    complete_run = history_dir / "br-complete"
    complete_run.mkdir()
    (complete_run / "summary_scorecard.csv").write_text("x\n1\n", encoding="utf-8")
    (complete_run / "manifest.json").write_text("{}", encoding="utf-8")
    incomplete_run = history_dir / "br-incomplete"
    incomplete_run.mkdir()
    (history_dir / "index.csv").write_text("run_id\nbr-complete\n", encoding="utf-8")

    session_root = tmp_path / "sessions"
    active_dir = session_root / "search-active"
    active_dir.mkdir(parents=True)
    (active_dir / "session.yaml").write_text(
        (
            "search_id: search-active\n"
            "status: running\n"
            "created_at: 2026-03-18T10:00:00+00:00\n"
            "updated_at: 2026-03-18T10:05:00+00:00\n"
            "summary:\n"
            "  total: 2\n"
            "  running: 1\n"
            "  completed: 1\n"
            "  failed: 0\n"
        ),
        encoding="utf-8",
    )
    stopped_dir = session_root / "search-stopped"
    stopped_dir.mkdir(parents=True)
    (stopped_dir / "session.yaml").write_text(
        (
            "search_id: search-stopped\n"
            "status: planned\n"
            "created_at: 2026-03-18T09:00:00+00:00\n"
            "updated_at: 2026-03-18T09:30:00+00:00\n"
            "summary:\n"
            "  total: 4\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager._is_search_process_running",
        lambda pid, search_id: search_id == "search-active",
    )

    dm = _make_dashboard_manager(monkeypatch, history_dir=history_dir, session_root=session_root)

    snapshot = dm.benchmark_history_snapshot

    assert snapshot["index_present"] is True
    assert snapshot["bundle_count"] == 2
    assert snapshot["complete_bundle_count"] == 1
    assert snapshot["incomplete_bundle_count"] == 1
    assert dm.active_search_id == "search-active"


def test_dashboard_data_manager_search_session_candidates_fall_back_to_seed_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Candidate summaries should return CSV seed rows when session YAML is missing."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    session_root = tmp_path / "sessions"
    session_dir = session_root / "search-seeded"
    session_dir.mkdir(parents=True)
    (session_dir / "candidate_summary.csv").write_text(
        "candidate_id,status,delta_county_mape_overall\ncand-1,completed,-0.3\n",
        encoding="utf-8",
    )

    dm = _make_dashboard_manager(monkeypatch, history_dir=history_dir, session_root=session_root)

    candidates = dm.search_session_candidates("search-seeded")

    assert list(candidates["candidate_id"]) == ["cand-1"]
    assert candidates.iloc[0]["status"] == "completed"


def test_dashboard_data_manager_search_session_best_candidates_sort_completed_rows(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-candidate view should sort by improvement among completed candidates."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    session_root = tmp_path / "sessions"

    dm = _make_dashboard_manager(monkeypatch, history_dir=history_dir, session_root=session_root)
    monkeypatch.setattr(
        dm,
        "search_session_candidates",
        lambda search_id: pd.DataFrame(
            [
                {
                    "candidate_id": "cand-2",
                    "status": "completed",
                    "delta_county_mape_overall": -0.10,
                    "county_mape_overall": 8.8,
                },
                {
                    "candidate_id": "cand-1",
                    "status": "completed",
                    "delta_county_mape_overall": -0.30,
                    "county_mape_overall": 8.9,
                },
                {
                    "candidate_id": "cand-3",
                    "status": "running",
                    "delta_county_mape_overall": -0.50,
                    "county_mape_overall": 8.1,
                },
            ],
        ),
    )

    best = dm.search_session_best_candidates("search-any")

    assert best["candidate_id"].tolist() == ["cand-1", "cand-2"]


def test_dashboard_data_manager_decision_brief_prefers_benchmark_when_available(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unified decision brief should prefer benchmark-backed evidence over session fallback."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    session_root = tmp_path / "sessions"
    index, scorecards, experiment_log = _build_fixture_frames()

    dm = _make_dashboard_manager(
        monkeypatch,
        history_dir=history_dir,
        session_root=session_root,
        index=index,
        scorecards=scorecards,
        experiment_log=experiment_log,
        run_ids=["br-champion", "br-review", "br-passed"],
    )

    brief = dm.decision_brief

    assert brief["source"] == "benchmark"
    assert brief["decision_state"] == "recommended"


def test_dashboard_data_manager_session_review_data_handles_no_sessions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Session review data should return a specific empty-state summary."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    session_root = tmp_path / "sessions"

    dm = _make_dashboard_manager(monkeypatch, history_dir=history_dir, session_root=session_root)

    summary = dm.session_review_data

    assert summary["session_decision_state"] == "not_executed"
    assert "No autonomous-search session" in summary["session_headline"]


def test_dashboard_data_manager_refresh_clears_cached_views_and_rebuilds_policy(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh should clear cached properties and rebuild store-backed dependencies."""
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    session_root = tmp_path / "sessions"
    alt_session_root = tmp_path / "sessions-refreshed"

    load_calls = {"count": 0}

    def fake_load_search_policy():
        load_calls["count"] += 1
        root = session_root if load_calls["count"] == 1 else alt_session_root
        return SimpleNamespace(
            project_root=tmp_path,
            session_root=root,
            default_run_budget=8,
            default_max_pending=3,
            default_max_recommended=2,
            include_recipe_catalog=True,
        )

    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.load_search_policy",
        fake_load_search_policy,
    )

    class DummyComparator:
        def __init__(self, store, config=None) -> None:
            self.store = store

        def _resolve_champion(self, scorecards: pd.DataFrame) -> str | None:
            return None

    class DummyRecommender:
        def __init__(self, store, comparator=None, config=None, bounds_catalog=None) -> None:
            pass

        def suggest_next_experiments(self, limit: int = 5):
            return []

        def identify_persistent_weaknesses(self) -> pd.DataFrame:
            return pd.DataFrame()

        def parameter_sensitivity_summary(self) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.ObservatoryComparator",
        DummyComparator,
    )
    monkeypatch.setattr(
        "cohort_projections.analysis.observatory.dashboard.data_manager.ObservatoryRecommender",
        DummyRecommender,
    )
    monkeypatch.setattr(DashboardDataManager, "_try_build_catalog", lambda self: None)

    store = _FakeResultsStore(history_dir=history_dir)
    dm = DashboardDataManager(store=cast(Any, store), config=store._config)

    _ = dm.search_policy
    _ = dm.search_sessions
    _ = dm.decision_brief

    dm.refresh()

    assert store.refreshed is True
    assert dm.search_policy.session_root == alt_session_root
    assert "search_sessions" not in dm.__dict__
    assert "decision_brief" not in dm.__dict__
    assert load_calls["count"] >= 2


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
    assert "What still needs judgment" in summary


def test_scorecard_delta_chart_returns_placeholder_when_no_selection_exists() -> None:
    """Empty scorecard comparisons should explain the gap instead of rendering a blank chart shell."""
    dm = SimpleNamespace(comparison_rows=pd.DataFrame(), champion_id=None)

    pane = _build_delta_bar_chart([], cast(DashboardDataManager, dm))

    assert isinstance(pane, pn.pane.HTML)
    assert "Select at least one challenger bundle" in pane.object


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


def test_projection_focus_summary_surfaces_selected_paths_and_endpoint_note() -> None:
    """Projection focus summary should replace the raw table as the first path-orientation cue."""
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

    pane = _build_projection_focus_summary(
        "2020",
        ["br-passed"],
        True,
        cast(DashboardDataManager, dm),
    )

    assert "Selected paths" in pane.object
    assert "Champion" in pane.object
    assert "College Blend 70" in pane.object
    assert "20,000 people higher than" in pane.object


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

    assert "Average county error (MAPE) rises to 12.3% by horizon 20." in summary
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

    assert "**What to try next:** college_blend_factor -> 1.0." in summary
    assert (
        "**Why the system is suggesting it:** Largest tested improvement in college counties."
        in summary
    )
    assert "1 tracked metric(s) still have no improving challenger." in summary
    assert "Most influential tested parameter so far: `college_blend_factor`." in summary


def test_recommendation_cards_render_low_risk_and_exploratory_states() -> None:
    """Portrait recommendation cards should explain both suggestion type and rationale."""
    recommender = MagicMock()
    recommender.suggest_next_experiments.return_value = [
        SimpleNamespace(
            priority=1,
            parameter="college_blend_factor",
            suggested_value=1.0,
            expected_impact="Improve college-county fit",
            direction="Could raise rural error slightly",
            requires_code_change=False,
            rationale="Largest tested improvement in urban-college counties.",
        ),
        SimpleNamespace(
            priority=2,
            parameter="migration_recipe",
            suggested_value="college_fix_v2",
            expected_impact="May reduce endpoint overshoot",
            direction="Requires method registration work",
            requires_code_change=True,
            rationale="Current search results still over-project at the long horizon.",
        ),
    ]
    dm = SimpleNamespace(recommender=recommender)

    cards = _build_recommendation_cards(cast(DashboardDataManager, dm))
    html = "\n".join(str(getattr(obj, "object", "")) for obj in cards.objects)

    assert "Low-risk follow-up" in html
    assert "Exploratory" in html
    assert "Why the system is suggesting it" in html


# ---------------------------------------------------------------------------
# New widget factory tests (Phase 1)
# ---------------------------------------------------------------------------

from cohort_projections.analysis.observatory.dashboard.widgets import (
    candidate_feed,
    completion_banner,
    hero_metric,
    illustrated_empty_state,
    info_tooltip,
    progress_ring,
    section_header,
    terminal_output,
    workflow_stepper,
)


class TestWorkflowStepper:
    """Tests for the horizontal workflow stepper widget."""

    def test_renders_all_steps(self) -> None:
        pane = workflow_stepper(["Launch", "Monitor", "Review", "Decide"], active=0)
        html = pane.object
        assert "Launch" in html
        assert "Monitor" in html
        assert "Review" in html
        assert "Decide" in html
        assert "obs-workflow-stepper" in html

    def test_marks_active_step(self) -> None:
        pane = workflow_stepper(["A", "B", "C"], active=1)
        html = pane.object
        # Step B (index 1) should have the active class
        assert 'class="obs-step active"' in html

    def test_marks_completed_steps(self) -> None:
        pane = workflow_stepper(["A", "B", "C"], active=2, completed=[0, 1])
        html = pane.object
        assert html.count("obs-step completed") == 2
        # Completed steps show checkmark
        assert "&#10003;" in html

    def test_connectors_between_steps(self) -> None:
        pane = workflow_stepper(["A", "B", "C"], active=0)
        html = pane.object
        # Should have 2 connectors for 3 steps
        assert html.count("obs-step-connector") == 2


class TestProgressRing:
    """Tests for the circular progress ring widget."""

    def test_renders_percentage(self) -> None:
        pane = progress_ring(75.0, "15/20", status="running")
        html = pane.object
        assert "75%" in html
        assert "15/20" in html
        assert "obs-progress-ring" in html

    def test_running_status_has_animation(self) -> None:
        pane = progress_ring(50.0, "", status="running")
        assert " running" in pane.object

    def test_complete_status_no_animation(self) -> None:
        pane = progress_ring(100.0, "", status="complete")
        assert " running" not in pane.object

    def test_clamps_percentage(self) -> None:
        pane = progress_ring(150.0, "")
        assert "--progress:100" in pane.object
        pane2 = progress_ring(-10.0, "")
        assert "--progress:0" in pane2.object


class TestCandidateFeed:
    """Tests for the live candidate results feed."""

    def test_empty_shows_waiting(self) -> None:
        pane = candidate_feed(pd.DataFrame())
        assert "Waiting for results" in pane.object

    def test_renders_candidates(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "candidate_id": "cand-1",
                    "outcome": "passed_all_gates",
                    "county_mape_overall": 8.50,
                    "delta_county_mape_overall": -0.36,
                },
                {
                    "candidate_id": "cand-2",
                    "outcome": "failed_hard_gate",
                    "county_mape_overall": 9.10,
                    "delta_county_mape_overall": 0.24,
                },
            ]
        )
        pane = candidate_feed(df)
        html = pane.object
        assert "cand-1" in html
        assert "cand-2" in html
        assert "obs-candidate-feed-item" in html
        assert "improved" in html  # cand-1 has negative delta
        assert "regressed" in html  # cand-2 has positive delta

    def test_respects_max_items(self) -> None:
        df = pd.DataFrame([{"candidate_id": f"c-{i}", "outcome": "untested"} for i in range(10)])
        pane = candidate_feed(df, max_items=3)
        assert pane.object.count("obs-candidate-feed-item") == 3


class TestHeroMetric:
    """Tests for the large focal metric display."""

    def test_renders_value_and_label(self) -> None:
        pane = hero_metric("8.86%", "Champion County Error")
        html = pane.object
        assert "8.86%" in html
        assert "Champion County Error" in html
        assert "obs-metric-hero" in html

    def test_delta_shown_when_provided(self) -> None:
        pane = hero_metric("8.86%", "Error", delta=-0.36)
        html = pane.object
        assert "&#9660;" in html  # down arrow for improvement
        assert "-0.36" in html

    def test_no_delta_when_none(self) -> None:
        pane = hero_metric("8.86%", "Error")
        assert "obs-mh-delta" not in pane.object


class TestCompletionBanner:
    """Tests for the post-search completion banner."""

    def test_success_variant(self) -> None:
        pane = completion_banner(20, "cand-best", -0.36, status="success")
        html = pane.object
        assert "obs-completion-banner success" in html
        assert "20 experiment(s) finished" in html
        assert "cand-best" in html
        assert "0.36pp" in html
        assert "Review Results" in html
        assert "Scorecards" in html
        assert "Experiment History" in html
        assert "Experiments" not in html

    def test_mixed_variant(self) -> None:
        pane = completion_banner(15, status="mixed")
        html = pane.object
        assert "obs-completion-banner mixed" in html
        assert "errors" in html.lower()
        assert "Decision Brief" in html
        assert "Experiment History" in html

    def test_failed_variant(self) -> None:
        pane = completion_banner(5, status="failed")
        html = pane.object
        assert "obs-completion-banner failed" in html
        assert "failed" in html.lower()


def test_enter_guided_review_sets_review_mode_and_navigates() -> None:
    """Guided review entry should enable review mode and open Decision Brief."""
    seeded = {"called": False}

    def _seed() -> None:
        seeded["called"] = True

    dm = SimpleNamespace(
        selection_state=SimpleNamespace(review_mode=False, review_step=0),
        initialize_guided_review_shortlist=_seed,
    )
    tabs = SimpleNamespace(active=0)

    _enter_guided_review(cast(DashboardDataManager, dm), cast(pn.Tabs, tabs))

    assert seeded["called"] is True
    assert dm.selection_state.review_mode is True
    assert dm.selection_state.review_step == 1
    assert tabs.active == 1


def test_resolve_stepper_state_uses_monitor_step_for_running_search() -> None:
    """The workflow stepper should show the monitor step during a live search."""
    dm = SimpleNamespace(
        selection_state=SimpleNamespace(review_mode=False),
        run_ids=[],
        search_sessions=pd.DataFrame([{"search_id": "search-1"}]),
        active_search_id="search-1",
        search_session_row=lambda search_id: pd.Series({"dashboard_process_running": True}),
    )

    active, completed = _resolve_stepper_state(cast(DashboardDataManager, dm), active_tab=0)

    assert active == 1
    assert completed == [0]


class TestInfoTooltip:
    """Tests for the hover tooltip widget."""

    def test_renders_tooltip_text(self) -> None:
        pane = info_tooltip("Help text here")
        html = pane.object
        assert "obs-tooltip" in html
        assert "Help text here" in html

    def test_escapes_html_in_text(self) -> None:
        pane = info_tooltip('Text with <script> & "quotes"')
        html = pane.object
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "&quot;" in html


class TestIllustratedEmptyState:
    """Tests for the SVG-illustrated empty state."""

    def test_search_illustration(self) -> None:
        pane = illustrated_empty_state("No data found", "search")
        html = pane.object
        assert "obs-empty-state" in html
        assert "<svg" in html
        assert "No data found" in html

    def test_rocket_illustration(self) -> None:
        pane = illustrated_empty_state("Run your first search", "rocket")
        assert "<svg" in pane.object

    def test_check_illustration(self) -> None:
        pane = illustrated_empty_state("All done", "check")
        assert "<svg" in pane.object

    def test_fallback_illustration(self) -> None:
        pane = illustrated_empty_state("Unknown", "nonexistent")
        # Falls back to search SVG
        assert "<svg" in pane.object


class TestTerminalOutput:
    """Tests for the dark-themed terminal log display."""

    def test_renders_text(self) -> None:
        pane = terminal_output("Hello log output")
        html = pane.object
        assert "obs-terminal" in html
        assert "Hello log output" in html

    def test_escapes_html(self) -> None:
        pane = terminal_output("<script>alert('xss')</script>")
        html = pane.object
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_max_height(self) -> None:
        pane = terminal_output("text", max_height=300)
        assert "max-height:300px" in pane.object


class TestSectionHeaderTooltip:
    """Tests for the tooltip extension on section_header."""

    def test_no_tooltip_by_default(self) -> None:
        pane = section_header("Title")
        assert "obs-tooltip" not in pane.object

    def test_tooltip_when_provided(self) -> None:
        pane = section_header("Title", tooltip="Helpful info")
        html = pane.object
        assert "obs-tooltip" in html
        assert "Helpful info" in html
