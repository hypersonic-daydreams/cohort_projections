"""Tests for shared Observatory status reconciliation helpers."""

from __future__ import annotations

from cohort_projections.analysis.observatory.status import (
    aggregate_statuses,
    is_gate_clean,
    map_scorecard_status,
    resolve_observatory_status,
)


def test_scorecard_candidate_maps_to_unresolved() -> None:
    assert map_scorecard_status("candidate") == "unresolved"
    assert map_scorecard_status("experiment") == "unresolved"


def test_log_outcome_overrides_catalog_status() -> None:
    assert (
        resolve_observatory_status(
            experiment_outcome="passed_all_gates",
            catalog_status="needs_human_review",
            scorecard_status="experiment",
        )
        == "passed_all_gates"
    )


def test_champion_takes_precedence() -> None:
    assert (
        resolve_observatory_status(
            experiment_outcome="failed_hard_gate",
            scorecard_status="champion",
            is_champion=True,
        )
        == "champion"
    )


def test_aggregate_statuses_prefers_champion_then_best_resolved_status() -> None:
    assert aggregate_statuses(["untested", "needs_human_review"]) == "needs_human_review"
    assert aggregate_statuses(["passed_all_gates", "champion"]) == "champion"


def test_gate_clean_statuses() -> None:
    assert is_gate_clean("champion") is True
    assert is_gate_clean("passed_all_gates") is True
    assert is_gate_clean("needs_human_review") is False
