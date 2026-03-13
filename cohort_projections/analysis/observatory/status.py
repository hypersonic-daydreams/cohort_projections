"""Canonical Observatory status reconciliation utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

STATUS_LABELS: dict[str, str] = {
    "champion": "Champion",
    "passed_all_gates": "Passed",
    "needs_human_review": "Review",
    "failed_hard_gate": "Failed",
    "inconclusive": "Inconclusive",
    "pending": "Pending",
    "not_run": "Not Run",
    "unresolved": "Unresolved",
    "untested": "Untested",
}

# Lower number = more trusted / more resolved.
STATUS_PRIORITY: dict[str, int] = {
    "champion": 0,
    "passed_all_gates": 1,
    "needs_human_review": 2,
    "failed_hard_gate": 3,
    "inconclusive": 4,
    "pending": 5,
    "not_run": 6,
    "unresolved": 7,
    "untested": 8,
}

GATE_CLEAN_STATUSES: frozenset[str] = frozenset({"champion", "passed_all_gates"})
_KNOWN_OUTCOME_STATUSES: frozenset[str] = frozenset(
    {
        "champion",
        "passed_all_gates",
        "needs_human_review",
        "failed_hard_gate",
        "inconclusive",
        "pending",
        "not_run",
        "unresolved",
        "untested",
    }
)


def normalize_status(value: object) -> str:
    """Return a lower-case status code or ``"untested"``."""
    if value is None or pd.isna(value):
        return "untested"
    status = str(value).strip().lower()
    return status or "untested"


def map_scorecard_status(value: object) -> str:
    """Map scorecard-local labels onto the canonical Observatory status model."""
    status = normalize_status(value)
    if status in {"candidate", "experiment"}:
        return "unresolved"
    if status in _KNOWN_OUTCOME_STATUSES:
        return status
    return status


def resolve_observatory_status(
    *,
    experiment_outcome: object = None,
    catalog_status: object = None,
    scorecard_status: object = None,
    is_champion: bool = False,
) -> str:
    """Resolve one canonical status across experiment log, catalog, and scorecard."""
    if is_champion:
        return "champion"

    for status in (experiment_outcome, catalog_status):
        normalized = normalize_status(status)
        if normalized in _KNOWN_OUTCOME_STATUSES and normalized != "untested":
            return normalized

    mapped_scorecard = map_scorecard_status(scorecard_status)
    if mapped_scorecard in _KNOWN_OUTCOME_STATUSES:
        return mapped_scorecard
    return "untested"


def aggregate_statuses(statuses: Iterable[object]) -> str:
    """Collapse many status codes to one conservative candidate-level status."""
    normalized = [normalize_status(status) for status in statuses]
    if not normalized:
        return "untested"
    if "champion" in normalized:
        return "champion"
    resolved = [
        status
        for status in normalized
        if status in _KNOWN_OUTCOME_STATUSES and status != "untested"
    ]
    if not resolved:
        return "untested"
    return min(resolved, key=lambda status: STATUS_PRIORITY.get(status, 99))


def is_gate_clean(status: object) -> bool:
    """Return ``True`` when a status is safe to trust in automated search."""
    return normalize_status(status) in GATE_CLEAN_STATUSES


def status_label(status: object) -> str:
    """Return a short human-readable label for a status code."""
    normalized = normalize_status(status)
    return STATUS_LABELS.get(normalized, normalized.replace("_", " ").title())
