"""Canonical candidate identity helpers for the Projection Observatory.

Policy
------
Candidate identity is resolved from ``config_id`` when present, because that
represents the reusable parameterized method profile across repeated benchmark
bundles. If a row has no ``config_id``, the Observatory falls back to
``run_id`` so older or partial artifacts still participate in comparison.
Rankings and Pareto analysis should operate on this candidate view rather than
raw scorecard rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cohort_projections.analysis.observatory.status import aggregate_statuses

_RESERVED_COLUMNS = {
    "candidate_id",
    "candidate_source",
    "candidate_status",
    "run_count",
    "run_ids",
}


def _first_non_empty(series: pd.Series) -> Any:
    """Return the first non-empty scalar from a series, or ``None``."""
    for value in series:
        if value is None or pd.isna(value):
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def add_candidate_identity_columns(scorecards: pd.DataFrame) -> pd.DataFrame:
    """Add ``candidate_id`` and ``candidate_source`` columns to scorecards."""
    if scorecards.empty:
        return scorecards.copy()

    work = scorecards.copy()
    config_series = (
        work["config_id"].fillna("").astype(str).str.strip()
        if "config_id" in work.columns
        else pd.Series("", index=work.index, dtype=object)
    )
    run_series = (
        work["run_id"].fillna("").astype(str).str.strip()
        if "run_id" in work.columns
        else pd.Series("", index=work.index, dtype=object)
    )
    method_series = (
        work["method_id"].fillna("").astype(str).str.strip()
        if "method_id" in work.columns
        else pd.Series("", index=work.index, dtype=object)
    )

    candidate_ids = config_series.mask(config_series.eq(""), run_series)
    candidate_ids = candidate_ids.mask(candidate_ids.eq(""), method_series)
    candidate_ids = candidate_ids.mask(candidate_ids.eq(""), "unknown-candidate")
    work["candidate_id"] = candidate_ids
    work["candidate_source"] = config_series.ne("").map(
        lambda has_config: "config_id" if has_config else "run_id"
    )
    return work


def build_candidate_view(scorecards: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw scorecard rows to one row per canonical candidate."""
    if scorecards.empty:
        return pd.DataFrame()

    work = add_candidate_identity_columns(scorecards)
    rows: list[dict[str, Any]] = []

    for (candidate_id, candidate_source), group in work.groupby(
        ["candidate_id", "candidate_source"],
        sort=False,
        dropna=False,
    ):
        run_ids = (
            sorted(group["run_id"].dropna().astype(str).unique().tolist())
            if "run_id" in group.columns
            else []
        )
        row: dict[str, Any] = {
            "candidate_id": str(candidate_id),
            "candidate_source": str(candidate_source),
            "candidate_status": aggregate_statuses(
                group["status_at_run"].tolist()
                if "status_at_run" in group.columns
                else []
            ),
            "run_count": len(run_ids) or len(group),
            "run_ids": run_ids,
        }
        if run_ids:
            row["run_id"] = run_ids[0]

        for column in group.columns:
            if column in _RESERVED_COLUMNS:
                continue
            if column == "run_id" and "run_id" in row:
                continue
            series = group[column]
            if pd.api.types.is_numeric_dtype(series):
                row[column] = float(series.dropna().mean()) if series.notna().any() else None
                continue
            if column == "status_at_run":
                row[column] = row["candidate_status"]
                continue
            row[column] = _first_non_empty(series)

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if "config_id" in result.columns:
        result["config_id"] = result["config_id"].fillna("")
    if "candidate_id" in result.columns:
        result = result.sort_values("candidate_id").reset_index(drop=True)
    return result
