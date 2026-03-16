"""Benchmark contract constants and validation helpers.

Centralizes the required manifest, scorecard, and index schemas used by the
benchmarking and Observatory tooling so schema drift is caught explicitly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BENCHMARK_INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_date",
    "method_id",
    "config_id",
    "scope",
    "benchmark_label",
    "benchmark_contract_version",
    "git_commit",
    "decision_id",
    "decision_status",
    "is_champion_at_run",
    "summary_scorecard_path",
    "manifest_path",
)

SUMMARY_SCORECARD_COLUMNS: tuple[str, ...] = (
    "run_id",
    "method_id",
    "config_id",
    "scope",
    "status_at_run",
    "state_ape_recent_short",
    "state_ape_recent_medium",
    "state_signed_bias_recent",
    "county_mape_overall",
    "county_mape_urban_college",
    "county_mape_rural",
    "county_mape_bakken",
    "county_mape_reservation",
    "county_mape_smallest",
    "county_mape_volatile_oil",
    "county_mape_college_heavy_non_core",
    "negative_population_violations",
    "scenario_order_violations",
    "aggregation_violations",
    "sensitivity_instability_flag",
    "artifact_completeness_flag",
    "reproducibility_logging_flag",
    "runtime_total_seconds",
    "slowest_stage_seconds",
)

MANIFEST_FIELDS: tuple[str, ...] = (
    "run_id",
    "run_date",
    "benchmark_label",
    "benchmark_contract_version",
    "created_at_utc",
    "git_commit",
    "git_dirty",
    "command",
    "scope",
    "methods",
    "champion_method_id",
    "challenger_method_ids",
    "input_artifacts",
    "output_artifacts",
    "script_versions",
    "execution_log_run_id",
    "duration_seconds",
    "runtime_summary",
)


def _missing_fields(actual_fields: set[str], required_fields: tuple[str, ...]) -> list[str]:
    """Return required fields missing from the provided field set."""
    return sorted(field for field in required_fields if field not in actual_fields)


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate the required top-level manifest fields."""
    missing = _missing_fields(set(manifest), MANIFEST_FIELDS)
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")


def validate_scorecard_columns(scorecard: pd.DataFrame) -> None:
    """Validate the required scorecard columns."""
    missing = _missing_fields(set(scorecard.columns), SUMMARY_SCORECARD_COLUMNS)
    if missing:
        raise ValueError(f"summary_scorecard.csv missing required columns: {missing}")


def validate_index_columns(index: pd.DataFrame) -> None:
    """Validate the required benchmark index columns."""
    missing = _missing_fields(set(index.columns), BENCHMARK_INDEX_COLUMNS)
    if missing:
        raise ValueError(f"benchmark_history/index.csv missing required columns: {missing}")
