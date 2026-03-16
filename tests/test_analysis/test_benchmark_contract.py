from __future__ import annotations

import pandas as pd
import pytest

from cohort_projections.analysis.benchmark_contract import (
    validate_index_columns,
    validate_manifest,
    validate_scorecard_columns,
)


def test_validate_manifest_accepts_required_fields() -> None:
    manifest = {
        "run_id": "run-1",
        "run_date": "2026-03-15",
        "benchmark_label": "test",
        "benchmark_contract_version": "1.0",
        "created_at_utc": "2026-03-15T00:00:00Z",
        "git_commit": "abcdef0",
        "git_dirty": False,
        "command": "python test.py",
        "scope": "county",
        "methods": [],
        "champion_method_id": "m2026",
        "challenger_method_ids": ["m2026r1"],
        "input_artifacts": [],
        "output_artifacts": [],
        "script_versions": {},
        "execution_log_run_id": "exec-1",
        "duration_seconds": 1.2,
        "runtime_summary": {},
    }
    validate_manifest(manifest)


def test_validate_scorecard_columns_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="summary_scorecard.csv missing required columns"):
        validate_scorecard_columns(pd.DataFrame([{"run_id": "run-1"}]))


def test_validate_index_columns_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="benchmark_history/index.csv missing required columns"):
        validate_index_columns(pd.DataFrame([{"run_id": "run-1"}]))
