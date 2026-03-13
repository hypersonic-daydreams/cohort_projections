from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "analysis"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_benchmark_suite as rbs


def test_build_runtime_summary_tracks_stage_shares_and_worker_config() -> None:
    summary = rbs._build_runtime_summary(
        stage_timings={
            "load_shared_inputs": 1.5,
            "annual_validation": 6.0,
            "sensitivity_analysis": 1.5,
        },
        total_duration_seconds=10.0,
        workers_arg=4,
    )

    assert summary["stage_timings_seconds"]["annual_validation"] == 6.0
    assert summary["stage_timings_seconds"]["other_overhead"] == 1.0
    assert summary["stage_shares"]["annual_validation"] == 0.6
    assert summary["slowest_stage"] == "annual_validation"
    assert summary["slowest_stage_seconds"] == 6.0
    assert summary["worker_config"]["shared_workers_arg"] == 4
    assert (
        summary["worker_config"]["annual_validation_county_workers_requested"]
        == 4
    )


def test_build_runtime_summary_handles_empty_timings() -> None:
    summary = rbs._build_runtime_summary(
        stage_timings={},
        total_duration_seconds=0.0,
        workers_arg=1,
    )

    assert summary["stage_timings_seconds"] == {}
    assert summary["stage_shares"] == {}
    assert summary["slowest_stage"] is None
    assert summary["slowest_stage_seconds"] == 0.0
