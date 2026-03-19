from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from cohort_projections.analysis.benchmark_contract import (
    BENCHMARK_INDEX_COLUMNS,
    MANIFEST_FIELDS,
    SUMMARY_SCORECARD_COLUMNS,
)
from cohort_projections.analysis.benchmarking import load_aliases

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "promote_method_script",
    PROJECT_ROOT / "scripts" / "analysis" / "promote_method.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_profile(profile_dir: Path, method_id: str, config_id: str) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / f"{method_id}__{config_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "method_id": method_id,
                "config_id": config_id,
                "status": "candidate",
                "resolved_config": {"alpha": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_run(
    history_dir: Path, run_id: str, run_date: str, method_id: str, config_id: str
) -> None:
    run_dir = history_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    scorecard_row = dict.fromkeys(SUMMARY_SCORECARD_COLUMNS, 0)
    scorecard_row.update(
        {
            "run_id": run_id,
            "method_id": method_id,
            "config_id": config_id,
            "scope": "county",
            "status_at_run": "candidate",
            "artifact_completeness_flag": True,
            "reproducibility_logging_flag": True,
            "runtime_summary_present": True,
            "runtime_total_seconds": 120.0,
            "slowest_stage_seconds": 60.0,
            "slowest_stage_share": 0.5,
        }
    )
    pd.DataFrame([scorecard_row]).to_csv(run_dir / "summary_scorecard.csv", index=False)
    (run_dir / "summary_scorecard.json").write_text("[]", encoding="utf-8")

    manifest = {field: [] for field in MANIFEST_FIELDS}
    manifest.update(
        {
            "run_id": run_id,
            "run_date": run_date,
            "benchmark_label": "promotion_test",
            "benchmark_contract_version": "1.0",
            "created_at_utc": "2026-03-19T12:00:00+00:00",
            "git_commit": "abcdef0123456789",
            "git_dirty": False,
            "command": ["python"],
            "scope": "county",
            "methods": [{"method_id": method_id}],
            "champion_method_id": method_id,
            "challenger_method_ids": [],
            "input_artifacts": [],
            "output_artifacts": [],
            "script_versions": {},
            "execution_log_run_id": "exec-001",
            "duration_seconds": 120.0,
            "runtime_summary": {
                "total_duration_seconds": 120.0,
                "slowest_stage": "annual_validation",
                "slowest_stage_seconds": 60.0,
                "slowest_stage_share": 0.5,
            },
            "operational_quality": {
                "artifact_completeness_flag": True,
                "reproducibility_logging_flag": True,
                "runtime_summary_present": True,
                "baseline_only": True,
            },
        }
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "runtime_summary.json").write_text(
        json.dumps(manifest["runtime_summary"], indent=2),
        encoding="utf-8",
    )

    row = {
        "run_id": run_id,
        "run_date": run_date,
        "method_id": method_id,
        "config_id": config_id,
        "scope": "county",
        "benchmark_label": "promotion_test",
        "benchmark_contract_version": "1.0",
        "git_commit": "abcdef0",
        "decision_id": "",
        "decision_status": "pending",
        "is_champion_at_run": "false",
        "summary_scorecard_path": str(run_dir / "summary_scorecard.csv"),
        "manifest_path": str(run_dir / "manifest.json"),
    }
    index_path = history_dir / "index.csv"
    if index_path.exists():
        existing = pd.read_csv(index_path)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row], columns=BENCHMARK_INDEX_COLUMNS)
    updated.to_csv(index_path, index=False)


def test_promote_method_revalidates_then_refreshes_latest_pointer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile_dir = tmp_path / "profiles"
    alias_path = tmp_path / "aliases.yaml"
    decision_dir = tmp_path / "benchmark_decisions"
    decision_dir.mkdir()
    history_dir = tmp_path / "benchmark_history"
    history_dir.mkdir()
    promotion_history = history_dir / "promotion_history.csv"

    _write_profile(profile_dir, "m2026r1", "cfg-new")
    alias_path.write_text(
        yaml.safe_dump(
            {
                "county_champion": {
                    "method_id": "m2026",
                    "config_id": "cfg-old",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (decision_dir / "2026-03-12-m2026r1-vs-m2026.md").write_text(
        "| Status | Approved |\n",
        encoding="utf-8",
    )

    _write_run(
        history_dir,
        run_id="br-20260312-090000-m2026r1-aaaaaaa",
        run_date="20260312",
        method_id="m2026r1",
        config_id="cfg-new",
    )

    def _fake_benchmark_run(cmd: list[str], cwd: Path, check: bool) -> None:
        assert "--baseline-only" in cmd
        _write_run(
            history_dir,
            run_id="br-20260312-110000-m2026r1-bbbbbbb",
            run_date="20260312",
            method_id="m2026r1",
            config_id="cfg-new",
        )

    monkeypatch.setattr(_MODULE.subprocess, "run", _fake_benchmark_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_method.py",
            "--scope",
            "county",
            "--alias",
            "county_champion",
            "--method-id",
            "m2026r1",
            "--config-id",
            "cfg-new",
            "--decision-id",
            "2026-03-12-m2026r1-vs-m2026",
            "--profile-dir",
            str(profile_dir),
            "--alias-path",
            str(alias_path),
            "--decision-dir",
            str(decision_dir),
            "--promotion-history",
            str(promotion_history),
            "--history-dir",
            str(history_dir),
            "--revalidate",
        ],
    )

    _MODULE.main()

    aliases = load_aliases(alias_path)
    pointer_payload = json.loads(
        (history_dir / "latest" / "county_champion" / "pointer.json").read_text(encoding="utf-8")
    )
    promotion_df = pd.read_csv(promotion_history)

    assert aliases["county_champion"] == {"method_id": "m2026r1", "config_id": "cfg-new"}
    assert promotion_df.iloc[0]["new_method_id"] == "m2026r1"
    assert pointer_payload["run_id"] == "br-20260312-110000-m2026r1-bbbbbbb"
