"""Tests for Projection Observatory fresh-start workspace reset."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import yaml

from cohort_projections.analysis.benchmark_contract import BENCHMARK_INDEX_COLUMNS
from cohort_projections.analysis.experiment_log import LOG_COLUMNS
from cohort_projections.analysis.observatory.workspace_reset import (
    reset_observatory_workspace,
)


def _write_header_csv(path: Path, fieldnames: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()


def _append_csv_row(
    path: Path, fieldnames: tuple[str, ...] | list[str], row: dict[str, str]
) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writerow(row)


def _write_configs(project_root: Path) -> tuple[Path, Path]:
    config_path = project_root / "config" / "observatory_config.yaml"
    search_policy_path = project_root / "config" / "observatory_search_policy.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "observatory": {
                    "history_dir": "data/analysis/benchmark_history",
                    "cache_dir": "data/analysis/observatory",
                    "experiment_log": "data/analysis/experiments/experiment_log.csv",
                    "variant_catalog": "config/observatory_variants.yaml",
                    "champion_method": "m2026",
                    "challenger_base_method": "m2026r1",
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    search_policy_path.write_text(
        yaml.safe_dump(
            {
                "search": {
                    "runtime_root": "data/analysis/observatory_runtime",
                    "session_root": "data/analysis/experiments/search_runs",
                    "mirror_repo": "data/analysis/observatory_runtime/repos/cohort_projections.git",
                    "worktree_root": "data/analysis/observatory_runtime/worktrees",
                    "recipe_catalog": "config/observatory_recipes.yaml",
                    "protected_paths": ["DEVELOPMENT_TRACKER.md"],
                    "allowed_recipe_roots": ["scripts/analysis", "cohort_projections"],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, search_policy_path


def _seed_workspace(project_root: Path) -> dict[str, Path]:
    history_dir = project_root / "data" / "analysis" / "benchmark_history"
    cache_dir = project_root / "data" / "analysis" / "observatory"
    experiments_dir = project_root / "data" / "analysis" / "experiments"
    search_runs_dir = experiments_dir / "search_runs"
    sweeps_dir = experiments_dir / "sweeps"
    runtime_root = project_root / "data" / "analysis" / "observatory_runtime"
    repos_dir = runtime_root / "repos"
    worktrees_dir = runtime_root / "worktrees"

    run_dir = history_dir / "br-20260319-120000-m2026r1-abcdef0"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "summary_scorecard.csv").write_text("run_id\n", encoding="utf-8")
    (history_dir / "latest").mkdir(parents=True)

    index_path = history_dir / "index.csv"
    _write_header_csv(index_path, BENCHMARK_INDEX_COLUMNS)
    _append_csv_row(
        index_path,
        BENCHMARK_INDEX_COLUMNS,
        {
            "run_id": run_dir.name,
            "run_date": "20260319",
            "method_id": "m2026r1",
            "config_id": "cfg-candidate",
            "scope": "county",
            "benchmark_label": "candidate",
            "benchmark_contract_version": "1.0",
            "git_commit": "abcdef0",
            "decision_id": "",
            "decision_status": "pending",
            "is_champion_at_run": "false",
            "summary_scorecard_path": str(run_dir / "summary_scorecard.csv"),
            "manifest_path": str(run_dir / "manifest.json"),
        },
    )
    promotion_history = history_dir / "promotion_history.csv"
    promotion_history.write_text(
        (
            "promoted_at_utc,alias_name,prior_method_id,prior_config_id,"
            "new_method_id,new_config_id,decision_id\n"
            "2026-03-19T00:00:00+00:00,county_champion,m2026,cfg-old,m2026r1,cfg-new,decision-1\n"
        ),
        encoding="utf-8",
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "consolidated_scorecards.parquet").write_text("cache", encoding="utf-8")

    experiments_dir.mkdir(parents=True, exist_ok=True)
    experiment_log = experiments_dir / "experiment_log.csv"
    _write_header_csv(experiment_log, LOG_COLUMNS)
    _append_csv_row(
        experiment_log,
        LOG_COLUMNS,
        {
            "experiment_id": "EXP-A",
            "run_date": "2026-03-19",
            "hypothesis": "candidate",
            "base_method": "m2026r1",
            "config_delta_summary": "foo=1",
            "run_id": run_dir.name,
            "outcome": "inconclusive",
            "key_metrics_summary": "",
            "interpretation": "",
            "next_action": "flag_for_review",
            "agent_or_human": "agent",
            "spec_path": "spec.yaml",
        },
    )
    search_runs_dir.mkdir(parents=True, exist_ok=True)
    (search_runs_dir / ".search-20260319.dashboard.log").write_text("log", encoding="utf-8")
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    (sweeps_dir / "resume.json").write_text("{}", encoding="utf-8")

    repos_dir.mkdir(parents=True, exist_ok=True)
    (repos_dir / "cohort_projections.git").mkdir()
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    (worktrees_dir / "candidate-a").mkdir()

    return {
        "history_dir": history_dir,
        "run_dir": run_dir,
        "index_path": index_path,
        "promotion_history": promotion_history,
        "cache_dir": cache_dir,
        "experiment_log": experiment_log,
        "search_runs_dir": search_runs_dir,
        "sweeps_dir": sweeps_dir,
        "runtime_root": runtime_root,
        "repos_dir": repos_dir,
        "worktrees_dir": worktrees_dir,
    }


def test_reset_observatory_workspace_archives_and_reinitializes(tmp_path: Path) -> None:
    config_path, search_policy_path = _write_configs(tmp_path)
    paths = _seed_workspace(tmp_path)

    result = reset_observatory_workspace(
        project_root=tmp_path,
        config_path=config_path,
        search_policy_path=search_policy_path,
        archive_root=tmp_path / "data" / "analysis" / "observatory_archives",
        archive_label="reset-test",
    )

    archive_dir = Path(result["archive_dir"])
    assert archive_dir.exists()
    assert (archive_dir / "reset_manifest.json").exists()
    assert (archive_dir / "benchmark_history" / paths["run_dir"].name).exists()
    assert (archive_dir / "benchmark_history" / "latest").exists()
    assert (archive_dir / "experiments" / "experiment_log.csv").exists()
    assert (archive_dir / "experiments" / "search_runs").exists()
    assert (archive_dir / "observatory_cache" / "consolidated_scorecards.parquet").exists()
    assert (archive_dir / "observatory_runtime" / "repos").exists()

    assert not paths["run_dir"].exists()
    index_df = pd.read_csv(paths["index_path"])
    assert index_df.empty
    assert list(index_df.columns) == list(BENCHMARK_INDEX_COLUMNS)

    assert paths["promotion_history"].exists()
    promotion_df = pd.read_csv(paths["promotion_history"])
    assert len(promotion_df) == 1

    log_df = pd.read_csv(paths["experiment_log"])
    assert log_df.empty
    assert list(log_df.columns) == LOG_COLUMNS

    assert paths["search_runs_dir"].exists()
    assert list(paths["search_runs_dir"].iterdir()) == []
    assert paths["sweeps_dir"].exists()
    assert list(paths["sweeps_dir"].iterdir()) == []

    assert paths["cache_dir"].exists()
    cache_entries = sorted(path.name for path in paths["cache_dir"].iterdir())
    assert cache_entries == ["fresh_start_state.json"]
    assert paths["runtime_root"].exists()
    assert paths["repos_dir"].exists()
    assert list(paths["repos_dir"].iterdir()) == []
    assert paths["worktrees_dir"].exists()
    assert list(paths["worktrees_dir"].iterdir()) == []


def test_reset_observatory_workspace_dry_run_leaves_workspace_unchanged(tmp_path: Path) -> None:
    config_path, search_policy_path = _write_configs(tmp_path)
    paths = _seed_workspace(tmp_path)

    result = reset_observatory_workspace(
        project_root=tmp_path,
        config_path=config_path,
        search_policy_path=search_policy_path,
        archive_root=tmp_path / "data" / "analysis" / "observatory_archives",
        archive_label="reset-dry-run",
        dry_run=True,
    )

    archive_dir = Path(result["archive_dir"])
    assert not archive_dir.exists()
    assert paths["run_dir"].exists()

    index_df = pd.read_csv(paths["index_path"])
    assert len(index_df) == 1
    log_df = pd.read_csv(paths["experiment_log"])
    assert len(log_df) == 1

    assert list(paths["search_runs_dir"].iterdir()) != []
    assert list(paths["sweeps_dir"].iterdir()) != []
    assert list(paths["cache_dir"].iterdir()) != []
    assert list(paths["repos_dir"].iterdir()) != []
    assert list(paths["worktrees_dir"].iterdir()) != []
