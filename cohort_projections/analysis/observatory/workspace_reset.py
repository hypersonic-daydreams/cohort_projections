"""Fresh-start reset utilities for the Projection Observatory workspace.

This module provides a non-destructive reset path for the active Observatory
workspace. Existing benchmark bundles, cache files, search sessions, and
experiment logs are archived into a timestamped directory, then the active
workspace is reinitialized into an empty-but-valid first-run state.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cohort_projections.analysis.benchmark_contract import BENCHMARK_INDEX_COLUMNS
from cohort_projections.analysis.experiment_log import LOG_COLUMNS
from cohort_projections.analysis.observatory.results_store import load_observatory_config
from cohort_projections.analysis.observatory.search_policy import load_search_policy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "observatory_config.yaml"
DEFAULT_SEARCH_POLICY_PATH = PROJECT_ROOT / "config" / "observatory_search_policy.yaml"
DEFAULT_ARCHIVE_ROOT = PROJECT_ROOT / "data" / "analysis" / "observatory_archives"
FRESH_START_MARKER_FILENAME = "fresh_start_state.json"


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _ensure_unique_archive_dir(archive_root: Path, archive_label: str) -> Path:
    candidate = archive_root / archive_label
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = archive_root / f"{archive_label}-{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _write_header_only_csv(path: Path, fieldnames: tuple[str, ...] | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()


def _move_children(
    source_dir: Path,
    destination_dir: Path,
    *,
    preserve_names: set[str] | None = None,
    dry_run: bool = False,
) -> list[str]:
    preserve_names = preserve_names or set()
    if not source_dir.exists():
        return []

    moved: list[str] = []
    for child in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if child.name in preserve_names:
            continue
        moved.append(child.name)
        if dry_run:
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(child), str(destination_dir / child.name))
    return moved


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in handle)
    return max(0, row_count - 1)


def build_fresh_start_marker_path(
    *,
    project_root: Path = PROJECT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Return the marker path that enables Observatory fresh-start mode."""
    project_root = Path(project_root).resolve()
    config_path = _resolve_project_path(project_root, config_path).resolve()
    config = load_observatory_config(config_path)
    cache_dir = _resolve_project_path(project_root, config["cache_dir"]).resolve()
    return cache_dir / FRESH_START_MARKER_FILENAME


def reset_observatory_workspace(
    *,
    project_root: Path = PROJECT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    search_policy_path: Path = DEFAULT_SEARCH_POLICY_PATH,
    archive_root: Path = DEFAULT_ARCHIVE_ROOT,
    archive_label: str | None = None,
    keep_promotion_history: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Archive active Observatory state and reinitialize a clean workspace.

    Args:
        project_root: Repository root used to resolve relative config paths.
        config_path: Observatory config YAML path.
        search_policy_path: Autonomous-search policy YAML path.
        archive_root: Parent directory that stores archived reset snapshots.
        archive_label: Optional explicit archive directory name.
        keep_promotion_history: When ``True``, leave
            ``benchmark_history/promotion_history.csv`` in place because it
            tracks alias governance rather than active run evidence.
        dry_run: When ``True``, report the planned reset without mutating the
            workspace.

    Returns:
        Summary metadata describing archived artifacts and the reinitialized
        workspace paths.
    """
    project_root = Path(project_root).resolve()
    config_path = _resolve_project_path(project_root, config_path).resolve()
    search_policy_path = _resolve_project_path(project_root, search_policy_path).resolve()
    archive_root = _resolve_project_path(project_root, archive_root).resolve()

    config = load_observatory_config(config_path)
    search_policy = load_search_policy(search_policy_path, project_root=project_root)

    history_dir = _resolve_project_path(project_root, config["history_dir"]).resolve()
    cache_dir = _resolve_project_path(project_root, config["cache_dir"]).resolve()
    experiment_log_path = _resolve_project_path(project_root, config["experiment_log"]).resolve()
    experiments_dir = experiment_log_path.parent.resolve()
    sweeps_dir = experiments_dir / "sweeps"
    runtime_root = search_policy.runtime_root.resolve()
    search_session_root = search_policy.session_root.resolve()
    worktree_root = search_policy.worktree_root.resolve()
    mirror_repo_parent = search_policy.mirror_repo.parent.resolve()
    fresh_start_marker_path = cache_dir / FRESH_START_MARKER_FILENAME

    timestamp = datetime.now(UTC)
    archive_name = archive_label or f"fresh-start-{timestamp.strftime('%Y%m%d-%H%M%S')}"
    archive_dir = _ensure_unique_archive_dir(archive_root, archive_name)

    benchmark_preserve = {"promotion_history.csv"} if keep_promotion_history else set()
    benchmark_history_archive = archive_dir / "benchmark_history"
    experiments_archive = archive_dir / "experiments"
    cache_archive = archive_dir / "observatory_cache"
    runtime_archive = archive_dir / "observatory_runtime"

    benchmark_history_moved = _move_children(
        history_dir,
        benchmark_history_archive,
        preserve_names=benchmark_preserve,
        dry_run=dry_run,
    )
    experiments_moved = _move_children(experiments_dir, experiments_archive, dry_run=dry_run)
    cache_moved = _move_children(cache_dir, cache_archive, dry_run=dry_run)
    runtime_moved = _move_children(runtime_root, runtime_archive, dry_run=dry_run)

    archive_manifest = {
        "created_at_utc": timestamp.isoformat(),
        "project_root": str(project_root),
        "dry_run": dry_run,
        "keep_promotion_history": keep_promotion_history,
        "archive_dir": str(archive_dir),
        "archived": {
            "benchmark_history": benchmark_history_moved,
            "experiments": experiments_moved,
            "cache": cache_moved,
            "runtime": runtime_moved,
        },
        "preserved": {
            "promotion_history_path": str(history_dir / "promotion_history.csv")
            if keep_promotion_history
            else "",
        },
        "pre_reset_counts": {
            "benchmark_runs": sum(1 for name in benchmark_history_moved if name.startswith("br-")),
            "experiment_log_rows": _count_csv_rows(experiment_log_path),
            "search_session_entries": sum(
                1 for name in experiments_moved if name == search_session_root.name
            ),
        },
    }

    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "reset_manifest.json").write_text(
            json.dumps(archive_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        history_dir.mkdir(parents=True, exist_ok=True)
        _write_header_only_csv(history_dir / "index.csv", BENCHMARK_INDEX_COLUMNS)

        cache_dir.mkdir(parents=True, exist_ok=True)
        fresh_start_marker_path.write_text(
            json.dumps(
                {
                    "mode": "fresh_start",
                    "created_at_utc": timestamp.isoformat(),
                    "archive_dir": str(archive_dir),
                    "keep_promotion_history": keep_promotion_history,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        runtime_root.mkdir(parents=True, exist_ok=True)
        mirror_repo_parent.mkdir(parents=True, exist_ok=True)
        worktree_root.mkdir(parents=True, exist_ok=True)

        experiments_dir.mkdir(parents=True, exist_ok=True)
        search_session_root.mkdir(parents=True, exist_ok=True)
        sweeps_dir.mkdir(parents=True, exist_ok=True)
        _write_header_only_csv(experiment_log_path, LOG_COLUMNS)

    return {
        "archive_dir": str(archive_dir),
        "dry_run": dry_run,
        "keep_promotion_history": keep_promotion_history,
        "moved": {
            "benchmark_history": benchmark_history_moved,
            "experiments": experiments_moved,
            "cache": cache_moved,
            "runtime": runtime_moved,
        },
        "active_paths": {
            "history_dir": str(history_dir),
            "index_path": str(history_dir / "index.csv"),
            "promotion_history_path": str(history_dir / "promotion_history.csv"),
            "cache_dir": str(cache_dir),
            "experiment_log_path": str(experiment_log_path),
            "fresh_start_marker_path": str(fresh_start_marker_path),
            "search_session_root": str(search_session_root),
            "sweeps_dir": str(sweeps_dir),
            "runtime_root": str(runtime_root),
            "mirror_repo_parent": str(mirror_repo_parent),
            "worktree_root": str(worktree_root),
        },
    }
