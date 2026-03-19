#!/usr/bin/env python3
"""Archive active Projection Observatory state and reinitialize a clean workspace.

Created: 2026-03-19
Author: Codex / N. Haarstad

Purpose:
    Provide a non-destructive "start fresh" operation for the Projection
    Observatory. Existing active benchmark runs, experiment logs, cache files,
    and autonomous-search runtime state are archived to a timestamped snapshot,
    then the active workspace is reset to an empty first-run state.

Method:
    1. Load Observatory and search-policy paths from config.
    2. Archive the current active benchmark history, cache, experiment state,
       and autonomous-search runtime state into `data/analysis/observatory_archives/`.
    3. Recreate an empty `benchmark_history/index.csv` header.
    4. Recreate an empty `experiment_log.csv` header plus clean `search_runs/`
       and `sweeps/` directories.
    5. Leave `promotion_history.csv` in place by default so alias-governance
       provenance is preserved even though active run evidence is cleared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohort_projections.analysis.observatory.workspace_reset import (
    DEFAULT_ARCHIVE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_SEARCH_POLICY_PATH,
    reset_observatory_workspace,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive active Observatory state and reinitialize a clean workspace.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to observatory_config.yaml.",
    )
    parser.add_argument(
        "--search-policy",
        type=Path,
        default=DEFAULT_SEARCH_POLICY_PATH,
        help="Path to observatory_search_policy.yaml.",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Directory that will receive archived reset snapshots.",
    )
    parser.add_argument(
        "--archive-label",
        help="Optional explicit name for the archive snapshot directory.",
    )
    parser.add_argument(
        "--drop-promotion-history",
        action="store_true",
        help="Archive promotion_history.csv too instead of keeping it in the active workspace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the reset plan without moving or recreating files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = reset_observatory_workspace(
        config_path=args.config,
        search_policy_path=args.search_policy,
        archive_root=args.archive_root,
        archive_label=args.archive_label,
        keep_promotion_history=not args.drop_promotion_history,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
