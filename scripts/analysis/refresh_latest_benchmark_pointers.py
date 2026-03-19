#!/usr/bin/env python3
"""Refresh convenience latest-pointer links for benchmark alias targets.

Created: 2026-03-19
Author: Codex / N. Haarstad

This script rebuilds the disposable ``data/analysis/benchmark_history/latest/``
layer from immutable benchmark history and alias mappings. The dated run
directories and ``index.csv`` remain canonical; ``latest/`` is convenience-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cohort_projections.analysis.benchmarking import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_HISTORY_DIR,
    refresh_latest_benchmark_pointers,
)
from cohort_projections.utils.reproducibility import log_execution


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh convenience latest-pointer links for benchmark aliases."
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Benchmark history directory containing index.csv and dated run bundles.",
    )
    parser.add_argument(
        "--alias-path",
        type=Path,
        default=DEFAULT_ALIAS_PATH,
        help="Alias mapping YAML path.",
    )
    parser.add_argument(
        "--scope",
        default="county",
        help="Alias scope prefix to refresh, such as county.",
    )
    parser.add_argument(
        "--alias",
        help="Optional single alias to refresh instead of rebuilding all aliases for the scope.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parameters = {
        "history_dir": str(args.history_dir),
        "alias_path": str(args.alias_path),
        "scope": args.scope,
        "alias": args.alias or "",
    }
    outputs = [args.history_dir / "latest"]
    with log_execution(__file__, parameters=parameters, outputs=outputs):
        results = refresh_latest_benchmark_pointers(
            history_dir=args.history_dir,
            alias_path=args.alias_path,
            scope=args.scope,
            alias_name=args.alias,
        )

    for result in results:
        run_display = result["run_id"] or "none"
        print(
            f"{result['alias_name']}: {result['status']} "
            f"(method={result['method_id']}, config={result['config_id']}, run={run_display})"
        )


if __name__ == "__main__":
    main()
