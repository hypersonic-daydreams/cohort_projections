#!/usr/bin/env python3
"""Register an existing benchmark bundle into benchmark_history/index.csv.

This is the scope-aware bridge for non-county benchmarking workflows such as
future place-level benchmark bundles that already have a manifest and scorecard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cohort_projections.analysis.benchmark_contract import (
    validate_manifest,
    validate_scorecard_columns,
)
from cohort_projections.analysis.benchmarking import DEFAULT_HISTORY_DIR, append_benchmark_index


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an existing benchmark bundle.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Benchmark bundle directory.")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Benchmark history root containing index.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = args.run_dir / "manifest.json"
    scorecard_path = args.run_dir / "summary_scorecard.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scorecard = pd.read_csv(scorecard_path)
    validate_manifest(manifest)
    validate_scorecard_columns(scorecard)

    append_benchmark_index(
        index_path=args.history_dir / "index.csv",
        scorecard=scorecard,
        manifest_path=manifest_path,
        benchmark_label=str(manifest.get("benchmark_label", "")),
        benchmark_contract_version=str(manifest.get("benchmark_contract_version", "")),
        git_commit=str(manifest.get("git_commit", "")),
        champion_method_id=str(manifest.get("champion_method_id", "")),
        decision_id=str(manifest.get("decision_id", "")) or None,
        decision_status=str(manifest.get("decision_status", "pending")),
    )
    print(args.run_dir)


if __name__ == "__main__":
    main()
