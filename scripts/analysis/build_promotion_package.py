#!/usr/bin/env python3
"""Build a review-ready promotion package for one benchmark bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from cohort_projections.analysis.benchmark_contract import (
    validate_manifest,
    validate_scorecard_columns,
)
from cohort_projections.analysis.benchmarking import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_HISTORY_DIR,
    load_aliases,
    render_benchmark_decision_record,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a promotion review package for one benchmark run.")
    parser.add_argument("--run-id", required=True, help="Benchmark run identifier.")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Benchmark history directory.",
    )
    parser.add_argument(
        "--alias",
        default="county_champion",
        help="Alias whose before/after targets should be documented.",
    )
    parser.add_argument(
        "--alias-path",
        type=Path,
        default=DEFAULT_ALIAS_PATH,
        help="Alias mapping YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/promotion_package.",
    )
    return parser.parse_args()


def _copy_if_exists(source: Path, destination: Path) -> None:
    """Copy a file when it exists."""
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    args = _parse_args()
    run_dir = args.history_dir / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Benchmark run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    scorecard_path = run_dir / "summary_scorecard.csv"
    comparison_path = run_dir / "comparison_to_champion.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scorecard = pd.read_csv(scorecard_path)
    validate_manifest(manifest)
    validate_scorecard_columns(scorecard)

    comparison: dict[str, Any] = {}
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

    output_dir = args.output_dir or (run_dir / "promotion_package")
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(scorecard_path, output_dir / "summary_scorecard.csv")
    _copy_if_exists(comparison_path, output_dir / "comparison_to_champion.json")
    _copy_if_exists(manifest_path, output_dir / "manifest.json")
    _copy_if_exists(run_dir / "runtime_summary.json", output_dir / "runtime_summary.json")
    _copy_if_exists(run_dir / "qc_summary.json", output_dir / "qc_summary.json")

    aliases = load_aliases(args.alias_path)
    prior_mapping = aliases.get(args.alias, {})
    challenger = {}
    if comparison.get("challengers"):
        challenger = comparison["challengers"][0]

    package_manifest = {
        "run_id": args.run_id,
        "scope": manifest.get("scope", ""),
        "alias_name": args.alias,
        "alias_before": prior_mapping,
        "alias_after_candidate": {
            "method_id": challenger.get("method_id", ""),
            "config_id": challenger.get("config_id", ""),
        },
        "files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    decision_record = render_benchmark_decision_record(manifest, scorecard, comparison)
    (output_dir / "draft_decision_record.md").write_text(
        decision_record,
        encoding="utf-8",
    )

    print(output_dir)


if __name__ == "__main__":
    main()
