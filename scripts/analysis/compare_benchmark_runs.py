#!/usr/bin/env python3
"""Build a human-readable challenger-vs-champion summary from a benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cohort_projections.analysis.benchmarking import (
    DEFAULT_HISTORY_DIR,
    PROJECT_ROOT,
    render_benchmark_decision_record,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark run outputs.")
    parser.add_argument("--run-id", required=True, help="Benchmark run identifier.")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Benchmark history directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "reviews" / "benchmark_decisions",
        help="Directory for the markdown comparison record.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.history_dir / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scorecard = pd.read_csv(run_dir / "summary_scorecard.csv")
    comparison = json.loads((run_dir / "comparison_to_champion.json").read_text(encoding="utf-8"))

    markdown = render_benchmark_decision_record(manifest, scorecard, comparison)
    champion = comparison["champion_method_id"]
    challenger = comparison["challengers"][0]["method_id"]
    decision_id = f"{manifest['run_date']}-{challenger}-vs-{champion}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{decision_id}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(output_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
