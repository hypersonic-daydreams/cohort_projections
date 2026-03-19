#!/usr/bin/env python3
"""Rebuild benchmark_history/index.csv from complete benchmark bundles.

This script scans ``data/analysis/benchmark_history/`` for dated run
directories, registers only bundles that contain both ``manifest.json`` and
``summary_scorecard.csv``, and rewrites ``index.csv`` from that canonical
artifact set. Incomplete run directories are reported but not treated as valid
benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cohort_projections.analysis.benchmarking import (
    DEFAULT_HISTORY_DIR,
    rebuild_benchmark_index,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild benchmark_history/index.csv from complete benchmark bundles.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help="Benchmark history root containing dated run bundles.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = rebuild_benchmark_index(history_dir=args.history_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
