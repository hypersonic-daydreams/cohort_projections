#!/usr/bin/env python3
"""Canonical full-pipeline projection entrypoint.

This wrapper keeps a stable Python command for full projection runs while
executing the shell orchestrator at `scripts/pipeline/run_complete_pipeline.sh`.

Usage:
    python scripts/projections/run_all_projections.py
    python scripts/projections/run_all_projections.py --dry-run
    python scripts/projections/run_all_projections.py --resume
    python scripts/projections/run_all_projections.py --fail-fast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI options and pass-through arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete North Dakota projection pipeline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without producing pipeline outputs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume projections from previous runs when supported.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first stage failure.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the shell pipeline runner and return its exit code."""
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    shell_runner = repo_root / "scripts" / "pipeline" / "run_complete_pipeline.sh"

    cmd = ["bash", str(shell_runner)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.resume:
        cmd.append("--resume")
    if args.fail_fast:
        cmd.append("--fail-fast")

    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
