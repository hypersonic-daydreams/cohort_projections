#!/usr/bin/env python3
"""Run a non-destructive multi-seed stability sweep for Module 9 Monte Carlo outputs.

This script runs `module_9_scenario_modeling.py` repeatedly with different RNG seeds,
writing each run's artifacts to a unique, timestamped output subdirectory so that
canonical artifacts (e.g., `results/module_9_scenario_modeling.json`) are never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def _parse_seed_spec(seed_spec: str) -> list[int]:
    """Parse a seed specification into a list of integers.

    Supported formats:
    - "42,43,44"
    - "42-51" (inclusive range)
    """
    spec = seed_spec.strip()
    if not spec:
        raise ValueError("Seed spec is empty.")

    if "-" in spec and "," not in spec:
        start_str, end_str = (part.strip() for part in spec.split("-", maxsplit=1))
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"Invalid seed range: {seed_spec}")
        return list(range(start, end + 1))

    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError(f"No seeds parsed from: {seed_spec}")
    return seeds


def _read_metrics(run_json_path: Path) -> dict[str, float | int]:
    data = json.loads(run_json_path.read_text())
    mc = data["results"]["monte_carlo"]
    baseline = mc["baseline_only"]
    wave = mc["wave_adjusted"]
    return {
        "n_draws": int(mc["n_draws"]),
        "baseline_median_2045": float(baseline["median_2045"]),
        "baseline_ci95_lo_2045": float(baseline["ci_95_2045"][0]),
        "baseline_ci95_hi_2045": float(baseline["ci_95_2045"][1]),
        "wave_median_2045": float(wave["median_2045"]),
        "wave_ci95_lo_2045": float(wave["ci_95_2045"][0]),
        "wave_ci95_hi_2045": float(wave["ci_95_2045"][1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed stability sweep for Module 9 Monte Carlo results."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42-51",
        help='Seed list/range, e.g. "42-51" or "42,43,44" (default: 42-51).',
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=25000,
        help="Monte Carlo draw count per run (default: 25000).",
    )
    parser.add_argument(
        "--duration-tag",
        type=str,
        default="P0",
        help="Duration/hazard bundle tag to pass through to Module 9 (default: P0).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker processes for Module 9 (default: 1; avoids multiprocessing issues).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Draws per chunk for Module 9 (default: 1000).",
    )
    parser.add_argument(
        "--base-subdir",
        type=str,
        default=None,
        help=(
            "Base output subdirectory under `results/` and `figures/`. "
            "Defaults to a timestamped `seed_sweeps/<timestamp>` directory."
        ),
    )
    parser.add_argument(
        "--keep-plots",
        action="store_true",
        help="Generate Module 9 plots for each seed (default: false).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    seeds = _parse_seed_spec(args.seeds)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_subdir = args.base_subdir or f"seed_sweeps/{timestamp}"

    module_dir = Path(__file__).parent
    results_root = module_dir / "results"
    sweep_root = results_root / base_subdir
    sweep_root.mkdir(parents=True, exist_ok=False)

    module_9_script = module_dir / "module_9_scenario_modeling.py"
    if not module_9_script.exists():
        raise FileNotFoundError(f"Missing Module 9 script: {module_9_script}")

    LOGGER.info("Seed sweep output root: %s", sweep_root)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        run_subdir = f"{base_subdir}/seed_{seed}"
        cmd = [
            sys.executable,
            str(module_9_script),
            "--n-draws",
            str(args.n_draws),
            "--seed",
            str(seed),
            "--n-jobs",
            str(args.n_jobs),
            "--chunk-size",
            str(args.chunk_size),
            "--duration-tag",
            str(args.duration_tag),
            "--output-subdir",
            run_subdir,
        ]
        if not args.keep_plots:
            cmd.append("--skip-plots")

        LOGGER.info("Running seed=%s (%s draws)...", seed, args.n_draws)
        subprocess.run(cmd, cwd=str(module_dir), check=True)

        run_json_path = results_root / run_subdir / "module_9_scenario_modeling.json"
        metrics = _read_metrics(run_json_path)
        rows.append({"seed": seed, **metrics})

    # Write summary CSV.
    csv_path = sweep_root / "module_9_seed_sweep_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write a small JSON summary for easy parsing.
    summary_path = sweep_root / "module_9_seed_sweep_summary.json"
    summary_path.write_text(json.dumps({"rows": rows}, indent=2, default=str))

    LOGGER.info("Wrote seed sweep summary: %s", csv_path)
    LOGGER.info("Wrote seed sweep summary: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
