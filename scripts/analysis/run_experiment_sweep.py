#!/usr/bin/env python3
"""Batch experiment sweep runner for the ND population projection benchmarking system.

Created: 2026-03-12
Author: Claude Code / N. Haarstad

Purpose:
    Runs multiple benchmark experiments sequentially through the existing
    ``run_experiment.py`` orchestrator, accumulating results and regenerating
    the experiment dashboard at the end. Supports three input modes:
    individual spec files, a parameter grid definition, or all pending specs.
    Queue controls support bounded unattended execution through explicit
    run budgets, deterministic prioritization, resumable checkpoints, and
    bounded retry of previously failed specs.

Method:
    1. Parse CLI arguments to determine mode (spec list, grid, or pending).
    2. For grid mode, generate experiment spec YAML files from a parameter
       grid definition (cartesian product or zip pairing).
    3. For each spec, check the experiment log for duplicates and skip
       already-tested experiment IDs.
    4. Apply the requested queue priority, run budget, and resume checkpoint.
    5. Run each eligible spec via subprocess call to
       ``run_experiment.py --spec <path>``.
    6. Capture stdout/stderr and exit code; continue on failure while
       checkpointing attempts for future resume/retry.
    7. After all specs complete, print a summary table and regenerate the
       experiment dashboard.

Inputs:
    - Experiment spec YAML files (``--specs``) or parameter grid (``--grid``)
      or pending directory contents (``--pending``)
    - ``data/analysis/experiments/experiment_log.csv`` — for dedup checks
    - ``data/analysis/experiments/pending/`` — default location for pending specs

Outputs:
    - Executed experiment specs moved from pending to completed
    - Updated ``data/analysis/experiments/experiment_log.csv``
    - Regenerated ``data/analysis/experiments/experiment_dashboard.html``

Usage::

    # Run specific specs
    python scripts/analysis/run_experiment_sweep.py --specs spec1.yaml spec2.yaml

    # Run from parameter grid
    python scripts/analysis/run_experiment_sweep.py --grid grid.yaml

    # Run all pending specs
    python scripts/analysis/run_experiment_sweep.py --pending

    # Dry run (show what would run without executing)
    python scripts/analysis/run_experiment_sweep.py --grid grid.yaml --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

from cohort_projections.analysis.experiment_log import (  # noqa: E402
    get_tested_hypotheses,
    is_config_delta_tested,
)

logger = logging.getLogger(__name__)

# Directories
DEFAULT_PENDING_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "pending"
DEFAULT_COMPLETED_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "completed"
DEFAULT_SWEEP_STATE_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "sweeps"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the sweep runner.

    Args:
        argv: Argument list for testing. Uses sys.argv when None.

    Returns:
        Parsed namespace with mode and options.
    """
    parser = argparse.ArgumentParser(
        description="Run a batch of experiment specs through the benchmarking pipeline.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--specs",
        type=Path,
        nargs="+",
        help="Paths to experiment spec YAML files to run in order.",
    )
    group.add_argument(
        "--grid",
        type=Path,
        help="Path to a parameter grid YAML definition.",
    )
    group.add_argument(
        "--pending",
        action="store_true",
        help="Run all spec files found in the pending directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing any experiments.",
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=DEFAULT_PENDING_DIR,
        help="Override the pending specs directory (default: data/analysis/experiments/pending/).",
    )
    parser.add_argument(
        "--run-budget",
        type=int,
        default=None,
        help="Maximum number of experiments to launch in this invocation.",
    )
    parser.add_argument(
        "--priority",
        choices=["as-listed", "experiment-id", "modified-time"],
        default="as-listed",
        help=(
            "Queue ordering strategy before execution. "
            "'as-listed' preserves input order; 'modified-time' sorts oldest first."
        ),
    )
    parser.add_argument(
        "--reverse-priority",
        action="store_true",
        help="Reverse the chosen priority order after sorting.",
    )
    parser.add_argument(
        "--retry-failures",
        type=int,
        default=0,
        help="Number of additional attempts allowed for specs that previously failed in the resume file.",
    )
    parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help=(
            "JSON checkpoint path used to resume bounded sweeps. "
            "Relative paths resolve from the project root."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

# Required fields in a grid YAML
REQUIRED_GRID_FIELDS = frozenset(
    {"base_method", "base_config", "scope", "requested_by", "parameters"}
)


def load_grid(grid_path: Path) -> dict[str, Any]:
    """Load and validate a parameter grid YAML file.

    Args:
        grid_path: Path to the grid YAML file.

    Returns:
        Validated grid definition dict.

    Raises:
        FileNotFoundError: If the grid file does not exist.
        ValueError: If required fields are missing or parameters is empty.
    """
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    if not isinstance(grid, dict):
        raise ValueError(f"Grid must be a YAML mapping: {grid_path}")

    missing = REQUIRED_GRID_FIELDS - set(grid.keys())
    if missing:
        raise ValueError(f"Grid is missing required fields: {sorted(missing)}")

    parameters = grid["parameters"]
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError("Grid 'parameters' must be a non-empty mapping.")

    # Validate that all parameter values are lists
    for key, values in parameters.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"Parameter '{key}' must be a non-empty list, got: {type(values).__name__}"
            )

    return grid


def generate_grid_combinations(
    parameters: dict[str, list[Any]],
    mode: str = "cartesian",
) -> list[dict[str, Any]]:
    """Generate parameter combinations from a grid definition.

    Args:
        parameters: Mapping of parameter names to lists of values.
        mode: Either ``"cartesian"`` for full product or ``"zip"`` for paired.

    Returns:
        List of dicts, each representing one parameter combination.

    Raises:
        ValueError: If mode is unknown or zip lengths are unequal.
    """
    if mode == "cartesian":
        keys = list(parameters.keys())
        values = [parameters[k] for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

    if mode == "zip":
        keys = list(parameters.keys())
        values = [parameters[k] for k in keys]
        lengths = {len(v) for v in values}
        if len(lengths) > 1:
            raise ValueError(
                f"Zip mode requires all parameter lists to have equal length. "
                f"Got lengths: {[len(v) for v in values]}"
            )
        return [dict(zip(keys, combo, strict=True)) for combo in zip(*values, strict=True)]

    raise ValueError(f"Unknown grid mode: {mode!r}. Must be 'cartesian' or 'zip'.")


def _slugify_value(value: Any) -> str:
    """Convert a parameter value to a filesystem-safe slug fragment.

    Args:
        value: A scalar parameter value.

    Returns:
        Lowercase string safe for use in filenames and experiment IDs.
    """
    s = str(value).lower().replace(".", "p").replace(" ", "-")
    # Remove characters not safe for filenames
    return "".join(c for c in s if c.isalnum() or c in ("-", "_"))


def generate_experiment_id(
    base_slug: str,
    combo: dict[str, Any],
    index: int,
) -> str:
    """Generate a unique experiment_id for a grid combination.

    Format: ``exp-{date}-sweep-{slug}-{index:03d}``

    Args:
        base_slug: Base slug derived from the grid file name or parameters.
        combo: The parameter combination dict.
        index: Zero-based index within the sweep.

    Returns:
        A unique experiment_id string.
    """
    today = dt.datetime.now(tz=dt.UTC).date().strftime("%Y%m%d")
    # Build a short suffix from parameter values
    value_parts = [_slugify_value(v) for v in combo.values()]
    suffix = "-".join(value_parts)
    return f"exp-{today}-sweep-{base_slug}-{suffix}"


def generate_specs_from_grid(
    grid: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    """Generate experiment spec YAML files from a parameter grid.

    Args:
        grid: Validated grid definition.
        output_dir: Directory to write generated spec files.

    Returns:
        List of paths to generated spec files, in order.
    """
    mode = grid.get("mode", "cartesian")
    parameters = grid["parameters"]
    combinations = generate_grid_combinations(parameters, mode=mode)

    # Derive base slug from parameter names
    param_keys = list(parameters.keys())
    base_slug = "-".join(k.replace("_", "-") for k in param_keys)
    # Truncate slug to keep IDs reasonable
    if len(base_slug) > 40:
        base_slug = base_slug[:40]

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_paths: list[Path] = []

    for idx, combo in enumerate(combinations):
        experiment_id = generate_experiment_id(base_slug, combo, idx)

        # Build hypothesis from parameter names and values
        param_desc = ", ".join(f"{k}={v}" for k, v in combo.items())
        hypothesis = (
            f"Grid sweep testing {param_desc} on {grid['base_method']} / {grid['base_config']}."
        )

        # Build config delta
        config_delta: dict[str, Any] = {}
        for key, value in combo.items():
            config_delta[key] = value

        # Build benchmark label
        value_label = "-".join(_slugify_value(v) for v in combo.values())
        benchmark_label = f"sweep-{value_label}"

        spec: dict[str, Any] = {
            "experiment_id": experiment_id,
            "hypothesis": hypothesis,
            "base_method": grid["base_method"],
            "base_config": grid["base_config"],
            "config_delta": config_delta,
            "scope": grid["scope"],
            "benchmark_label": benchmark_label,
            "requested_by": grid["requested_by"],
        }

        # Preserve optional fields from the grid
        if "notes" in grid:
            spec["notes"] = grid["notes"]

        spec_path = output_dir / f"{experiment_id}.yaml"
        spec_path.write_text(
            yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        spec_paths.append(spec_path)

    return spec_paths


# ---------------------------------------------------------------------------
# Spec collection
# ---------------------------------------------------------------------------


def collect_pending_specs(pending_dir: Path) -> list[Path]:
    """Collect all YAML spec files from the pending directory.

    Args:
        pending_dir: Path to the pending experiments directory.

    Returns:
        Sorted list of YAML file paths found in the directory.
    """
    if not pending_dir.exists():
        logger.warning("Pending directory does not exist: %s", pending_dir)
        return []

    specs = sorted(pending_dir.glob("*.yaml"))
    if not specs:
        logger.info("No pending specs found in %s", pending_dir)
    return specs


def resolve_project_path(path: Path | None) -> Path | None:
    """Resolve *path* relative to the project root when needed."""
    if path is None:
        return None
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_resume_file() -> Path:
    """Return a timestamped default resume checkpoint path."""
    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_SWEEP_STATE_DIR / f"sweep_state_{stamp}.json"


def order_spec_paths(
    spec_paths: list[Path],
    *,
    priority: str = "as-listed",
    reverse: bool = False,
) -> list[Path]:
    """Order spec paths according to the requested priority strategy."""
    ordered = list(spec_paths)
    if priority == "experiment-id":
        ordered = sorted(ordered, key=_extract_experiment_id)
    elif priority == "modified-time":
        ordered = sorted(ordered, key=lambda path: path.stat().st_mtime)

    if reverse:
        ordered.reverse()
    return ordered


def load_resume_state(resume_file: Path | None) -> dict[str, Any]:
    """Load the sweep checkpoint file, returning an empty state when absent."""
    if resume_file is None or not resume_file.exists():
        return {"entries": {}}
    state = json.loads(resume_file.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"Resume file must contain a JSON object: {resume_file}")
    entries = state.get("entries")
    if not isinstance(entries, dict):
        state["entries"] = {}
    return state


def persist_resume_state(
    resume_file: Path | None,
    state: dict[str, Any],
) -> None:
    """Write the current checkpoint state to disk."""
    if resume_file is None:
        return
    resume_file.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = dt.datetime.now(tz=dt.UTC).isoformat()
    resume_file.write_text(
        json.dumps(state, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _resume_skip_result(
    experiment_id: str,
    spec_path: Path,
    *,
    outcome: str,
    key_delta: str = "-",
) -> dict[str, Any]:
    """Build a synthetic result row for skipped checkpoint entries."""
    return {
        "experiment_id": experiment_id,
        "spec_path": str(spec_path),
        "exit_code": -1,
        "outcome": outcome,
        "key_delta": key_delta,
        "stdout": "",
        "stderr": "",
    }


def _resume_entry(
    state: dict[str, Any],
    experiment_id: str,
) -> dict[str, Any] | None:
    """Return the checkpoint entry for *experiment_id* when present."""
    entries = state.get("entries", {})
    if not isinstance(entries, dict):
        return None
    entry = entries.get(experiment_id)
    return entry if isinstance(entry, dict) else None


def _record_resume_result(
    state: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Persist the latest execution result into the checkpoint state."""
    entries = state.setdefault("entries", {})
    experiment_id = str(result["experiment_id"])
    previous = entries.get(experiment_id, {})
    attempts = int(previous.get("attempts", 0)) + 1
    entries[experiment_id] = {
        "experiment_id": experiment_id,
        "attempts": attempts,
        "last_exit_code": int(result.get("exit_code", 0)),
        "last_outcome": str(result.get("outcome", "unknown")),
        "last_key_delta": str(result.get("key_delta", "-")),
        "last_spec_path": str(result.get("spec_path", "")),
        "updated_at_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def run_single_experiment(
    spec_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single experiment via subprocess to run_experiment.py.

    Args:
        spec_path: Path to the experiment spec YAML.
        dry_run: If True, pass --dry-run flag to the orchestrator.

    Returns:
        Dict with keys: experiment_id, spec_path, exit_code, outcome,
        key_delta, stdout, stderr.
    """
    runner_script = SCRIPT_DIR / "run_experiment.py"

    cmd = [sys.executable, str(runner_script), "--spec", str(spec_path)]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    # Try to parse the experiment_id from the spec
    experiment_id = _extract_experiment_id(spec_path)

    # Try to parse outcome from JSON output
    outcome = _parse_outcome_from_stdout(stdout)
    key_delta = _parse_key_delta_from_stdout(stdout)

    return {
        "experiment_id": experiment_id,
        "spec_path": str(spec_path),
        "exit_code": result.returncode,
        "outcome": outcome,
        "key_delta": key_delta,
        "stdout": stdout,
        "stderr": stderr,
    }


def _extract_experiment_id(spec_path: Path) -> str:
    """Extract experiment_id from a spec YAML file.

    Args:
        spec_path: Path to the spec YAML.

    Returns:
        The experiment_id string, or the filename stem as fallback.
    """
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        if isinstance(spec, dict):
            return str(spec.get("experiment_id", spec_path.stem))
    except Exception:
        pass
    return spec_path.stem


def _parse_outcome_from_stdout(stdout: str) -> str:
    """Parse the outcome classification from run_experiment.py JSON output.

    Args:
        stdout: Captured stdout from the subprocess.

    Returns:
        Outcome string or "unknown" if parsing fails.
    """
    # run_experiment.py prints a JSON block at the end
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(_extract_json_block(stdout))
                return str(data.get("outcome", "unknown"))
            except (json.JSONDecodeError, ValueError):
                pass
            break
    return "unknown"


def _extract_json_block(text: str) -> str:
    """Extract the last JSON object block from text output.

    Args:
        text: Full text output potentially containing a JSON block.

    Returns:
        The extracted JSON string.

    Raises:
        ValueError: If no JSON block is found.
    """
    lines = text.splitlines()
    # Find the last line that starts with "{"
    json_start = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("{"):
            json_start = i
            break

    if json_start < 0:
        raise ValueError("No JSON block found in output")

    # Collect lines from json_start to the closing "}"
    json_lines = []
    brace_depth = 0
    for i in range(json_start, len(lines)):
        line = lines[i]
        json_lines.append(line)
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            break

    return "\n".join(json_lines)


def _parse_key_delta_from_stdout(stdout: str) -> str:
    """Parse key metric delta from run_experiment.py JSON output.

    Args:
        stdout: Captured stdout from the subprocess.

    Returns:
        Summary string of the overall MAPE delta, or "-" if unavailable.
    """
    try:
        data = json.loads(_extract_json_block(stdout))
        details = data.get("classification_details", {})
        tradeoff = details.get("tradeoff_results", {})
        overall = tradeoff.get("county_mape_overall", {})
        delta = overall.get("delta")
        if delta is not None:
            return f"overall: {delta:+.2f}"
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return "-"


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------


def format_summary_table(results: list[dict[str, Any]]) -> str:
    """Format sweep results as a summary table.

    Args:
        results: List of result dicts from run_single_experiment or skipped entries.

    Returns:
        Formatted table string suitable for terminal output.
    """
    if not results:
        return "No experiments were processed."

    # Column widths
    id_width = max(len(r["experiment_id"]) for r in results)
    id_width = max(id_width, len("Experiment"))

    outcome_width = max(len(r.get("outcome", "-")) for r in results)
    outcome_width = max(outcome_width, len("Outcome"))

    delta_width = max(len(r.get("key_delta", "-")) for r in results)
    delta_width = max(delta_width, len("Key Delta"))

    # Build table
    sep_line = f"+{'-' * (id_width + 2)}+{'-' * (outcome_width + 2)}+{'-' * (delta_width + 2)}+"
    header = (
        f"| {'Experiment':<{id_width}} "
        f"| {'Outcome':<{outcome_width}} "
        f"| {'Key Delta':<{delta_width}} |"
    )

    lines = [
        "",
        "Sweep Summary:",
        sep_line,
        header,
        sep_line,
    ]

    for r in results:
        exp_id = r["experiment_id"]
        outcome = r.get("outcome", "-")
        key_delta = r.get("key_delta", "-")
        lines.append(
            f"| {exp_id:<{id_width}} | {outcome:<{outcome_width}} | {key_delta:<{delta_width}} |"
        )

    lines.append(sep_line)
    outcome_counts: dict[str, int] = {}
    for row in results:
        outcome = str(row.get("outcome", "-"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    counts = ", ".join(f"{outcome}={count}" for outcome, count in sorted(outcome_counts.items()))
    lines.append(f"Outcome counts: {counts}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dashboard regeneration
# ---------------------------------------------------------------------------


def regenerate_dashboard() -> bool:
    """Invoke build_experiment_dashboard.py to regenerate the dashboard.

    Returns:
        True if the dashboard was regenerated successfully, False otherwise.
    """
    dashboard_script = SCRIPT_DIR / "build_experiment_dashboard.py"
    if not dashboard_script.exists():
        logger.warning("Dashboard script not found: %s", dashboard_script)
        return False

    logger.info("Regenerating experiment dashboard...")
    result = subprocess.run(
        [sys.executable, str(dashboard_script)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        logger.warning(
            "Dashboard regeneration failed (rc=%d): %s",
            result.returncode,
            result.stderr.strip()[:500],
        )
        return False

    logger.info("Dashboard regenerated successfully.")
    return True


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_sweep(
    spec_paths: list[Path],
    dry_run: bool = False,
    run_budget: int | None = None,
    retry_failures: int = 0,
    resume_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a sweep of experiments with dedup checking.

    Args:
        spec_paths: Ordered list of spec file paths to process.
        dry_run: If True, validate and display without running.
        run_budget: Optional cap on launched experiments in this invocation.
        retry_failures: Number of additional attempts allowed for prior failures.
        resume_file: Optional checkpoint path for resumable bounded sweeps.

    Returns:
        List of result dicts, one per spec (including skipped entries).
    """
    if run_budget is not None and run_budget <= 0:
        raise ValueError("run_budget must be positive when provided.")
    if retry_failures < 0:
        raise ValueError("retry_failures must be zero or greater.")

    tested = get_tested_hypotheses()
    resume_state = load_resume_state(resume_file)
    results: list[dict[str, Any]] = []
    launched = 0

    total = len(spec_paths)
    logger.info("Sweep contains %d experiment(s).", total)

    for i, spec_path in enumerate(spec_paths, 1):
        experiment_id = _extract_experiment_id(spec_path)
        logger.info(
            "[%d/%d] Processing: %s (%s)",
            i,
            total,
            experiment_id,
            spec_path.name,
        )

        entry = _resume_entry(resume_state, experiment_id)
        if entry is not None:
            last_exit_code = int(entry.get("last_exit_code", 1))
            attempts = int(entry.get("attempts", 0))
            last_key_delta = str(entry.get("last_key_delta", "-"))
            if last_exit_code == 0:
                logger.info(
                    "Skipping %s — already completed in resume file %s.",
                    experiment_id,
                    resume_file,
                )
                results.append(
                    _resume_skip_result(
                        experiment_id,
                        spec_path,
                        outcome="(resume-skip)",
                        key_delta=last_key_delta,
                    )
                )
                continue
            if attempts > retry_failures:
                logger.info(
                    "Skipping %s — retry budget exhausted (%d prior attempt(s), %d retry allowed).",
                    experiment_id,
                    attempts,
                    retry_failures,
                )
                results.append(
                    _resume_skip_result(
                        experiment_id,
                        spec_path,
                        outcome="(retry-exhausted)",
                        key_delta=last_key_delta,
                    )
                )
                continue

        # Dedup check — experiment ID level
        if experiment_id in tested:
            logger.info(
                "Skipping %s — already tested (found in experiment log).",
                experiment_id,
            )
            results.append(
                {
                    "experiment_id": experiment_id,
                    "spec_path": str(spec_path),
                    "exit_code": -1,
                    "outcome": "(skipped)",
                    "key_delta": "-",
                    "stdout": "",
                    "stderr": "",
                }
            )
            continue

        # Dedup check — parameter-level matching
        try:
            spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
            config_delta = spec_data.get("config_delta") if isinstance(spec_data, dict) else None
            if (
                isinstance(config_delta, dict)
                and config_delta
                and is_config_delta_tested(config_delta)
            ):
                logger.info(
                    "Skipping %s — config delta already tested (parameter-level match).",
                    experiment_id,
                )
                results.append(
                    {
                        "experiment_id": experiment_id,
                        "spec_path": str(spec_path),
                        "exit_code": -1,
                        "outcome": "(skipped)",
                        "key_delta": "-",
                        "stdout": "",
                        "stderr": "",
                    }
                )
                continue
        except Exception:
            pass  # If we can't read the spec, proceed to run it

        if run_budget is not None and launched >= run_budget:
            logger.info(
                "Deferring %s — run budget of %d reached.",
                experiment_id,
                run_budget,
            )
            results.append(
                _resume_skip_result(
                    experiment_id,
                    spec_path,
                    outcome="(deferred-budget)",
                )
            )
            continue

        if dry_run:
            logger.info("  [dry-run] Would run: %s", spec_path)
            result = run_single_experiment(spec_path, dry_run=True)
            result["outcome"] = "(dry-run)"
            result["key_delta"] = "-"
            launched += 1
            results.append(result)
            continue

        # Execute
        result = run_single_experiment(spec_path, dry_run=False)
        launched += 1

        if result["exit_code"] != 0:
            logger.warning(
                "Experiment %s failed (exit code %d). Continuing with next spec.",
                experiment_id,
                result["exit_code"],
            )
            if result["outcome"] == "unknown":
                result["outcome"] = "error"
        else:
            logger.info(
                "Experiment %s completed: %s",
                experiment_id,
                result["outcome"],
            )

        # Add to tested set so subsequent duplicate IDs in this sweep are skipped
        tested.add(experiment_id)
        _record_resume_result(resume_state, result)
        persist_resume_state(resume_file, resume_state)
        results.append(result)

    return results


def main(argv: list[str] | None = None) -> None:
    """Entry point for the batch experiment sweep runner.

    Args:
        argv: Argument list for testing. Uses sys.argv when None.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)
    if args.run_budget is not None and args.run_budget <= 0:
        raise ValueError("--run-budget must be positive when provided.")
    if args.retry_failures < 0:
        raise ValueError("--retry-failures must be zero or greater.")

    # Determine spec paths based on mode
    if args.specs:
        spec_paths = list(args.specs)
        logger.info("Spec list mode: %d spec(s) provided.", len(spec_paths))

    elif args.grid:
        logger.info("Grid mode: loading grid from %s", args.grid)
        grid = load_grid(args.grid)
        spec_paths = generate_specs_from_grid(grid, output_dir=args.pending_dir)
        logger.info("Generated %d spec(s) from grid.", len(spec_paths))

    elif args.pending:
        logger.info("Pending mode: scanning %s", args.pending_dir)
        spec_paths = collect_pending_specs(args.pending_dir)
        logger.info("Found %d pending spec(s).", len(spec_paths))

    else:
        logger.error("No mode specified. Use --specs, --grid, or --pending.")
        sys.exit(1)

    if not spec_paths:
        logger.info("No specs to process. Exiting.")
        return

    spec_paths = order_spec_paths(
        spec_paths,
        priority=args.priority,
        reverse=args.reverse_priority,
    )

    resume_file = resolve_project_path(args.resume_file)
    if resume_file is None and not args.dry_run:
        resume_file = default_resume_file()

    logger.info(
        "Queue controls: priority=%s, reverse=%s, run_budget=%s, retry_failures=%d, concurrency=sequential",
        args.priority,
        args.reverse_priority,
        args.run_budget if args.run_budget is not None else "unbounded",
        args.retry_failures,
    )
    if resume_file is not None:
        logger.info("Resume checkpoint: %s", resume_file)

    # Run the sweep
    results = run_sweep(
        spec_paths,
        dry_run=args.dry_run,
        run_budget=args.run_budget,
        retry_failures=args.retry_failures,
        resume_file=resume_file,
    )

    # Print summary table
    summary = format_summary_table(results)
    print(summary)  # noqa: T201

    # Regenerate dashboard (skip on dry-run)
    if not args.dry_run:
        regenerate_dashboard()

    # Report final counts
    ran = sum(1 for r in results if r["outcome"] not in ("(skipped)", "(dry-run)"))
    skipped = sum(1 for r in results if r["outcome"] == "(skipped)")
    errors = sum(1 for r in results if r["outcome"] == "error")

    logger.info(
        "Sweep complete: %d ran, %d skipped, %d errors out of %d total.",
        ran,
        skipped,
        errors,
        len(results),
    )


if __name__ == "__main__":
    main()
