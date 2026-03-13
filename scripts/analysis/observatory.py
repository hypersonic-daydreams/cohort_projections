#!/usr/bin/env python3
"""Projection Observatory: unified CLI for results analysis and experiment planning.

Created: 2026-03-12
Author: Claude Code / N. Haarstad

Purpose:
    Provides a single command-line entry point for all Observatory operations:
    inventory status, N-way comparison, ranking, recommendation, pending
    experiment execution, HTML report generation, and cache management.

Method:
    1. Parse subcommand and options via argparse.
    2. Load observatory config from YAML (default: ``config/observatory_config.yaml``).
    3. Construct ResultsStore (always) and Comparator/Recommender/VariantCatalog
       as needed, with graceful degradation if modules are not yet available.
    4. Dispatch to the appropriate subcommand handler.
    5. Print results to stdout; write artifacts to the configured output dirs.

Inputs:
    - ``config/observatory_config.yaml`` — observatory settings
    - ``data/analysis/benchmark_history/`` — benchmark run artifacts
    - ``data/analysis/experiments/experiment_log.csv`` — experiment outcomes
    - ``config/observatory_variants.yaml`` — variant catalog (optional)

Outputs:
    - Console output for ``status``, ``compare``, ``rank``, ``recommend``
    - HTML report for ``report``
    - Experiment spec files + subprocess calls for ``run-pending``
    - Refreshed cache for ``refresh``

Usage::

    python scripts/analysis/observatory.py status
    python scripts/analysis/observatory.py compare [--top N]
    python scripts/analysis/observatory.py rank <metric>
    python scripts/analysis/observatory.py recommend
    python scripts/analysis/observatory.py run-pending [--dry-run]
    python scripts/analysis/observatory.py report [--output path]
    python scripts/analysis/observatory.py refresh
    python scripts/analysis/observatory.py diff <run_a> <run_b>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)
DEFAULT_SWEEP_STATE_DIR = PROJECT_ROOT / "data" / "analysis" / "experiments" / "sweeps"


# ---------------------------------------------------------------------------
# ANSI color utilities (mirrors report.py; duplicated here for CLI-only paths
# that don't go through the report module)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c_green(s: str) -> str:
    """Wrap *s* in ANSI green."""
    return f"\033[32m{s}\033[0m" if _USE_COLOR else s


def _c_red(s: str) -> str:
    """Wrap *s* in ANSI red."""
    return f"\033[31m{s}\033[0m" if _USE_COLOR else s


def _c_yellow(s: str) -> str:
    """Wrap *s* in ANSI yellow."""
    return f"\033[33m{s}\033[0m" if _USE_COLOR else s


def _c_bold(s: str) -> str:
    """Wrap *s* in ANSI bold."""
    return f"\033[1m{s}\033[0m" if _USE_COLOR else s


def _c_delta(value: float, threshold: float = 0.005) -> str:
    """Color a delta value: green if negative (improvement), red if positive."""
    formatted = f"{value:+.3f}"
    if abs(value) < threshold:
        return _c_yellow(formatted)
    elif value < 0:
        return _c_green(formatted)
    else:
        return _c_red(formatted)


# ---------------------------------------------------------------------------
# Lazy imports with graceful degradation
# ---------------------------------------------------------------------------

_IMPORT_ERRORS: dict[str, str] = {}


def _try_import(name: str) -> Any:
    """Attempt to import a module and return it, or None on failure."""
    if name in _IMPORT_ERRORS:
        return None
    try:
        import importlib

        return importlib.import_module(name)
    except ImportError as e:
        _IMPORT_ERRORS[name] = str(e)
        logger.debug("Optional import failed: %s (%s)", name, e)
        return None


def _load_results_store(config_path: Path) -> Any:
    """Load ResultsStore from config, or None on failure."""
    try:
        from cohort_projections.analysis.observatory.results_store import (
            ResultsStore,
        )

        return ResultsStore.from_config(config_path)
    except Exception as e:
        logger.error("Failed to load ResultsStore: %s", e)
        return None


def _load_comparator(store: Any, config: dict[str, Any]) -> Any:
    """Load ObservatoryComparator, or None if not available."""
    mod = _try_import("cohort_projections.analysis.observatory.comparator")
    if mod is None:
        return None
    try:
        cls = mod.ObservatoryComparator
        return cls(store=store, config=config)
    except Exception as e:
        logger.warning("Failed to construct ObservatoryComparator: %s", e)
        return None


def _load_recommender(
    store: Any,
    config: dict[str, Any],
    comparator: Any | None = None,
    bounds_catalog: Any | None = None,
) -> Any:
    """Load ObservatoryRecommender, or None if not available."""
    mod = _try_import("cohort_projections.analysis.observatory.recommender")
    if mod is None:
        return None
    try:
        # Build a comparator if not provided
        if comparator is None:
            comparator = _load_comparator(store, config)
        if comparator is None:
            return None
        cls = mod.ObservatoryRecommender
        return cls(
            store=store,
            comparator=comparator,
            config=config,
            bounds_catalog=bounds_catalog,
        )
    except Exception as e:
        logger.warning("Failed to construct ObservatoryRecommender: %s", e)
        return None


def _load_variant_catalog(config: dict[str, Any]) -> Any:
    """Load VariantCatalog, or None if not available."""
    mod = _try_import("cohort_projections.analysis.observatory.variant_catalog")
    if mod is None:
        return None
    try:
        cls = mod.VariantCatalog
        catalog_path_rel = config.get("variant_catalog", "config/observatory_variants.yaml")
        catalog_path = PROJECT_ROOT / catalog_path_rel
        return cls(catalog_path)
    except Exception as e:
        logger.warning("Failed to load VariantCatalog: %s", e)
        return None


def _load_report_class() -> Any:
    """Load ObservatoryReport class, or None if not available."""
    try:
        from cohort_projections.analysis.observatory.report import ObservatoryReport

        return ObservatoryReport
    except ImportError as e:
        logger.error("Failed to import ObservatoryReport: %s", e)
        return None


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load the observatory config section from YAML."""
    try:
        from cohort_projections.analysis.observatory.results_store import (
            load_observatory_config,
        )

        return load_observatory_config(config_path)
    except Exception as e:
        logger.error("Failed to load observatory config from %s: %s", config_path, e)
        return {}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "observatory_config.yaml"


def _add_common_cli_options(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Attach shared global options to the main parser or a subparser."""
    default_config: Path | str = argparse.SUPPRESS if suppress_defaults else DEFAULT_CONFIG_PATH
    default_verbose: bool | str = argparse.SUPPRESS if suppress_defaults else False
    default_format: str = argparse.SUPPRESS if suppress_defaults else "table"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to observatory config YAML (default: config/observatory_config.yaml).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=default_verbose,
        help="Enable verbose (DEBUG) logging.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default=default_format,
        dest="output_format",
        help="Output format: table (default, human-readable), csv, or json (machine-readable).",
    )


def _resolve_project_path(path: Path | None) -> Path | None:
    """Resolve project-relative paths to absolute filesystem paths."""
    if path is None:
        return None
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _default_observatory_resume_file(prefix: str) -> Path:
    """Return the canonical Observatory checkpoint file for a queue."""
    return DEFAULT_SWEEP_STATE_DIR / f"{prefix}_resume.json"


def _slugify_fragment(value: Any) -> str:
    """Convert a value to a concise filesystem-safe slug fragment."""
    if isinstance(value, dict):
        parts = [
            f"{_slugify_fragment(key)}-{_slugify_fragment(subvalue)}"
            for key, subvalue in sorted(value.items(), key=lambda item: str(item[0]))
        ]
        raw = "-".join(part for part in parts if part)
    elif isinstance(value, (list, tuple, set)):
        raw = "-".join(_slugify_fragment(item) for item in value)
    else:
        raw = str(value).lower().replace(".", "p").replace(" ", "-").replace("_", "-")

    slug = "".join(c for c in raw if c.isalnum() or c in ("-", "_"))
    slug = re.sub(r"-{2,}", "-", slug).strip("-_")
    if not slug:
        return "value"
    if len(slug) <= 56:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:40].rstrip('-_')}-{digest}"


def _recommendation_config_delta(rec: Any) -> dict[str, Any] | None:
    """Build a config delta from a recommendation object."""
    suggested_value = getattr(rec, "suggested_value", None)
    parameter = getattr(rec, "parameter", "")
    if isinstance(suggested_value, dict):
        return dict(suggested_value)
    if suggested_value is None or not parameter:
        return None
    return {str(parameter): suggested_value}


def _recommendation_identifiers(rec: Any) -> tuple[str, str]:
    """Return normalized experiment and benchmark labels for a recommendation."""
    today = datetime.now(tz=UTC).date().strftime("%Y%m%d")
    config_delta = _recommendation_config_delta(rec) or {}
    if config_delta:
        fragments = [
            f"{_slugify_fragment(key)}-{_slugify_fragment(value)}"
            for key, value in sorted(config_delta.items(), key=lambda item: str(item[0]))
        ]
    else:
        fragments = ["recommendation"]
    slug = _slugify_fragment("-".join(fragments))
    return (f"exp-{today}-rec-{slug}", f"rec-{slug}")


def _build_recommendation_spec(
    rec: Any,
    config: dict[str, Any],
    catalog: Any | None = None,
) -> dict[str, Any] | None:
    """Build a canonical experiment spec for a config-only recommendation."""
    config_delta = _recommendation_config_delta(rec)
    if not config_delta:
        return None

    experiment_id, benchmark_label = _recommendation_identifiers(rec)
    priority = getattr(rec, "priority", None)
    notes = [
        "Generated from Observatory recommender.",
        f"Recommendation parameter: {getattr(rec, 'parameter', 'unknown')}",
    ]
    if isinstance(priority, int):
        notes.append(f"Recommendation priority: {priority}")
    grid_suggestion = getattr(rec, "grid_suggestion", None)
    if isinstance(grid_suggestion, dict):
        notes.append(f"Suggested follow-up grid: {json.dumps(grid_suggestion, sort_keys=True)}")

    base_method = getattr(
        catalog,
        "base_method",
        config.get("challenger_base_method", "m2026r1"),
    )
    base_config = getattr(
        catalog,
        "base_config",
        config.get("base_config", "cfg-20260309-college-fix-v1"),
    )

    return {
        "experiment_id": experiment_id,
        "hypothesis": str(getattr(rec, "rationale", "")).strip()[:200]
        or "Observatory recommendation generated from prior benchmark results.",
        "base_method": base_method,
        "base_config": base_config,
        "config_delta": config_delta,
        "scope": "county",
        "benchmark_label": benchmark_label,
        "requested_by": "agent",
        "notes": notes,
    }


def _print_queue_controls(
    *,
    queue_name: str,
    dry_run: bool,
    priority: str,
    run_budget: int | None,
    retry_failures: int,
    resume_file: Path | None,
) -> None:
    """Print the queue-control settings for an Observatory run command."""
    print(f"{queue_name} queue controls:")  # noqa: T201
    print(f"  Priority: {priority}")  # noqa: T201
    print(f"  Run budget: {run_budget if run_budget is not None else 'unbounded'}")  # noqa: T201
    print(f"  Retry failures: {retry_failures}")  # noqa: T201
    print("  Concurrency: sequential")  # noqa: T201
    if resume_file is not None:
        print(f"  Resume file: {resume_file}")  # noqa: T201
    if dry_run:
        print("  Mode: dry-run preview")  # noqa: T201


def _run_sweep_command(
    *,
    spec_paths: list[str],
    run_budget: int | None,
    retry_failures: int,
    resume_file: Path | None,
) -> int:
    """Delegate the prepared spec list to ``run_experiment_sweep.py``."""
    sweep_script = PROJECT_ROOT / "scripts" / "analysis" / "run_experiment_sweep.py"
    if not sweep_script.exists():
        print(f"Error: sweep script not found at {sweep_script}")  # noqa: T201
        return 1

    cmd = [sys.executable, str(sweep_script), "--specs", *spec_paths]
    if run_budget is not None:
        cmd.extend(["--run-budget", str(run_budget)])
    if retry_failures:
        cmd.extend(["--retry-failures", str(retry_failures)])
    if resume_file is not None:
        cmd.extend(["--resume-file", str(resume_file)])

    print(f"Running: {' '.join(cmd[:4])} ... ({len(spec_paths)} specs)")  # noqa: T201
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def _sort_pending_variants(
    variants: list[dict[str, Any]],
    priority: str,
) -> list[dict[str, Any]]:
    """Return runnable pending variants in the desired queue order."""
    if priority == "variant-id":
        return sorted(
            variants,
            key=lambda variant: str(variant.get("variant_id", "")),
        )
    return sorted(
        variants,
        key=lambda variant: (
            int(variant.get("tier", 999)),
            str(variant.get("variant_id", "")),
        ),
    )


def _sort_recommendations(
    recommendations: list[Any],
    priority: str,
) -> list[Any]:
    """Return recommendations in the desired queue order."""
    if priority == "parameter":
        return sorted(
            recommendations,
            key=lambda rec: str(getattr(rec, "parameter", "")),
        )
    return sorted(
        recommendations,
        key=lambda rec: (
            int(getattr(rec, "priority", 999)),
            str(getattr(rec, "parameter", "")),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with subcommands.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="observatory",
        description=(
            "Projection Observatory: unified CLI for benchmark results analysis, "
            "comparison, recommendation, and report generation."
        ),
    )
    _add_common_cli_options(parser)
    common = argparse.ArgumentParser(add_help=False)
    _add_common_cli_options(common, suppress_defaults=True)

    sub = parser.add_subparsers(dest="command", help="Subcommand to run.")

    # --- status ---
    sub.add_parser(
        "status",
        parents=[common],
        help="Display run inventory and variant catalog status.",
    )

    # --- compare ---
    compare_parser = sub.add_parser(
        "compare",
        parents=[common],
        help="Run full N-way comparison across all benchmark runs.",
    )
    compare_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top variants to display (default: 10).",
    )

    # --- rank ---
    rank_parser = sub.add_parser(
        "rank",
        parents=[common],
        help="Rank variants by a specific metric.",
    )
    rank_parser.add_argument(
        "metric",
        help="Metric name to rank by (e.g. county_mape_overall).",
    )
    rank_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top variants to display (default: 10).",
    )

    # --- recommend ---
    sub.add_parser(
        "recommend",
        parents=[common],
        help="Generate next-experiment recommendations.",
    )

    # --- run-pending ---
    pending_parser = sub.add_parser(
        "run-pending",
        parents=[common],
        help="Run all untested variants from the catalog.",
    )
    pending_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing experiments.",
    )
    pending_parser.add_argument(
        "--run-budget",
        type=int,
        default=None,
        help="Maximum number of runnable catalog variants to launch in this invocation.",
    )
    pending_parser.add_argument(
        "--retry-failures",
        type=int,
        default=0,
        help="Number of extra attempts allowed for prior failed entries in the resume file.",
    )
    pending_parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="Checkpoint file for resumable bounded queue runs (project-relative or absolute).",
    )
    pending_parser.add_argument(
        "--priority",
        choices=["tier", "variant-id"],
        default="tier",
        help="Ordering strategy for runnable variants before queue submission.",
    )

    # --- report ---
    report_parser = sub.add_parser(
        "report",
        parents=[common],
        help="Generate full HTML observatory report.",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the HTML report (default: auto-generated in data/analysis/observatory/).",
    )

    # --- diff ---
    diff_parser = sub.add_parser(
        "diff",
        parents=[common],
        help="Head-to-head comparison of two runs (by run_id or config_id).",
    )
    diff_parser.add_argument(
        "run_a",
        help="First run or config identifier.",
    )
    diff_parser.add_argument(
        "run_b",
        help="Second run or config identifier.",
    )

    # --- history ---
    history_parser = sub.add_parser(
        "history",
        parents=[common],
        help="Show chronological progression of experiments with key metrics.",
    )
    history_parser.add_argument(
        "--metric",
        default="county_mape_overall",
        help="Metric to track across experiments (default: county_mape_overall).",
    )

    # --- run-recommended ---
    recommended_parser = sub.add_parser(
        "run-recommended",
        parents=[common],
        help="Run config-only recommendations from the recommender.",
    )
    recommended_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing experiments.",
    )
    recommended_parser.add_argument(
        "--run-budget",
        type=int,
        default=None,
        help="Maximum number of recommendations to launch in this invocation.",
    )
    recommended_parser.add_argument(
        "--retry-failures",
        type=int,
        default=0,
        help="Number of extra attempts allowed for prior failed entries in the resume file.",
    )
    recommended_parser.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        help="Checkpoint file for resumable recommendation queues (project-relative or absolute).",
    )
    recommended_parser.add_argument(
        "--priority",
        choices=["recommendation", "parameter"],
        default="recommendation",
        help="Ordering strategy for config-only recommendations before queue submission.",
    )

    # --- refresh ---
    sub.add_parser(
        "refresh",
        parents=[common],
        help="Rebuild the results cache from benchmark history.",
    )

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_status(
    store: Any,
    config: dict[str, Any],
    _args: argparse.Namespace,
) -> int:
    """Handle the ``status`` subcommand.

    Prints run inventory and variant catalog summary.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    fmt = getattr(_args, "output_format", "table")

    if store is None:
        if fmt != "table":
            print(json.dumps({"error": "ResultsStore unavailable"}, indent=2))  # noqa: T201
        else:
            print("Projection Observatory Status")  # noqa: T201
            print("=" * 35)  # noqa: T201
            print("  (ResultsStore unavailable)")  # noqa: T201
        return 1

    # Collect status data into a dict for structured output
    status_data: dict[str, Any] = {}

    # Run inventory
    try:
        index = store.get_index()
        run_ids = store.get_run_ids()
        n_runs = len(run_ids)
    except Exception as e:
        if fmt == "table":
            print(f"  Error loading index: {e}")  # noqa: T201
        n_runs = 0
        index = None

    status_data["completed_runs"] = n_runs

    date_range: list[str] = []
    methods: list[str] = []

    if index is not None and not index.empty:
        for date_col in ("run_date", "created_at_utc"):
            if date_col in index.columns:
                dates = index[date_col].dropna()
                if not dates.empty:
                    date_range = [str(dates.min())[:10], str(dates.max())[:10]]
                break

        for method_col in ("method", "method_id", "challenger_method_id"):
            if method_col in index.columns:
                methods = sorted(str(m) for m in index[method_col].dropna().unique().tolist())
                break

    status_data["date_range"] = date_range
    status_data["methods"] = methods

    # Champion and best challenger
    primary_metric = config.get("comparison", {}).get("primary_metric", "county_mape_overall")
    champ_info = _get_champion_challenger(store, primary_metric)
    status_data["champion"] = champ_info.get("champion", {})
    status_data["best_challenger"] = champ_info.get("best_challenger", {})

    # Variant catalog
    catalog = _load_variant_catalog(config)
    catalog_data: dict[str, Any] = {}
    if catalog is not None:
        try:
            catalog_data = catalog.get_inventory_summary()
        except Exception as e:
            catalog_data["error"] = str(e)

    status_data["variant_catalog"] = catalog_data

    # --- Output ---
    if fmt == "json":
        print(json.dumps(status_data, indent=2, default=str))  # noqa: T201
        return 0

    if fmt == "csv":
        # Flatten status_data into a single-row CSV
        flat: dict[str, Any] = {
            "completed_runs": status_data["completed_runs"],
            "date_min": date_range[0] if date_range else "",
            "date_max": date_range[1] if len(date_range) > 1 else "",
            "methods": ";".join(methods),
            "champion_config_id": status_data["champion"].get("config_id", ""),
            "champion_metric": status_data["champion"].get("metric", ""),
            "best_challenger_config_id": status_data["best_challenger"].get("config_id", ""),
            "best_challenger_delta": status_data["best_challenger"].get("delta", ""),
            "catalog_total": catalog_data.get("total", ""),
            "catalog_tested": catalog_data.get("tested", ""),
            "catalog_untested_total": catalog_data.get("untested_total", ""),
            "catalog_untested_runnable": catalog_data.get("untested_runnable", ""),
            "catalog_untested_requires_code_change": catalog_data.get(
                "untested_requires_code_change", ""
            ),
            "catalog_untested_ids": ";".join(catalog_data.get("untested_ids", [])),
            "catalog_untested_runnable_ids": ";".join(
                catalog_data.get("untested_runnable_ids", [])
            ),
            "catalog_untested_requires_code_change_ids": ";".join(
                catalog_data.get("untested_requires_code_change_ids", [])
            ),
            "grid_total": catalog_data.get("grid_total", ""),
            "grid_runnable": catalog_data.get("grid_runnable", ""),
            "grid_blocked": catalog_data.get("grid_blocked", ""),
            "grid_blocked_ids": ";".join(catalog_data.get("grid_blocked_ids", [])),
        }
        writer = csv.DictWriter(sys.stdout, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
        return 0

    # table format — original behavior
    print(_c_bold("Projection Observatory Status"))  # noqa: T201
    print("=" * 35)  # noqa: T201
    print(f"Completed runs: {n_runs}")  # noqa: T201

    if date_range:
        print(f"Date range: {date_range[0]} to {date_range[1]}")  # noqa: T201
    if methods:
        print(f"Methods tested: {', '.join(methods)}")  # noqa: T201

    _print_champion_challenger(store, primary_metric)
    _print_experiment_log_summary(store)

    print()  # noqa: T201
    print(_c_bold("Variant Catalog:"))  # noqa: T201

    if catalog is None:
        print("  (VariantCatalog not available)")  # noqa: T201
    elif "error" in catalog_data:
        print(f"  Error reading catalog: {catalog_data['error']}")  # noqa: T201
    else:
        print(f"  Total variants: {catalog_data.get('total', 0)}")  # noqa: T201
        print(f"  Tested: {catalog_data.get('tested', 0)}")  # noqa: T201
        untested_count = catalog_data.get("untested_total", 0)
        untested_ids_list = catalog_data.get("untested_ids", [])
        if untested_ids_list:
            print(f"  Untested: {untested_count} — {', '.join(untested_ids_list)}")  # noqa: T201
        else:
            print(f"  Untested: {untested_count}")  # noqa: T201
        print(f"  Untested runnable: {catalog_data.get('untested_runnable', 0)}")  # noqa: T201
        blocked_ids = catalog_data.get("untested_requires_code_change_ids", [])
        blocked_count = catalog_data.get("untested_requires_code_change", 0)
        if blocked_ids:
            print(f"  Untested requiring code changes: {blocked_count} — {', '.join(blocked_ids)}")  # noqa: T201
        else:
            print(f"  Untested requiring code changes: {blocked_count}")  # noqa: T201

        print(f"  Defined grids: {catalog_data.get('grid_total', 0)}")  # noqa: T201
        blocked_grids = catalog_data.get("grid_blocked_ids", [])
        if blocked_grids:
            print(
                "  Blocked grids: "
                f"{catalog_data.get('grid_blocked', len(blocked_grids))} — "
                f"{', '.join(blocked_grids)}"
            )  # noqa: T201
        else:
            print("  Blocked grids: 0")  # noqa: T201

    return 0


def cmd_compare(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``compare`` subcommand.

    Runs a full N-way comparison and prints the console report.

    Returns
    -------
    int
        Exit code.
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    comparator = _load_comparator(store, config)
    if comparator is None:
        print("Error: ObservatoryComparator is not available.")  # noqa: T201
        return 1

    try:
        result = comparator.full_comparison()
    except Exception as e:
        print(f"Error running comparison: {e}")  # noqa: T201
        return 1

    bounds_catalog = _load_variant_catalog(config)
    recommender = _load_recommender(
        store, config, comparator=comparator, bounds_catalog=bounds_catalog
    )
    recommendations: list[Any] = []
    if recommender is not None:
        try:
            recommendations = recommender.suggest_next_experiments()
        except Exception as e:
            logger.warning("Recommender failed: %s", e)

    fmt = getattr(args, "output_format", "table")

    if fmt in ("csv", "json"):
        # Extract ranking DataFrame from comparison result
        ranking_df = _safe_attr(result, "ranking")
        if ranking_df is None:
            ranking_df = _safe_attr(result, "rankings")
        if ranking_df is None:
            ranking_df = _safe_attr(result, "summary")
        if ranking_df is not None and hasattr(ranking_df, "to_csv"):
            if fmt == "csv":
                ranking_df.to_csv(sys.stdout, index=False)
            else:
                print(
                    json.dumps(  # noqa: T201
                        json.loads(ranking_df.to_json(orient="records", default_handler=str)),
                        indent=2,
                    )
                )
        else:
            # Fallback: serialize the result as-is
            if fmt == "json":
                print(
                    json.dumps(
                        {"error": "No ranking DataFrame available in comparison result"}, indent=2
                    )
                )  # noqa: T201
            else:
                print("No ranking DataFrame available in comparison result")  # noqa: T201
        return 0

    report_class = _load_report_class()
    if report_class is None:
        # Fallback to comparator's own formatting
        print(comparator.format_comparison_summary(result))  # noqa: T201
        return 0

    report = report_class(
        comparator_result=result,
        recommendations=recommendations,
        store=store,
    )
    print(report.generate_console_report())  # noqa: T201
    return 0


def cmd_rank(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``rank`` subcommand.

    Ranks variants by a specified metric.

    Returns
    -------
    int
        Exit code.
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    comparator = _load_comparator(store, config)
    if comparator is None:
        print("Error: ObservatoryComparator is not available.")  # noqa: T201
        print("  This module has not been implemented yet.")  # noqa: T201
        return 1

    try:
        result = comparator.rank_by(metric=args.metric)
    except Exception as e:
        print(f"Error ranking by {args.metric}: {e}")  # noqa: T201
        return 1

    fmt = getattr(args, "output_format", "table")

    if result.empty:
        if fmt == "json":
            print(json.dumps([], indent=2))  # noqa: T201
        elif fmt == "csv":
            print()  # noqa: T201
        else:
            print("No results to rank.")  # noqa: T201
        return 0

    display = result.head(args.top)

    if fmt in ("csv", "json"):
        if fmt == "csv":
            display.to_csv(sys.stdout, index=False)
        else:
            print(
                json.dumps(  # noqa: T201
                    json.loads(display.to_json(orient="records", default_handler=str)),
                    indent=2,
                )
            )
        return 0

    print(_c_bold(f"Ranking by {args.metric} (top {args.top}):"))  # noqa: T201
    print("-" * 72)  # noqa: T201
    # Prefer config_id as primary label; include run_id only as secondary
    id_cols = [c for c in ("config_id", "method_id") if c in display.columns]
    if not id_cols:
        id_cols = [c for c in ("run_id",) if c in display.columns]
    show_cols = id_cols + [args.metric, "rank"]
    show_cols = [c for c in show_cols if c in display.columns]
    print(display[show_cols].to_string(index=False))  # noqa: T201
    return 0


def cmd_recommend(
    store: Any,
    config: dict[str, Any],
    _args: argparse.Namespace,
) -> int:
    """Handle the ``recommend`` subcommand.

    Generates and prints next-experiment recommendations.

    Returns
    -------
    int
        Exit code.
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    comparator = _load_comparator(store, config)
    bounds_catalog = _load_variant_catalog(config)
    recommender = _load_recommender(
        store, config, comparator=comparator, bounds_catalog=bounds_catalog
    )
    if recommender is None:
        print("Error: ObservatoryRecommender is not available.")  # noqa: T201
        return 1

    try:
        recommendations = recommender.suggest_next_experiments()
    except Exception as e:
        print(f"Error generating recommendations: {e}")  # noqa: T201
        return 1

    fmt = getattr(_args, "output_format", "table")

    if not recommendations:
        if fmt == "json":
            print(json.dumps([], indent=2))  # noqa: T201
        elif fmt == "csv":
            print()  # noqa: T201
        else:
            print("No recommendations generated.")  # noqa: T201
        return 0

    if fmt in ("csv", "json"):
        rec_dicts = []
        for rec in recommendations:
            if hasattr(rec, "__dataclass_fields__"):
                import dataclasses

                d = dataclasses.asdict(rec)
            elif hasattr(rec, "__dict__"):
                d = dict(rec.__dict__)
            elif isinstance(rec, dict):
                d = dict(rec)
            else:
                d = {"value": str(rec)}
            rec_dicts.append(d)

        if fmt == "json":
            print(json.dumps(rec_dicts, indent=2, default=str))  # noqa: T201
        else:
            # CSV: flatten grid_suggestion dict to string
            import pandas as pd

            df = pd.DataFrame(rec_dicts)
            if "grid_suggestion" in df.columns:
                df["grid_suggestion"] = df["grid_suggestion"].apply(
                    lambda x: json.dumps(x, default=str) if x is not None else ""
                )
            df.to_csv(sys.stdout, index=False)
        return 0

    print(recommender.format_recommendations(recommendations))  # noqa: T201
    return 0


def cmd_run_pending(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``run-pending`` subcommand.

    Loads the variant catalog, generates spec files for untested variants,
    and delegates to ``run_experiment_sweep.py``.

    Returns
    -------
    int
        Exit code.
    """
    if args.run_budget is not None and args.run_budget <= 0:
        print("Error: --run-budget must be positive when provided.")  # noqa: T201
        return 1
    if args.retry_failures < 0:
        print("Error: --retry-failures must be zero or greater.")  # noqa: T201
        return 1

    catalog = _load_variant_catalog(config)
    if catalog is None:
        print("Error: VariantCatalog is not available.")  # noqa: T201
        print("  This module has not been implemented yet.")  # noqa: T201
        return 1

    # Get untested variants
    try:
        untested = catalog.get_untested()
        inventory = catalog.get_inventory_summary()
    except Exception as e:
        print(f"Error loading untested variants: {e}")  # noqa: T201
        return 1

    if not untested:
        print(
            "No untested runnable variants found. All config-only catalog entries "
            "have either been tested already or require code changes."
        )  # noqa: T201
        return 0

    ordered = _sort_pending_variants(untested, priority=args.priority)
    resume_file = _resolve_project_path(args.resume_file) or _default_observatory_resume_file(
        "observatory_pending"
    )

    print(f"Found {len(ordered)} untested runnable variant(s).")  # noqa: T201
    print(
        "Blocked from unattended queueing: "
        f"{inventory.get('untested_requires_code_change', 0)} variant(s), "
        f"{inventory.get('grid_blocked', 0)} grid(s)."
    )  # noqa: T201
    _print_queue_controls(
        queue_name="Pending",
        dry_run=args.dry_run,
        priority=args.priority,
        run_budget=args.run_budget,
        retry_failures=args.retry_failures,
        resume_file=resume_file,
    )

    with tempfile.TemporaryDirectory(prefix="observatory_pending_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        spec_paths: list[str] = []

        for variant in ordered:
            vid = variant.get("variant_id")
            if vid is None:
                continue
            try:
                spec_path = catalog.generate_spec(
                    vid,
                    output_dir=tmpdir_path,
                    requested_by="agent",
                    extra_notes=["Generated from Observatory run-pending queue."],
                )
                spec_paths.append(str(spec_path))
            except Exception as e:
                logger.warning("Failed to generate spec for %s: %s", vid, e)

        if not spec_paths:
            print("No valid specs generated from untested variants.")  # noqa: T201
            return 0

        print(f"Generated {len(spec_paths)} experiment spec(s).")  # noqa: T201

        if args.dry_run:
            print("Dry run -- would execute:")  # noqa: T201
            preview_paths = spec_paths[: args.run_budget] if args.run_budget else spec_paths
            for sp in preview_paths:
                print(f"  {sp}")  # noqa: T201
            if args.run_budget is not None and len(spec_paths) > len(preview_paths):
                print(f"  ... {len(spec_paths) - len(preview_paths)} more deferred by run budget")  # noqa: T201
            return 0

        return _run_sweep_command(
            spec_paths=spec_paths,
            run_budget=args.run_budget,
            retry_failures=args.retry_failures,
            resume_file=resume_file,
        )


def cmd_run_recommended(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``run-recommended`` subcommand.

    Gets recommendations from the recommender, filters to config-only ones,
    generates spec files via the variant catalog, and delegates to
    ``run_experiment_sweep.py``.

    Returns
    -------
    int
        Exit code.
    """
    if args.run_budget is not None and args.run_budget <= 0:
        print("Error: --run-budget must be positive when provided.")  # noqa: T201
        return 1
    if args.retry_failures < 0:
        print("Error: --retry-failures must be zero or greater.")  # noqa: T201
        return 1

    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    comparator = _load_comparator(store, config)
    bounds_catalog = _load_variant_catalog(config)
    recommender = _load_recommender(
        store, config, comparator=comparator, bounds_catalog=bounds_catalog
    )
    if recommender is None:
        print("Error: ObservatoryRecommender is not available.")  # noqa: T201
        return 1

    catalog = _load_variant_catalog(config)

    try:
        recommendations = recommender.suggest_next_experiments(n=20)
    except Exception as e:
        print(f"Error generating recommendations: {e}")  # noqa: T201
        return 1

    # Filter to config-only recommendations
    config_only_recs = [r for r in recommendations if not r.requires_code_change]
    if not config_only_recs:
        print("No config-only recommendations available.")  # noqa: T201
        return 0

    ordered_recs = _sort_recommendations(config_only_recs, priority=args.priority)
    resume_file = _resolve_project_path(args.resume_file) or _default_observatory_resume_file(
        "observatory_recommended"
    )

    print(f"Found {len(ordered_recs)} config-only recommendation(s).")  # noqa: T201
    _print_queue_controls(
        queue_name="Recommended",
        dry_run=args.dry_run,
        priority=args.priority,
        run_budget=args.run_budget,
        retry_failures=args.retry_failures,
        resume_file=resume_file,
    )

    with tempfile.TemporaryDirectory(prefix="observatory_recommended_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        spec_paths: list[str] = []

        variants_df = None
        if catalog is not None:
            try:
                variants_df = catalog.list_variants()
            except Exception:
                variants_df = None

        for rec in ordered_recs:
            try:
                config_delta = _recommendation_config_delta(rec)
                if not config_delta:
                    continue

                # Try to use catalog.generate_spec if the recommendation
                # matches a catalog variant
                generated = False
                if catalog is not None and variants_df is not None:
                    try:
                        for _, row in variants_df.iterrows():
                            if (
                                row["parameter"] == rec.parameter
                                and row["value"] == rec.suggested_value
                                and not row["tested"]
                            ):
                                sp = catalog.generate_spec(
                                    row["variant_id"],
                                    output_dir=tmpdir_path,
                                    requested_by="agent",
                                    extra_notes=[
                                        "Generated from Observatory recommendation queue.",
                                    ],
                                )
                                spec_paths.append(str(sp))
                                generated = True
                                break
                    except Exception:
                        pass

                if not generated:
                    spec = _build_recommendation_spec(rec, config, catalog=catalog)
                    if spec is None:
                        continue
                    spec_path = tmpdir_path / f"{spec['experiment_id']}.yaml"
                    spec_path.write_text(
                        yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                        encoding="utf-8",
                    )
                    spec_paths.append(str(spec_path))

            except Exception as e:
                logger.warning(
                    "Failed to generate spec for recommendation %s=%s: %s",
                    rec.parameter,
                    rec.suggested_value,
                    e,
                )

        if not spec_paths:
            print("No valid specs generated from recommendations.")  # noqa: T201
            return 0

        print(f"Generated {len(spec_paths)} experiment spec(s).")  # noqa: T201

        if args.dry_run:
            print("Dry run -- would execute:")  # noqa: T201
            preview_paths = spec_paths[: args.run_budget] if args.run_budget else spec_paths
            for sp in preview_paths:
                print(f"  {sp}")  # noqa: T201
            if args.run_budget is not None and len(spec_paths) > len(preview_paths):
                print(f"  ... {len(spec_paths) - len(preview_paths)} more deferred by run budget")  # noqa: T201
            return 0

        return _run_sweep_command(
            spec_paths=spec_paths,
            run_budget=args.run_budget,
            retry_failures=args.retry_failures,
            resume_file=resume_file,
        )


def cmd_report(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``report`` subcommand.

    Generates a full HTML observatory report.

    Returns
    -------
    int
        Exit code.
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    report_class = _load_report_class()
    if report_class is None:
        print("Error: ObservatoryReport is not available.")  # noqa: T201
        return 1

    # Run comparison if available
    comparator_result = None
    comparator = _load_comparator(store, config)
    if comparator is not None:
        try:
            comparator_result = comparator.full_comparison()
        except Exception as e:
            logger.warning("Comparison failed, report will have limited content: %s", e)

    # Get recommendations if available
    recommendations: list[Any] = []
    bounds_catalog = _load_variant_catalog(config)
    recommender = _load_recommender(
        store, config, comparator=comparator, bounds_catalog=bounds_catalog
    )
    if recommender is not None:
        try:
            recommendations = recommender.suggest_next_experiments()
        except Exception as e:
            logger.warning("Recommender failed: %s", e)

    report = report_class(
        comparator_result=comparator_result,
        recommendations=recommendations,
        store=store,
    )

    output_target = _resolve_project_path(args.output)
    output_path = report.generate_html_report(output_path=output_target)
    print(f"Observatory report written to: {output_path}")  # noqa: T201

    # Also print the summary to console
    print()  # noqa: T201
    print(report.generate_summary())  # noqa: T201

    return 0


def cmd_refresh(
    store: Any,
    config: dict[str, Any],
    _args: argparse.Namespace,
) -> int:
    """Handle the ``refresh`` subcommand.

    Clears and rebuilds the results cache.

    Returns
    -------
    int
        Exit code.
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    print("Clearing existing cache...")  # noqa: T201
    try:
        store.clear_cache()
    except Exception as e:
        logger.warning("Cache clear failed: %s", e)

    print("Refreshing results store...")  # noqa: T201
    try:
        store.refresh()
    except Exception as e:
        print(f"Error refreshing store: {e}")  # noqa: T201
        return 1

    # Force-load all consolidated frames to populate cache
    print("Loading consolidated metrics...")  # noqa: T201
    loaded = 0
    for getter_name in (
        "get_consolidated_scorecards",
        "get_consolidated_county_metrics",
        "get_consolidated_state_metrics",
        "get_consolidated_projection_curves",
        "get_consolidated_sensitivity_summary",
    ):
        try:
            getter = getattr(store, getter_name)
            df = getter()
            n = len(df) if df is not None else 0
            print(f"  {getter_name}: {n} rows")  # noqa: T201
            loaded += 1
        except Exception as e:
            logger.warning("Failed to load %s: %s", getter_name, e)

    # Write cache
    print("Writing cache...")  # noqa: T201
    try:
        store.write_cache()
    except Exception as e:
        logger.warning("Cache write failed: %s", e)

    n_runs = len(store.get_run_ids())
    print(f"Refresh complete: {n_runs} run(s), {loaded} dataset(s) loaded.")  # noqa: T201
    return 0


def cmd_history(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``history`` subcommand.

    Shows all experiments in chronological order with a tracked metric,
    delta from previous row, and trend arrow so the user can see whether
    experiments are converging or stalling.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    metric = args.metric

    try:
        scorecards = store.get_consolidated_scorecards()
    except Exception as e:
        print(f"Error loading scorecards: {e}")  # noqa: T201
        return 1

    if scorecards.empty:
        print("No scorecards available.")  # noqa: T201
        return 0

    if metric not in scorecards.columns:
        print(f"Error: Metric '{metric}' not found in scorecards.")  # noqa: T201
        available = [
            c
            for c in scorecards.columns
            if c not in ("run_id", "config_id", "method_id", "status_at_run")
        ]
        if available:
            print(f"Available metrics: {', '.join(sorted(available))}")  # noqa: T201
        return 1

    # Deduplicate by config_id: keep the row with the latest run_id
    # (run_ids are timestamp-based, so lexicographic sort works)
    if "config_id" not in scorecards.columns:
        print("Error: scorecards missing 'config_id' column.")  # noqa: T201
        return 1

    # Sort by run_id so that .last() picks the latest run per config
    scorecards_sorted = scorecards.sort_values("run_id")
    deduped = scorecards_sorted.groupby("config_id", sort=False).last().reset_index()

    # Sort chronologically by run_id (timestamp-based)
    deduped = deduped.sort_values("run_id").reset_index(drop=True)

    # Extract date from run_id (format: br-YYYYMMDD-HHMMSS-...)
    def _extract_date(run_id: str) -> str:
        """Extract YYYY-MM-DD date from a run_id like br-YYYYMMDD-HHMMSS-..."""
        parts = str(run_id).split("-")
        if len(parts) >= 2 and len(parts[1]) == 8:
            d = parts[1]
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return str(run_id)[:10]

    # Build history rows
    history_rows: list[dict[str, Any]] = []
    prev_val: float | None = None
    best_val: float | None = None
    best_config: str = ""

    for i, (_, row) in enumerate(deduped.iterrows(), start=1):
        config_id = str(row["config_id"])
        val = float(row[metric])
        run_id = str(row["run_id"])
        date_str = _extract_date(run_id)

        delta = val - prev_val if prev_val is not None else None

        # Trend arrow: lower is better for error metrics
        if delta is None:
            trend = ""
        elif abs(delta) < 0.005:
            trend = "\u2192"  # → flat
        elif delta < 0:
            trend = "\u2193"  # ↓ better (lower error)
        else:
            trend = "\u2191"  # ↑ worse (higher error)

        if best_val is None or val < best_val:
            best_val = val
            best_config = config_id

        history_rows.append(
            {
                "seq": i,
                "date": date_str,
                "config_id": config_id,
                "value": val,
                "delta": delta,
                "trend": trend,
            }
        )
        prev_val = val

    # --- Output ---
    fmt = getattr(args, "output_format", "table")

    if fmt == "json":
        payload: dict[str, Any] = {
            "metric": metric,
            "history": history_rows,
            "best_config": best_config,
            "best_value": best_val,
            "improvement_range": round((history_rows[0]["value"] - best_val), 3)
            if best_val is not None and history_rows
            else None,
        }
        print(json.dumps(payload, indent=2, default=str))  # noqa: T201
        return 0

    if fmt == "csv":
        fieldnames = ["seq", "date", "config_id", metric, "delta", "trend"]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row_data in history_rows:
            writer.writerow(
                {
                    "seq": row_data["seq"],
                    "date": row_data["date"],
                    "config_id": row_data["config_id"],
                    metric: f"{row_data['value']:.3f}",
                    "delta": f"{row_data['delta']:+.3f}" if row_data["delta"] is not None else "",
                    "trend": row_data["trend"],
                }
            )
        return 0

    # --- table format ---
    print(_c_bold(f"Experiment History ({metric})"))  # noqa: T201
    print("=" * 40)  # noqa: T201
    print()  # noqa: T201

    # Compute column widths
    config_w = max(30, *(len(r["config_id"]) for r in history_rows))
    # Truncate config_id display to keep table reasonable
    config_w = min(config_w, 40)

    hdr = (
        f" {'#':>3}  {'Date':<10}  {'Config':<{config_w}}  "
        f"{'Overall':>9}  {'Delta':>8}  {'Trend':>5}"
    )
    print(hdr)  # noqa: T201
    print("\u2500" * len(hdr))  # noqa: T201

    for row_data in history_rows:
        config_display = row_data["config_id"]
        if len(config_display) > config_w:
            config_display = config_display[: config_w - 1] + "\u2026"

        if row_data["delta"] is not None:
            delta_raw = f"{row_data['delta']:+.3f}"
            delta_colored = _c_delta(row_data["delta"])
            # Right-pad to 8 visible chars, then apply color
            delta_str = " " * (8 - len(delta_raw)) + delta_colored
        else:
            delta_str = "\u2014".rjust(8)

        trend_raw = row_data["trend"] if row_data["trend"] else "\u2014"
        if trend_raw == "\u2193":  # ↓ better
            trend_str = " " * 4 + _c_green(trend_raw)
        elif trend_raw == "\u2191":  # ↑ worse
            trend_str = " " * 4 + _c_red(trend_raw)
        elif trend_raw == "\u2192":  # → flat
            trend_str = " " * 4 + _c_yellow(trend_raw)
        else:
            trend_str = f"{trend_raw:>5}"

        print(  # noqa: T201
            f" {row_data['seq']:>3}  {row_data['date']:<10}  "
            f"{config_display:<{config_w}}  "
            f"{row_data['value']:>9.3f}  {delta_str}  {trend_str}"
        )

    print()  # noqa: T201

    if best_val is not None and history_rows:
        first_val = history_rows[0]["value"]
        improvement = first_val - best_val
        print(f"Best: {_c_bold(best_config)} ({best_val:.3f})")  # noqa: T201
        imp_str = _c_green(f"{improvement:.3f}") if improvement > 0 else f"{improvement:.3f}"
        print(f"Improvement range: {imp_str}pp from first to best")  # noqa: T201

    return 0


def cmd_diff(
    store: Any,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Handle the ``diff`` subcommand.

    Displays a head-to-head comparison of two benchmark runs, identified by
    run_id or config_id.  Outputs metrics side-by-side with deltas and
    optionally shows config differences between the two runs.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    if store is None:
        print("Error: ResultsStore unavailable.")  # noqa: T201
        return 1

    try:
        scorecards = store.get_consolidated_scorecards()
    except Exception as e:
        print(f"Error loading scorecards: {e}")  # noqa: T201
        return 1

    if scorecards.empty:
        print("Error: No scorecards available.")  # noqa: T201
        return 1

    id_a = args.run_a
    id_b = args.run_b

    row_a = _resolve_scorecard_row(scorecards, id_a)
    row_b = _resolve_scorecard_row(scorecards, id_b)

    if row_a is None:
        print(f"Error: No scorecard found for '{id_a}'.")  # noqa: T201
        return 1
    if row_b is None:
        print(f"Error: No scorecard found for '{id_b}'.")  # noqa: T201
        return 1

    # Determine labels (prefer config_id, fall back to run_id)
    label_a = str(row_a.get("config_id", row_a.get("run_id", id_a)))
    label_b = str(row_b.get("config_id", row_b.get("run_id", id_b)))
    run_id_a = str(row_a.get("run_id", id_a))
    run_id_b = str(row_b.get("run_id", id_b))

    # Collect metrics from comparator module constants
    from cohort_projections.analysis.observatory.comparator import (
        METRIC_COLUMNS,
        SENTINEL_COLUMNS,
    )

    all_metrics = METRIC_COLUMNS + SENTINEL_COLUMNS
    available_metrics = [m for m in all_metrics if m in row_a.index and m in row_b.index]

    # Build rows: (metric_name, val_a, val_b, delta)
    metric_rows: list[dict[str, Any]] = []
    for metric in available_metrics:
        val_a = float(row_a[metric])
        val_b = float(row_b[metric])
        delta = val_a - val_b
        metric_rows.append({"metric": metric, "a": val_a, "b": val_b, "delta": delta})

    # Load config differences
    config_diffs: dict[str, tuple[Any, Any]] = {}
    try:
        cfg_a = store.get_run_config(run_id_a)
        cfg_b = store.get_run_config(run_id_b)
        config_diffs = _compute_config_diffs(cfg_a, cfg_b)
    except Exception:
        pass

    fmt = getattr(args, "output_format", "table")

    if fmt == "json":
        payload: dict[str, Any] = {
            "run_a": {"id": id_a, "label": label_a, "run_id": run_id_a},
            "run_b": {"id": id_b, "label": label_b, "run_id": run_id_b},
            "metrics": metric_rows,
            "config_diffs": {k: {"a": v[0], "b": v[1]} for k, v in config_diffs.items()},
        }
        print(json.dumps(payload, indent=2, default=str))  # noqa: T201
        return 0

    if fmt == "csv":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=["metric", label_a, label_b, "delta"],
        )
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(
                {
                    "metric": row["metric"],
                    label_a: f"{row['a']:.3f}",
                    label_b: f"{row['b']:.3f}",
                    "delta": f"{row['delta']:+.3f}",
                }
            )
        return 0

    # --- table format ---
    # Shorten labels for display (use last segment after last dash-group)
    short_a = _short_label(label_a)
    short_b = _short_label(label_b)

    print(_c_bold(f"Head-to-Head: {label_a} vs {label_b}"))  # noqa: T201
    print("=" * 72)  # noqa: T201
    print()  # noqa: T201

    # Compute column widths
    metric_w = max(len("Metric"), *(len(r["metric"]) for r in metric_rows)) + 2
    val_w = max(len(short_a), len(short_b), 10) + 2
    delta_w = 10

    # Header
    hdr = f"{'Metric':<{metric_w}}{short_a:>{val_w}}{short_b:>{val_w}}{'Delta':>{delta_w}}"
    print(hdr)  # noqa: T201
    print("\u2500" * len(hdr))  # noqa: T201

    # Split main metrics and sentinel metrics
    main_metrics = [r for r in metric_rows if not r["metric"].startswith("sentinel_")]
    sentinel_metrics = [r for r in metric_rows if r["metric"].startswith("sentinel_")]

    for row in main_metrics:
        delta_raw = f"{row['delta']:+.3f}"
        delta_colored = _c_delta(row["delta"])
        delta_padded = " " * (delta_w - len(delta_raw)) + delta_colored
        print(  # noqa: T201
            f"{row['metric']:<{metric_w}}{row['a']:>{val_w}.3f}{row['b']:>{val_w}.3f}{delta_padded}"
        )

    if sentinel_metrics:
        print()  # noqa: T201
        print(_c_bold("Sentinel Counties:"))  # noqa: T201
        for row in sentinel_metrics:
            delta_raw = f"{row['delta']:+.3f}"
            delta_colored = _c_delta(row["delta"])
            delta_padded = " " * (delta_w - len(delta_raw)) + delta_colored
            print(  # noqa: T201
                f"{row['metric']:<{metric_w}}"
                f"{row['a']:>{val_w}.3f}"
                f"{row['b']:>{val_w}.3f}"
                f"{delta_padded}"
            )

    if config_diffs:
        print()  # noqa: T201
        print(_c_bold("Config Differences:"))  # noqa: T201
        for key, (va, vb) in sorted(config_diffs.items()):
            print(f"  {key}: {va} \u2192 {vb}")  # noqa: T201

    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_scorecard_row(
    scorecards: pd.DataFrame,
    identifier: str,
) -> pd.Series | None:
    """Resolve an identifier (run_id or config_id) to a scorecard row.

    If *identifier* matches a ``run_id`` exactly, that row is returned.
    Otherwise, if it matches a ``config_id``, the row from the latest
    (last-occurring) run with that config is returned.  Returns ``None``
    when no match is found.
    """
    # Try run_id first
    if "run_id" in scorecards.columns:
        match = scorecards[scorecards["run_id"] == identifier]
        if not match.empty:
            return match.iloc[-1]

    # Try config_id
    if "config_id" in scorecards.columns:
        match = scorecards[scorecards["config_id"] == identifier]
        if not match.empty:
            return match.iloc[-1]

    return None


def _compute_config_diffs(
    cfg_a: dict[str, Any],
    cfg_b: dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Compare two run config dicts and return keys that differ.

    Both *cfg_a* and *cfg_b* are ``{method_id: config_dict}`` mappings.
    For simplicity, we flatten all method configs into a single namespace
    and report any keys whose values differ.

    Returns a dict mapping ``key`` to ``(value_a, value_b)``.
    """
    flat_a = _flatten_config(cfg_a)
    flat_b = _flatten_config(cfg_b)

    diffs: dict[str, tuple[Any, Any]] = {}
    all_keys = sorted(set(flat_a) | set(flat_b))
    for key in all_keys:
        va = flat_a.get(key)
        vb = flat_b.get(key)
        if va != vb:
            diffs[key] = (va, vb)
    return diffs


def _flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested ``{method_id: {param: value}}`` config to a flat dict.

    If the top-level values are dicts, the inner keys are used directly
    (collisions resolved by last-write-wins).  If the top-level values are
    scalars, they are kept as-is.
    """
    flat: dict[str, Any] = {}
    for _method_id, value in cfg.items():
        if isinstance(value, dict):
            for k, v in value.items():
                flat[k] = v
        else:
            flat[_method_id] = value
    return flat


def _short_label(label: str) -> str:
    """Shorten a config_id label for column headers.

    Strips common prefixes like ``cfg-YYYYMMDD-`` to produce a shorter
    display label.  If the label does not match the expected pattern,
    returns it unchanged.
    """
    import re

    # Strip "cfg-YYYYMMDD-" prefix
    m = re.match(r"^cfg-\d{8}-(.+)$", label)
    if m:
        return m.group(1)
    return label


def _get_champion_challenger(store: Any, primary_metric: str) -> dict[str, Any]:
    """Return champion and best challenger info as a dict.

    Parameters
    ----------
    store:
        A ResultsStore instance.
    primary_metric:
        The metric name used for ranking (e.g. ``county_mape_overall``).

    Returns
    -------
    dict
        Keys ``champion`` and ``best_challenger``, each a dict with relevant fields.
    """
    result: dict[str, Any] = {"champion": {}, "best_challenger": {}}

    try:
        scorecards = store.get_consolidated_scorecards()
    except Exception:
        return result

    if scorecards.empty or primary_metric not in scorecards.columns:
        return result

    from cohort_projections.analysis.observatory.candidates import build_candidate_view

    candidates = build_candidate_view(scorecards)
    if candidates.empty or primary_metric not in candidates.columns:
        return result

    champion_df = (
        candidates[candidates.get("status_at_run") == "champion"]
        if "status_at_run" in candidates.columns
        else candidates.head(0)
    )

    if not champion_df.empty:
        champ_config = (
            champion_df["config_id"].iloc[0] if "config_id" in champion_df.columns else "unknown"
        )
        champ_value = float(champion_df[primary_metric].iloc[0])
    else:
        best_idx = candidates[primary_metric].idxmin()
        best_row = candidates.loc[best_idx]
        champ_config = best_row.get("config_id", "unknown")
        champ_value = float(best_row[primary_metric])

    result["champion"] = {"config_id": str(champ_config), "metric": champ_value}

    non_champion = (
        candidates[candidates.get("status_at_run") != "champion"]
        if "status_at_run" in candidates.columns
        else candidates.head(0)
    )
    if not non_champion.empty:
        best_idx = non_champion[primary_metric].idxmin()
        best_challenger_id = non_champion.loc[best_idx].get("config_id", "unknown")
        best_challenger_val = float(non_champion.loc[best_idx][primary_metric])

        delta = best_challenger_val - champ_value
        result["best_challenger"] = {
            "config_id": str(best_challenger_id),
            "delta": round(delta, 3),
        }

    return result


def _print_champion_challenger(store: Any, primary_metric: str) -> None:
    """Print champion config and best challenger delta to stdout.

    Parameters
    ----------
    store:
        A ResultsStore instance.
    primary_metric:
        The metric name used for ranking (e.g. ``county_mape_overall``).
    """
    try:
        scorecards = store.get_consolidated_scorecards()
    except Exception:
        return

    if scorecards.empty or primary_metric not in scorecards.columns:
        return

    from cohort_projections.analysis.observatory.candidates import build_candidate_view

    candidates = build_candidate_view(scorecards)
    if candidates.empty or primary_metric not in candidates.columns:
        return

    print()  # noqa: T201

    # Identify champion: rows where status_at_run == "champion"
    champion_df = (
        candidates[candidates.get("status_at_run") == "champion"]
        if "status_at_run" in candidates.columns
        else candidates.head(0)
    )

    if not champion_df.empty:
        champ_config = (
            champion_df["config_id"].iloc[0] if "config_id" in champion_df.columns else "unknown"
        )
        champ_value = champion_df[primary_metric].iloc[0]
        print(f"Champion: {champ_config} ({primary_metric}: {champ_value:.3f})")  # noqa: T201
    else:
        best_idx = candidates[primary_metric].idxmin()
        best_row = candidates.loc[best_idx]
        champ_config = best_row.get("config_id", "unknown")
        champ_value = best_row[primary_metric]
        print(f"Best config: {champ_config} ({primary_metric}: {champ_value:.3f})")  # noqa: T201

    # Best challenger: best non-champion config
    non_champion = (
        candidates[candidates.get("status_at_run") != "champion"]
        if "status_at_run" in candidates.columns
        else candidates.head(0)
    )
    if not non_champion.empty:
        best_idx = non_champion[primary_metric].idxmin()
        best_challenger_id = non_champion.loc[best_idx].get("config_id", "unknown")
        best_challenger_val = non_champion.loc[best_idx][primary_metric]

        delta = best_challenger_val - champ_value
        delta_str = _c_delta(delta)
        print(f"Best challenger: {best_challenger_id} ({delta_str}pp vs champion)")  # noqa: T201


def _print_experiment_log_summary(store: Any) -> None:
    """Print experiment log outcome counts to stdout.

    Parameters
    ----------
    store:
        A ResultsStore instance.
    """
    try:
        log_df = store.get_experiment_log()
    except Exception:
        return

    if log_df.empty:
        return

    # The outcome column contains status values like "needs_human_review", "failed_hard_gate", etc.
    outcome_col = "outcome"
    if outcome_col not in log_df.columns:
        return

    counts = log_df[outcome_col].value_counts()
    if counts.empty:
        return

    print()  # noqa: T201
    print(_c_bold("Experiment Log:"))  # noqa: T201
    for status_val, count in sorted(counts.items()):
        print(f"  {status_val}: {count}")  # noqa: T201


def _safe_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute or dict key from obj, with fallback."""
    if obj is None:
        return default
    try:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
    except Exception:
        pass
    try:
        return obj[attr]  # type: ignore[index]
    except (KeyError, TypeError, IndexError):
        pass
    return default


def _safe_call(obj: Any, method: str, *a: Any, default: Any = None, **kw: Any) -> Any:
    """Call a method on obj if it exists, otherwise return default."""
    fn = getattr(obj, method, None)
    if fn is None or not callable(fn):
        return default
    try:
        return fn(*a, **kw)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, Any] = {
    "status": cmd_status,
    "compare": cmd_compare,
    "rank": cmd_rank,
    "recommend": cmd_recommend,
    "run-pending": cmd_run_pending,
    "run-recommended": cmd_run_recommended,
    "report": cmd_report,
    "refresh": cmd_refresh,
    "diff": cmd_diff,
    "history": cmd_history,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Observatory CLI.

    Parameters
    ----------
    argv:
        Argument list for testing.  Uses ``sys.argv`` when None.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        return 0

    handler = _COMMANDS.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")  # noqa: T201
        parser.print_help()
        return 1

    # Load config and store
    config = _load_config(args.config)
    store = _load_results_store(args.config)

    # Report import issues
    if _IMPORT_ERRORS:
        for mod_name, err in _IMPORT_ERRORS.items():
            logger.debug("Module not available: %s (%s)", mod_name, err)

    return handler(store, config, args)


if __name__ == "__main__":
    sys.exit(main())
