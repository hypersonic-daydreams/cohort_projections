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
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


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
        cls = getattr(mod, "ObservatoryComparator")
        return cls(store=store, config=config)
    except Exception as e:
        logger.warning("Failed to construct ObservatoryComparator: %s", e)
        return None


def _load_recommender(
    store: Any, config: dict[str, Any], comparator: Any | None = None
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
        cls = getattr(mod, "ObservatoryRecommender")
        return cls(store=store, comparator=comparator, config=config)
    except Exception as e:
        logger.warning("Failed to construct ObservatoryRecommender: %s", e)
        return None


def _load_variant_catalog(config: dict[str, Any]) -> Any:
    """Load VariantCatalog, or None if not available."""
    mod = _try_import("cohort_projections.analysis.observatory.variant_catalog")
    if mod is None:
        return None
    try:
        cls = getattr(mod, "VariantCatalog")
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
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to observatory config YAML (default: config/observatory_config.yaml).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    sub = parser.add_subparsers(dest="command", help="Subcommand to run.")

    # --- status ---
    sub.add_parser(
        "status",
        help="Display run inventory and variant catalog status.",
    )

    # --- compare ---
    compare_parser = sub.add_parser(
        "compare",
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
        help="Generate next-experiment recommendations.",
    )

    # --- run-pending ---
    pending_parser = sub.add_parser(
        "run-pending",
        help="Run all untested variants from the catalog.",
    )
    pending_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without executing experiments.",
    )

    # --- report ---
    report_parser = sub.add_parser(
        "report",
        help="Generate full HTML observatory report.",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the HTML report (default: auto-generated in data/analysis/observatory/).",
    )

    # --- refresh ---
    sub.add_parser(
        "refresh",
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
    print("Projection Observatory Status")  # noqa: T201
    print("=" * 35)  # noqa: T201

    if store is None:
        print("  (ResultsStore unavailable)")  # noqa: T201
        return 1

    # Run inventory
    try:
        index = store.get_index()
        run_ids = store.get_run_ids()
        n_runs = len(run_ids)
    except Exception as e:
        print(f"  Error loading index: {e}")  # noqa: T201
        n_runs = 0
        index = None

    print(f"Completed runs: {n_runs}")  # noqa: T201

    if index is not None and not index.empty:
        # Date range
        for date_col in ("run_date", "created_at_utc"):
            if date_col in index.columns:
                dates = index[date_col].dropna()
                if not dates.empty:
                    print(f"Date range: {str(dates.min())[:10]} to {str(dates.max())[:10]}")  # noqa: T201
                break

        # Methods
        for method_col in ("method", "method_id", "challenger_method_id"):
            if method_col in index.columns:
                methods = sorted(index[method_col].dropna().unique().tolist())
                print(f"Methods tested: {', '.join(str(m) for m in methods)}")  # noqa: T201
                break

    # Variant catalog
    catalog = _load_variant_catalog(config)
    print()  # noqa: T201
    print("Variant Catalog:")  # noqa: T201

    if catalog is None:
        print("  (VariantCatalog not available)")  # noqa: T201
    else:
        try:
            variants_df = catalog.list_variants()
            total = len(variants_df)
            tested = int(variants_df["tested"].sum()) if "tested" in variants_df.columns else 0
            untested = total - tested

            print(f"  Total variants: {total}")  # noqa: T201
            print(f"  Tested: {tested}")  # noqa: T201
            print(f"  Untested: {untested}")  # noqa: T201
            grids = catalog._grids if hasattr(catalog, "_grids") else {}
            print(f"  Defined grids: {len(grids)}")  # noqa: T201
        except Exception as e:
            print(f"  Error reading catalog: {e}")  # noqa: T201

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

    recommender = _load_recommender(store, config, comparator=comparator)
    recommendations: list[Any] = []
    if recommender is not None:
        try:
            recommendations = recommender.suggest_next_experiments()
        except Exception as e:
            logger.warning("Recommender failed: %s", e)

    ReportClass = _load_report_class()
    if ReportClass is None:
        # Fallback to comparator's own formatting
        print(comparator.format_comparison_summary(result))  # noqa: T201
        return 0

    report = ReportClass(
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

    if result.empty:
        print("No results to rank.")  # noqa: T201
        return 0

    print(f"Ranking by {args.metric} (top {args.top}):")  # noqa: T201
    print("-" * 72)  # noqa: T201
    display = result.head(args.top)
    id_cols = [c for c in ("run_id", "method_id", "config_id") if c in display.columns]
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
    recommender = _load_recommender(store, config, comparator=comparator)
    if recommender is None:
        print("Error: ObservatoryRecommender is not available.")  # noqa: T201
        return 1

    try:
        recommendations = recommender.suggest_next_experiments()
    except Exception as e:
        print(f"Error generating recommendations: {e}")  # noqa: T201
        return 1

    if not recommendations:
        print("No recommendations generated.")  # noqa: T201
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
    catalog = _load_variant_catalog(config)
    if catalog is None:
        print("Error: VariantCatalog is not available.")  # noqa: T201
        print("  This module has not been implemented yet.")  # noqa: T201
        return 1

    # Get untested variants
    try:
        untested = catalog.get_untested()
    except Exception as e:
        print(f"Error loading untested variants: {e}")  # noqa: T201
        return 1

    if not untested:
        print("No untested variants found. All catalog entries have been tested.")  # noqa: T201
        return 0

    print(f"Found {len(untested)} untested variant(s).")  # noqa: T201

    # Generate spec files in a temp directory
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required for spec generation.")  # noqa: T201
        return 1

    with tempfile.TemporaryDirectory(prefix="observatory_pending_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        spec_paths: list[str] = []

        for variant in untested:
            spec = _variant_to_spec(variant, config)
            if spec is None:
                continue
            exp_id = spec.get("experiment_id", "unknown")
            spec_path = tmpdir_path / f"{exp_id}.yaml"
            spec_path.write_text(
                yaml.safe_dump(spec, sort_keys=False, default_flow_style=False),
                encoding="utf-8",
            )
            spec_paths.append(str(spec_path))

        if not spec_paths:
            print("No valid specs generated from untested variants.")  # noqa: T201
            return 0

        print(f"Generated {len(spec_paths)} experiment spec(s).")  # noqa: T201

        if args.dry_run:
            print("Dry run -- would execute:")  # noqa: T201
            for sp in spec_paths:
                print(f"  {sp}")  # noqa: T201
            return 0

        # Delegate to run_experiment_sweep.py
        sweep_script = PROJECT_ROOT / "scripts" / "analysis" / "run_experiment_sweep.py"
        if not sweep_script.exists():
            print(f"Error: sweep script not found at {sweep_script}")  # noqa: T201
            return 1

        cmd = [sys.executable, str(sweep_script), "--specs", *spec_paths]
        print(f"Running: {' '.join(cmd[:4])} ... ({len(spec_paths)} specs)")  # noqa: T201

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode

    return 0


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

    ReportClass = _load_report_class()
    if ReportClass is None:
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
    recommender = _load_recommender(store, config, comparator=comparator)
    if recommender is not None:
        try:
            recommendations = recommender.suggest_next_experiments()
        except Exception as e:
            logger.warning("Recommender failed: %s", e)

    report = ReportClass(
        comparator_result=comparator_result,
        recommendations=recommendations,
        store=store,
    )

    output_path = report.generate_html_report(output_path=args.output)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _variant_to_spec(variant: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a catalog variant to an experiment spec dict.

    Parameters
    ----------
    variant:
        A variant object/dict from the VariantCatalog.
    config:
        The observatory config dict.

    Returns
    -------
    dict or None
        An experiment spec suitable for ``run_experiment_sweep.py``,
        or None if the variant cannot be converted.
    """
    # Try multiple key names for the variant ID
    exp_id = _safe_attr(variant, "variant_id")
    if exp_id is None:
        exp_id = _safe_attr(variant, "experiment_id")
    if exp_id is None:
        exp_id = _safe_attr(variant, "id")
    if exp_id is None:
        return None

    base_method = config.get("challenger_base_method", "m2026r1")

    hypothesis = _safe_attr(variant, "hypothesis", f"Catalog variant: {exp_id}")

    # Build config_delta from parameter + value
    param = _safe_attr(variant, "parameter")
    value = _safe_attr(variant, "value")
    config_delta: dict[str, Any] = {}
    if param and value is not None:
        config_delta[param] = value

    benchmark_label = f"observatory-{exp_id}"

    return {
        "experiment_id": str(exp_id),
        "hypothesis": str(hypothesis),
        "method": str(base_method),
        "profile": f"observatory-{exp_id}",
        "resolved_config": config_delta,
        "scope": "county",
        "benchmark_label": str(benchmark_label),
        "requested_by": "observatory-cli",
    }


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, Any] = {
    "status": cmd_status,
    "compare": cmd_compare,
    "rank": cmd_rank,
    "recommend": cmd_recommend,
    "run-pending": cmd_run_pending,
    "report": cmd_report,
    "refresh": cmd_refresh,
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
