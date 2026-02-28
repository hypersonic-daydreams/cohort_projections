#!/usr/bin/env python3
"""
Compute Age-Specific Convergence Interpolation (Phase 2).

Takes the per-period residual migration rates from Phase 1 and computes
time-varying migration rates for each projection year (2025-2045) using
5-10-5 convergence interpolation.

Each county x age_group x sex cell converges independently from its recent
value toward its long-term mean.

Key ADRs and config:
    ADR-036: Convergence interpolation methodology (5-10-5 schedule)
    Config: rates.migration.interpolation.convergence_schedule
    Config: rates.migration.interpolation.{recent,medium,longterm}_period

Usage:
    python scripts/pipeline/01b_compute_convergence.py
    python scripts/pipeline/01b_compute_convergence.py --config config/projection_config.yaml
"""

import argparse
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.convergence_interpolation import (  # noqa: E402
    run_convergence_pipeline,
)
from cohort_projections.utils import get_logger_from_config, load_projection_config  # noqa: E402

logger = get_logger_from_config(__name__)


def _log_path_status(label: str, paths: Iterable[Path]) -> list[Path]:
    """Log path existence and return missing paths."""
    missing: list[Path] = []
    for path in paths:
        if path.exists():
            logger.info(f"[DRY RUN] OK: {label}: {path}")
        else:
            logger.error(f"[DRY RUN] MISSING: {label}: {path}")
            missing.append(path)
    return missing


def _dry_run_validate_inputs(config: dict, root: Path, variants: list[str | None]) -> bool:
    """Validate key convergence dependencies without writing outputs."""
    output_dir = root / "data" / "processed" / "migration"

    required_inputs = [output_dir / "residual_migration_rates.parquet"]
    missing_inputs = _log_path_status("input", required_inputs)

    if "high" in variants:
        high_variant_inputs = [
            output_dir / "migration_rates_pep_baseline.parquet",
            output_dir / "migration_rates_pep_high.parquet",
        ]
        missing_inputs.extend(_log_path_status("high_variant_input", high_variant_inputs))

    compression = config.get("output", {}).get("compression", "gzip")
    logger.info(f"[DRY RUN] Configured output compression: {compression}")

    for variant in variants:
        suffix = f"_{variant}" if variant else ""
        logger.info(f"[DRY RUN] Would write: {output_dir / f'convergence_rates_by_year{suffix}.parquet'}")
        logger.info(f"[DRY RUN] Would write: {output_dir / f'convergence_metadata{suffix}.json'}")

    return len(missing_inputs) == 0


def main() -> int:
    """Run the convergence interpolation pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute age-specific convergence rates for ND population projections",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to projection config YAML (default: config/projection_config.yaml)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=["high"],
        help="Scenario variant to generate (default: baseline only; 'high' for BEBR high rates)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Generate convergence rates for all variants (baseline + high)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without computing or writing outputs",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading configuration")
        config = load_projection_config(args.config)

        # Determine which variants to run
        variants: list[str | None] = [None]  # Always run baseline
        if args.all_variants:
            variants.append("high")
        elif args.variant:
            variants = [args.variant]

        if args.dry_run:
            logger.info("Dry run enabled: validating convergence dependencies")
            if _dry_run_validate_inputs(config, project_root, variants):
                logger.info("Dry run completed successfully")
                return 0
            logger.error("Dry run failed due to missing required inputs")
            return 1

        for variant in variants:
            variant_label = f" (variant={variant})" if variant else " (baseline)"
            logger.info(f"Starting convergence interpolation pipeline{variant_label}")
            results = run_convergence_pipeline(config, variant=variant)

            logger.info(f"Pipeline completed successfully{variant_label}")
            logger.info(f"  Total rows: {results['total_rows']}")
            logger.info(f"  Output: {results['output_path']}")
            logger.info(f"  Metadata: {results['metadata_path']}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
