#!/usr/bin/env python3
"""
Compute Residual Migration Rates for North Dakota Population Projections.

This script runs the residual migration pipeline, which:
1. Loads Census/PEP population data at 6 time points (2000-2024)
2. Computes residual migration rates for 5 inter-censal periods
3. Applies oil-boom dampening, male dampening, and college-age smoothing
4. Averages rates across all periods
5. Saves results to data/processed/migration/

Key ADRs and config:
    ADR-035: Census PEP data source
    ADR-036: BEBR multi-period averaging and convergence methodology
    ADR-040: Bakken boom dampening (config: rates.migration.domestic.dampening)
    Config: rates.migration.domestic.residual.periods, dampening.boom_periods

Usage:
    python scripts/pipeline/01a_compute_residual_migration.py
    python scripts/pipeline/01a_compute_residual_migration.py --config config/projection_config.yaml
"""

import argparse
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.residual_migration import (  # noqa: E402
    run_residual_migration_pipeline,
)
from cohort_projections.utils import (  # noqa: E402
    get_logger_from_config,
    load_projection_config,
    resolve_sdc_rate_file,
)

logger = get_logger_from_config(__name__)


def _resolve_path(path_value: str | Path, root: Path) -> Path:
    """Resolve a configured path value to an absolute filesystem path."""
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _log_path_status(label: str, paths: Iterable[Path]) -> list[Path]:
    """Log existence status for a list of paths and return missing ones."""
    missing: list[Path] = []
    for path in paths:
        if path.exists():
            logger.info(f"[DRY RUN] OK: {label}: {path}")
        else:
            logger.error(f"[DRY RUN] MISSING: {label}: {path}")
            missing.append(path)
    return missing


def _dry_run_validate_inputs(config: dict, root: Path) -> bool:
    """Validate key input dependencies without running the pipeline."""
    data_paths = config.get("data_paths", {})
    shared_data_root = Path.home() / "workspace" / "shared-data"

    required_inputs = [
        _resolve_path(
            data_paths.get(
                "census_2000_county_age_sex",
                str(
                    root
                    / "data"
                    / "raw"
                    / "nd_sdc_2024_projections"
                    / "source_files"
                    / "reference"
                    / "Census 2000 County Age and Sex.xlsx"
                ),
            ),
            root,
        ),
        _resolve_path(
            data_paths.get(
                "pep_2010_2019_county_age_sex",
                str(
                    root
                    / "data"
                    / "raw"
                    / "nd_sdc_2024_projections"
                    / "source_files"
                    / "reference"
                    / "cc-est2019-agesex-38 (1).xlsx"
                ),
            ),
            root,
        ),
        _resolve_path(
            data_paths.get(
                "pep_2020_2024_county_age_sex",
                str(
                    shared_data_root
                    / "census"
                    / "popest"
                    / "parquet"
                    / "2020-2024"
                    / "county"
                    / "cc-est2024-agesex-all.parquet"
                ),
            ),
            root,
        ),
        _resolve_path(
            data_paths.get(
                "base_population_2020",
                str(resolve_sdc_rate_file("base_population_by_county.csv", project_root=root)),
            ),
            root,
        ),
        _resolve_path(
            data_paths.get(
                "survival_rates",
                str(
                    root
                    / "data"
                    / "processed"
                    / "sdc_2024"
                    / "survival_rates_sdc_2024_by_age_group.csv"
                ),
            ),
            root,
        ),
    ]

    missing_inputs = _log_path_status("input", required_inputs)

    residual_cfg = config.get("rates", {}).get("migration", {}).get("domestic", {}).get("residual", {})

    gq_cfg = residual_cfg.get("gq_correction", {})
    if gq_cfg.get("enabled", False):
        gq_path = _resolve_path(
            gq_cfg.get("historical_gq_path", "data/processed/gq_county_age_sex_historical.parquet"),
            root,
        )
        missing_inputs.extend(_log_path_status("gq_correction", [gq_path]))

    pep_cfg = residual_cfg.get("pep_recalibration", {})
    if pep_cfg.get("enabled", False):
        pep_path = _resolve_path(
            pep_cfg.get("pep_data_path", "data/processed/pep_county_components_2000_2025.parquet"),
            root,
        )
        missing_inputs.extend(_log_path_status("pep_recalibration", [pep_path]))

    output_dir = root / "data" / "processed" / "migration"
    expected_outputs = [
        output_dir / "residual_migration_rates.parquet",
        output_dir / "residual_migration_rates_averaged.parquet",
        output_dir / "residual_migration_metadata.json",
    ]
    for output_path in expected_outputs:
        logger.info(f"[DRY RUN] Would write: {output_path}")

    return len(missing_inputs) == 0


def main() -> int:
    """Run the residual migration pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute residual migration rates for ND population projections",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to projection config YAML (default: config/projection_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and input dependencies without computing or writing outputs",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading configuration")
        config = load_projection_config(args.config)

        if args.dry_run:
            logger.info("Dry run enabled: validating residual migration dependencies")
            if _dry_run_validate_inputs(config, project_root):
                logger.info("Dry run completed successfully")
                return 0
            logger.error("Dry run failed due to missing required inputs")
            return 1

        logger.info("Starting residual migration pipeline")
        results = run_residual_migration_pipeline(config)

        all_periods = results["all_periods"]
        averaged = results["averaged"]

        logger.info("Pipeline completed successfully")
        logger.info(f"  All-period rates: {len(all_periods)} rows")
        logger.info(f"  Averaged rates: {len(averaged)} rows")
        logger.info(f"  Counties: {averaged['county_fips'].nunique()}")
        logger.info(f"  Mean migration rate: {averaged['migration_rate'].mean():.4f}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
