#!/usr/bin/env python3
"""
Compute ND-Adjusted Mortality Improvement Projections.

This script runs the mortality improvement pipeline (Phase 3), which:
1. Loads Census Bureau NP2023-A4 national survival ratio projections
2. Loads ND CDC 2020 baseline survival rates
3. Computes ND-to-national adjustment factors
4. Applies adjustments to all projection years (2025-2045)
5. Saves ND-adjusted survival projections to data/processed/mortality/

Usage:
    python scripts/pipeline/01c_compute_mortality_improvement.py
    python scripts/pipeline/01c_compute_mortality_improvement.py --config config/projection_config.yaml
"""

import argparse
import sys
import traceback
from collections.abc import Iterable
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.mortality_improvement import (  # noqa: E402
    run_mortality_improvement_pipeline,
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


def _dry_run_validate_inputs(root: Path, config: dict) -> bool:
    """Validate key mortality-improvement dependencies without writing outputs."""
    required_inputs = [
        root
        / "data"
        / "raw"
        / "census_bureau_methodology"
        / "np2023_a4_survival_ratios.csv",
        root
        / "data"
        / "processed"
        / "sdc_2024"
        / "survival_rates_sdc_2024_by_age_group.csv",
    ]
    missing_inputs = _log_path_status("input", required_inputs)

    output_dir = root / "data" / "processed" / "mortality"
    logger.info(f"[DRY RUN] Would write: {output_dir / 'nd_adjusted_survival_projections.parquet'}")
    logger.info(f"[DRY RUN] Would write: {output_dir / 'mortality_improvement_metadata.json'}")
    logger.info(
        "[DRY RUN] Projection range from config: "
        f"{config.get('project', {}).get('base_year', 2025)}-"
        f"{config.get('project', {}).get('base_year', 2025) + config.get('project', {}).get('projection_horizon', 20)}"
    )

    return len(missing_inputs) == 0


def main() -> int:
    """Run the mortality improvement pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute ND-adjusted mortality improvement projections",
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
        help="Validate configuration and inputs without computing or writing outputs",
    )
    args = parser.parse_args()

    try:
        logger.info("Loading configuration")
        config = load_projection_config(args.config)

        if args.dry_run:
            logger.info("Dry run enabled: validating mortality improvement dependencies")
            if _dry_run_validate_inputs(project_root, config):
                logger.info("Dry run completed successfully")
                return 0
            logger.error("Dry run failed due to missing required inputs")
            return 1

        logger.info("Starting mortality improvement pipeline")
        result = run_mortality_improvement_pipeline(config)

        logger.info("Pipeline completed successfully")
        logger.info(f"  Total rows: {len(result)}")
        logger.info(f"  Years: {result['year'].min()}-{result['year'].max()}")
        logger.info(f"  Ages: {result['age'].min()}-{result['age'].max()}")
        logger.info(f"  Sexes: {sorted(result['sex'].unique().tolist())}")
        logger.info(
            f"  Survival rate range: "
            f"[{result['survival_rate'].min():.6f}, "
            f"{result['survival_rate'].max():.6f}]"
        )

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
