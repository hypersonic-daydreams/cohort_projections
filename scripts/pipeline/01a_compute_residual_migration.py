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
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.residual_migration import (  # noqa: E402
    run_residual_migration_pipeline,
)
from cohort_projections.utils import get_logger_from_config, load_projection_config  # noqa: E402

logger = get_logger_from_config(__name__)


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
    args = parser.parse_args()

    try:
        logger.info("Loading configuration")
        config = load_projection_config(args.config)

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
