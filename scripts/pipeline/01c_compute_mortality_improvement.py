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
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.data.process.mortality_improvement import (  # noqa: E402
    run_mortality_improvement_pipeline,
)
from cohort_projections.utils import get_logger_from_config, load_projection_config  # noqa: E402

logger = get_logger_from_config(__name__)


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
    args = parser.parse_args()

    try:
        logger.info("Loading configuration")
        config = load_projection_config(args.config)

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
