#!/usr/bin/env python3
"""
Initialize cohort projections project.

Creates necessary directories and validates environment setup.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.utils.logger import setup_logger
from cohort_projections.utils.config_loader import ConfigLoader

logger = setup_logger(__name__)


def create_directories():
    """Create all necessary project directories."""
    logger.info("Creating project directories...")

    dirs_to_create = [
        "data/raw/census/pep",
        "data/raw/census/acs",
        "data/raw/census/decennial",
        "data/raw/seer",
        "data/raw/vital_stats",
        "data/raw/migration",
        "data/raw/geographic",
        "data/processed/base_population",
        "data/processed/fertility",
        "data/processed/mortality",
        "data/processed/migration",
        "data/processed/validation",
        "data/projections/state",
        "data/projections/county",
        "data/projections/places",
        "data/projections/scenarios",
        "data/metadata",
        "logs",
    ]

    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created: {full_path}")

    logger.info(f"Created {len(dirs_to_create)} directories")


def validate_config():
    """Validate configuration files exist and are readable."""
    logger.info("Validating configuration...")

    config_loader = ConfigLoader()

    try:
        config = config_loader.get_projection_config()
        logger.info("✓ Projection configuration loaded successfully")

        # Validate key parameters
        base_year = config['project']['base_year']
        horizon = config['project']['projection_horizon']
        logger.info(f"  Base year: {base_year}")
        logger.info(f"  Projection horizon: {horizon} years")
        logger.info(f"  End year: {base_year + horizon}")

        # Check demographics
        demo = config['demographics']
        logger.info(f"  Age range: {demo['age_groups']['min_age']}-{demo['age_groups']['max_age']}+")
        logger.info(f"  Race/ethnicity categories: {len(demo['race_ethnicity']['categories'])}")

        return True

    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False


def check_environment():
    """Check Python environment and dependencies."""
    logger.info("Checking Python environment...")

    import importlib

    required_packages = [
        'pandas',
        'numpy',
        'pyarrow',
        'requests',
        'yaml',
        'tqdm',
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"✗ {package} not found")
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Run: pip install -r requirements.txt")
        return False

    logger.info("All required packages installed")
    return True


def main():
    """Main initialization function."""
    logger.info("="  * 60)
    logger.info("Cohort Projections - Project Initialization")
    logger.info("=" * 60)

    # Create directories
    create_directories()

    # Validate configuration
    config_valid = validate_config()

    # Check environment
    env_valid = check_environment()

    # Summary
    logger.info("=" * 60)
    if config_valid and env_valid:
        logger.info("✓ Project initialization complete!")
        logger.info("  Ready to download base data")
        logger.info("  Next: Run scripts/setup/02_download_base_data.py")
        return 0
    else:
        logger.error("✗ Initialization incomplete - fix errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
