#!/usr/bin/env python3
"""
Data Processing Pipeline for North Dakota Population Projections.

This script orchestrates the processing of raw demographic data into the
standardized format required by the projection engine. It processes:
- Fertility rates (SEER data → cohort fertility table)
- Survival rates (life tables → cohort survival table)
- Migration rates (IRS flows → cohort migration table)

Usage:
    # Process all demographic data
    python 01_process_demographic_data.py --all

    # Process specific components
    python 01_process_demographic_data.py --fertility
    python 01_process_demographic_data.py --survival
    python 01_process_demographic_data.py --migration

    # Process multiple components
    python 01_process_demographic_data.py --fertility --survival

    # Dry run (show what would be processed)
    python 01_process_demographic_data.py --all --dry-run

    # Fail-fast mode (stop on first error)
    python 01_process_demographic_data.py --all --fail-fast
"""

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from project_utils import setup_logger  # noqa: E402

from cohort_projections.data.process.fertility_rates import (  # noqa: E402
    process_fertility_rates,
    validate_fertility_rates,
)
from cohort_projections.data.process.migration_rates import (  # noqa: E402
    process_migration_rates,
    validate_migration_data,
)
from cohort_projections.data.process.survival_rates import (  # noqa: E402
    process_survival_rates,
    validate_survival_rates,
)
from cohort_projections.utils import load_projection_config  # noqa: E402

# Set up logging
logger = setup_logger(__name__, log_level="INFO")


class DataProcessingResult:
    """Container for data processing results."""

    def __init__(self, component: str):
        self.component = component
        self.success = False
        self.error: str | None = None
        self.output_file: Path | None = None
        self.records_processed = 0
        self.processing_time = 0.0
        self.validation_results: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}


class DataProcessingReport:
    """Generate processing report with statistics."""

    def __init__(self):
        self.results: list[DataProcessingResult] = []
        self.start_time = datetime.now(UTC)
        self.end_time: datetime | None = None

    def add_result(self, result: DataProcessingResult):
        """Add a processing result."""
        self.results.append(result)

    def finalize(self):
        """Finalize the report."""
        self.end_time = datetime.now(UTC)

    def get_summary(self) -> dict[str, Any]:
        """Get report summary."""
        successful = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)

        return {
            "total_components": len(self.results),
            "successful": successful,
            "failed": failed,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration": (
                (self.end_time - self.start_time).total_seconds() if self.end_time else None
            ),
            "components": [
                {
                    "component": r.component,
                    "success": r.success,
                    "error": r.error,
                    "output_file": str(r.output_file) if r.output_file else None,
                    "records_processed": r.records_processed,
                    "processing_time": r.processing_time,
                    "validation_passed": (
                        r.validation_results.get("valid", False) if r.validation_results else None
                    ),
                }
                for r in self.results
            ],
        }

    def print_summary(self):
        """Print human-readable summary to console."""
        print("\n" + "=" * 80)
        print("DATA PROCESSING PIPELINE SUMMARY")
        print("=" * 80)

        summary = self.get_summary()
        print(f"\nStart Time: {summary['start_time']}")
        print(f"End Time: {summary['end_time']}")
        print(
            f"Total Duration: {summary['total_duration']:.2f} seconds"
            if summary["total_duration"]
            else "N/A"
        )
        print(f"\nComponents Processed: {summary['total_components']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")

        print("\n" + "-" * 80)
        print("COMPONENT DETAILS")
        print("-" * 80)

        for comp in summary["components"]:
            status = "✓ SUCCESS" if comp["success"] else "✗ FAILED"
            print(f"\n{comp['component'].upper()}: {status}")
            if comp["success"]:
                print(f"  Output: {comp['output_file']}")
                print(f"  Records: {comp['records_processed']:,}")
                print(f"  Processing Time: {comp['processing_time']:.2f}s")
                print(f"  Validation: {'Passed' if comp['validation_passed'] else 'Failed'}")
            else:
                print(f"  Error: {comp['error']}")

        print("\n" + "=" * 80 + "\n")

    def save_report(self, output_dir: Path):
        """Save detailed report to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"data_processing_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(f"Processing report saved to {report_file}")
        return report_file


def process_fertility_data(config: dict[str, Any], dry_run: bool = False) -> DataProcessingResult:
    """
    Process fertility rates data.

    Args:
        config: Project configuration dictionary
        dry_run: If True, only validate inputs without processing

    Returns:
        DataProcessingResult with processing outcome
    """
    result = DataProcessingResult("fertility")
    start_time = datetime.now(UTC)

    try:
        logger.info("Processing fertility rates...")

        # Get configuration
        pipeline_config = config.get("pipeline", {}).get("data_processing", {})
        fertility_config = pipeline_config.get("fertility", {})

        if not fertility_config.get("enabled", True):
            logger.info("Fertility processing disabled in configuration")
            result.success = True
            result.error = "Disabled in configuration"
            return result

        input_file = Path(fertility_config.get("input_file", ""))
        output_file = Path(fertility_config.get("output_file", ""))

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")

        if dry_run:
            logger.info("[DRY RUN] Would process fertility data")
            result.success = True
            return result

        # Process data using the unified pipeline function
        rate_config = config.get("rates", {}).get("fertility", {})
        year_range = (
            2018,
            2022,
        )  # Could come from averaging_period in config

        fertility_rates = process_fertility_rates(
            input_path=input_file,
            output_dir=output_file.parent,
            config=config,
            year_range=year_range,
            averaging_period=rate_config.get("averaging_period", 5),
        )
        logger.info(f"Processed {len(fertility_rates):,} fertility rate records")

        # Validate output
        validation = validate_fertility_rates(fertility_rates)
        result.validation_results = validation

        if not validation.get("valid", False):
            logger.warning(f"Validation warnings: {validation.get('warnings', [])}")

        # Save processed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fertility_rates.to_parquet(output_file, compression="gzip")
        logger.info(f"Saved processed fertility rates to {output_file}")

        # Update result
        result.success = True
        result.output_file = output_file
        result.records_processed = len(fertility_rates)
        result.processing_time = (datetime.now(UTC) - start_time).total_seconds()
        result.metadata = {
            "input_file": str(input_file),
            "year_range": year_range,
            "age_groups": len(fertility_rates["age"].unique()),
            "race_ethnicity_groups": len(fertility_rates["race_ethnicity"].unique()),
        }

    except Exception as e:
        logger.error(f"Error processing fertility data: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.processing_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def process_survival_data(config: dict[str, Any], dry_run: bool = False) -> DataProcessingResult:
    """
    Process survival rates data.

    Args:
        config: Project configuration dictionary
        dry_run: If True, only validate inputs without processing

    Returns:
        DataProcessingResult with processing outcome
    """
    result = DataProcessingResult("survival")
    start_time = datetime.now(UTC)

    try:
        logger.info("Processing survival rates...")

        # Get configuration
        pipeline_config = config.get("pipeline", {}).get("data_processing", {})
        survival_config = pipeline_config.get("survival", {})

        if not survival_config.get("enabled", True):
            logger.info("Survival processing disabled in configuration")
            result.success = True
            result.error = "Disabled in configuration"
            return result

        input_file = Path(survival_config.get("input_file", ""))
        output_file = Path(survival_config.get("output_file", ""))

        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")

        if dry_run:
            logger.info("[DRY RUN] Would process survival data")
            result.success = True
            return result

        # Process data using the unified pipeline function
        rate_config = config.get("rates", {}).get("mortality", {})
        life_table_year = rate_config.get("life_table_year", 2020)

        survival_rates = process_survival_rates(
            input_path=input_file,
            output_dir=output_file.parent,
            config=config,
            base_year=life_table_year,
            improvement_factor=rate_config.get("improvement_factor", 0.005),
        )
        logger.info(f"Processed {len(survival_rates):,} survival rate records")

        # Validate output
        validation = validate_survival_rates(survival_rates)
        result.validation_results = validation

        if not validation.get("valid", False):
            logger.warning(f"Validation warnings: {validation.get('warnings', [])}")

        # Save processed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        survival_rates.to_parquet(output_file, compression="gzip")
        logger.info(f"Saved processed survival rates to {output_file}")

        # Update result
        result.success = True
        result.output_file = output_file
        result.records_processed = len(survival_rates)
        result.processing_time = (datetime.now(UTC) - start_time).total_seconds()
        result.metadata = {
            "input_file": str(input_file),
            "life_table_year": life_table_year,
            "age_groups": len(survival_rates["age"].unique()),
            "sex_groups": len(survival_rates["sex"].unique()),
            "race_ethnicity_groups": len(survival_rates["race_ethnicity"].unique()),
        }

    except Exception as e:
        logger.error(f"Error processing survival data: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.processing_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def process_migration_data(config: dict[str, Any], dry_run: bool = False) -> DataProcessingResult:
    """
    Process migration rates data.

    Args:
        config: Project configuration dictionary
        dry_run: If True, only validate inputs without processing

    Returns:
        DataProcessingResult with processing outcome
    """
    result = DataProcessingResult("migration")
    start_time = datetime.now(UTC)

    try:
        logger.info("Processing migration rates...")

        # Get configuration
        pipeline_config = config.get("pipeline", {}).get("data_processing", {})
        migration_config = pipeline_config.get("migration", {})

        if not migration_config.get("enabled", True):
            logger.info("Migration processing disabled in configuration")
            result.success = True
            result.error = "Disabled in configuration"
            return result

        domestic_input = Path(migration_config.get("domestic_input", ""))
        output_file = Path(migration_config.get("output_file", ""))

        if not domestic_input.exists():
            raise FileNotFoundError(f"Input file not found: {domestic_input}")

        logger.info(f"Domestic input: {domestic_input}")
        logger.info(f"Output file: {output_file}")

        if dry_run:
            logger.info("[DRY RUN] Would process migration data")
            result.success = True
            return result

        # Process data using the unified pipeline function
        year_range = (2018, 2022)  # From averaging_period

        migration_rates = process_migration_rates(
            irs_path=domestic_input,
            output_dir=output_file.parent,
            config=config,
            year_range=year_range,
            target_county_fips=config.get("geography", {}).get("state", "38"),
        )
        logger.info(f"Processed {len(migration_rates):,} migration rate records")

        # Validate output
        validation = validate_migration_data(migration_rates)
        result.validation_results = validation

        if not validation.get("valid", False):
            logger.warning(f"Validation warnings: {validation.get('warnings', [])}")

        # Save processed data
        output_file.parent.mkdir(parents=True, exist_ok=True)
        migration_rates.to_parquet(output_file, compression="gzip")
        logger.info(f"Saved processed migration rates to {output_file}")

        # Update result
        result.success = True
        result.output_file = output_file
        result.records_processed = len(migration_rates)
        result.processing_time = (datetime.now(UTC) - start_time).total_seconds()
        result.metadata = {
            "domestic_input": str(domestic_input),
            "year_range": year_range,
            "geographies": len(migration_rates["fips"].unique()),
        }

    except Exception as e:
        logger.error(f"Error processing migration data: {e}")
        logger.debug(traceback.format_exc())
        result.success = False
        result.error = str(e)

    result.processing_time = (datetime.now(UTC) - start_time).total_seconds()
    return result


def validate_processed_data(report: DataProcessingReport) -> bool:
    """
    Cross-validate all processed data outputs.

    Args:
        report: Processing report with all results

    Returns:
        True if all validation passes
    """
    logger.info("Cross-validating processed data...")

    all_valid = True
    for result in report.results:
        if not result.success:
            continue

        validation = result.validation_results
        if not validation.get("valid", False):
            logger.warning(
                f"{result.component}: Validation failed - {validation.get('errors', [])}"
            )
            all_valid = False

    if all_valid:
        logger.info("All data validation checks passed")
    else:
        logger.warning("Some validation checks failed")

    return all_valid


def generate_processing_report(report: DataProcessingReport, config: dict[str, Any]) -> Path:
    """
    Generate and save processing report.

    Args:
        report: Processing report
        config: Project configuration

    Returns:
        Path to saved report file
    """
    logger.info("Generating processing report...")

    output_dir = (
        Path(
            config.get("pipeline", {})
            .get("data_processing", {})
            .get("output_dir", "data/processed")
        )
        / "reports"
    )

    report_file = report.save_report(output_dir)
    report.print_summary()

    return report_file


def process_all_demographic_data(
    config: dict[str, Any],
    components: list[str],
    dry_run: bool = False,
    fail_fast: bool = False,
) -> DataProcessingReport:
    """
    Main orchestrator for processing all demographic data.

    Args:
        config: Project configuration dictionary
        components: List of components to process ('fertility', 'survival', 'migration')
        dry_run: If True, only validate inputs without processing
        fail_fast: If True, stop on first error

    Returns:
        DataProcessingReport with all results
    """
    logger.info("=" * 80)
    logger.info("DATA PROCESSING PIPELINE - North Dakota Population Projections")
    logger.info("=" * 80)
    logger.info(f"Processing components: {', '.join(components)}")
    logger.info(f"Dry run: {dry_run}")
    logger.info(f"Fail-fast mode: {fail_fast}")
    logger.info("")

    report = DataProcessingReport()

    # Process each component
    if "fertility" in components:
        result = process_fertility_data(config, dry_run=dry_run)
        report.add_result(result)
        if fail_fast and not result.success:
            logger.error("Fail-fast enabled: Stopping due to fertility processing error")
            report.finalize()
            return report

    if "survival" in components:
        result = process_survival_data(config, dry_run=dry_run)
        report.add_result(result)
        if fail_fast and not result.success:
            logger.error("Fail-fast enabled: Stopping due to survival processing error")
            report.finalize()
            return report

    if "migration" in components:
        result = process_migration_data(config, dry_run=dry_run)
        report.add_result(result)
        if fail_fast and not result.success:
            logger.error("Fail-fast enabled: Stopping due to migration processing error")
            report.finalize()
            return report

    # Validate all processed data
    if not dry_run and config.get("pipeline", {}).get("data_processing", {}).get(
        "validate_outputs", True
    ):
        validate_processed_data(report)

    report.finalize()
    return report


def main():
    """Main entry point for data processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process demographic data for North Dakota population projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all demographic data
  python 01_process_demographic_data.py --all

  # Process specific components
  python 01_process_demographic_data.py --fertility
  python 01_process_demographic_data.py --fertility --survival

  # Dry run mode
  python 01_process_demographic_data.py --all --dry-run

  # Fail-fast mode (stop on first error)
  python 01_process_demographic_data.py --all --fail-fast
        """,
    )

    # Component selection
    parser.add_argument("--all", action="store_true", help="Process all demographic data types")
    parser.add_argument("--fertility", action="store_true", help="Process fertility rates")
    parser.add_argument("--survival", action="store_true", help="Process survival rates")
    parser.add_argument("--migration", action="store_true", help="Process migration rates")

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop processing on first error (default: continue on error)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file (default: config/projection_config.yaml)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating processing report",
    )

    args = parser.parse_args()

    # Determine which components to process
    components = []
    if args.all:
        components = ["fertility", "survival", "migration"]
    else:
        if args.fertility:
            components.append("fertility")
        if args.survival:
            components.append("survival")
        if args.migration:
            components.append("migration")

    if not components:
        parser.error("No components specified. Use --all or specify individual components.")

    try:
        # Load configuration
        config = load_projection_config(args.config)

        # Override fail-fast setting if specified
        if args.fail_fast:
            config.setdefault("pipeline", {}).setdefault("data_processing", {})["fail_fast"] = True

        # Process data
        report = process_all_demographic_data(
            config,
            components=components,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )

        # Generate report
        if not args.no_report and config.get("pipeline", {}).get("data_processing", {}).get(
            "generate_report", True
        ):
            generate_processing_report(report, config)
        else:
            report.print_summary()

        # Exit code based on success
        summary = report.get_summary()
        if summary["failed"] > 0:
            logger.error(f"Pipeline completed with {summary['failed']} failures")
            return 1
        else:
            logger.info("Pipeline completed successfully")
            return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
