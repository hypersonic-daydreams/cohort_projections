#!/usr/bin/env python3
"""
Prepare Processed Data for North Dakota Population Projections.

This script converts raw CSV data files into the Parquet format required by
the projection pipeline. It serves as the first step (00_) in the pipeline,
running before 01_process_demographic_data.py.

The script performs simple CSV-to-Parquet conversion without data transformation,
preserving the data as-is while converting to a more efficient format.

Source files (from data/raw/):
    - fertility/asfr_processed.csv          -> fertility_rates.parquet
    - mortality/survival_rates_processed.csv -> survival_rates.parquet
    - migration/nd_migration_processed.csv  -> migration_rates.parquet
    - population/nd_county_population.csv   -> county_population.parquet
    - population/nd_age_sex_race_distribution.csv -> age_sex_race_distribution.parquet

Usage:
    # Convert all CSV files to Parquet
    python 00_prepare_processed_data.py

    # Preview what would be done (dry run)
    python 00_prepare_processed_data.py --dry-run

    # Skip files that already exist
    python 00_prepare_processed_data.py --skip-existing

    # Force overwrite existing files (default behavior)
    python 00_prepare_processed_data.py --force
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cohort_projections.utils.logger import setup_logger  # noqa: E402

# Set up logging
logger = setup_logger(__name__, log_level="INFO")


@dataclass
class FileMapping:
    """Mapping from source CSV to destination Parquet."""

    source: Path
    destination: Path
    description: str


@dataclass
class ConversionResult:
    """Result of a single file conversion."""

    source: Path
    destination: Path
    success: bool
    rows: int = 0
    columns: int = 0
    error: str | None = None
    skipped: bool = False


@dataclass
class ConversionSummary:
    """Summary of all file conversions."""

    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    results: list[ConversionResult] = field(default_factory=list)

    def add_result(self, result: ConversionResult):
        """Add a conversion result."""
        self.results.append(result)

    def finalize(self):
        """Finalize the summary."""
        self.end_time = datetime.now(UTC)

    @property
    def successful(self) -> int:
        """Count of successful conversions."""
        return sum(1 for r in self.results if r.success and not r.skipped)

    @property
    def failed(self) -> int:
        """Count of failed conversions."""
        return sum(1 for r in self.results if not r.success)

    @property
    def skipped(self) -> int:
        """Count of skipped conversions."""
        return sum(1 for r in self.results if r.skipped)

    @property
    def total_rows(self) -> int:
        """Total rows processed."""
        return sum(r.rows for r in self.results if r.success)


def get_file_mappings(data_dir: Path) -> list[FileMapping]:
    """
    Get the file mappings from raw CSV to processed Parquet.

    Args:
        data_dir: Base data directory (typically 'data/')

    Returns:
        List of FileMapping objects
    """
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    return [
        FileMapping(
            source=raw_dir / "fertility" / "asfr_processed.csv",
            destination=processed_dir / "fertility_rates.parquet",
            description="Age-specific fertility rates",
        ),
        FileMapping(
            source=raw_dir / "mortality" / "survival_rates_processed.csv",
            destination=processed_dir / "survival_rates.parquet",
            description="Age/sex/race survival rates",
        ),
        FileMapping(
            source=raw_dir / "migration" / "nd_migration_processed.csv",
            destination=processed_dir / "migration_rates.parquet",
            description="County migration flows",
        ),
        FileMapping(
            source=raw_dir / "population" / "nd_county_population.csv",
            destination=processed_dir / "county_population.parquet",
            description="County population totals",
        ),
        FileMapping(
            source=raw_dir / "population" / "nd_age_sex_race_distribution.csv",
            destination=processed_dir / "age_sex_race_distribution.parquet",
            description="Age/sex/race population distribution",
        ),
    ]


def validate_source_files(
    mappings: list[FileMapping],
) -> tuple[list[FileMapping], list[FileMapping]]:
    """
    Validate that source CSV files exist.

    Args:
        mappings: List of file mappings to validate

    Returns:
        Tuple of (valid_mappings, missing_mappings)
    """
    valid = []
    missing = []

    for mapping in mappings:
        if mapping.source.exists():
            valid.append(mapping)
        else:
            missing.append(mapping)

    return valid, missing


def convert_csv_to_parquet(
    mapping: FileMapping,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> ConversionResult:
    """
    Convert a single CSV file to Parquet format.

    Args:
        mapping: Source and destination file mapping
        skip_existing: If True, skip if destination exists
        dry_run: If True, don't actually convert

    Returns:
        ConversionResult with outcome details
    """
    result = ConversionResult(
        source=mapping.source,
        destination=mapping.destination,
        success=False,
    )

    try:
        # Check if destination exists
        if mapping.destination.exists() and skip_existing:
            logger.info(f"Skipping (exists): {mapping.destination.name}")
            result.success = True
            result.skipped = True
            # Get row count from existing file for summary
            existing_df = pd.read_parquet(mapping.destination)
            result.rows = len(existing_df)
            result.columns = len(existing_df.columns)
            return result

        if dry_run:
            logger.info(
                f"[DRY RUN] Would convert: {mapping.source.name} -> {mapping.destination.name}"
            )
            result.success = True
            return result

        # Read CSV
        logger.info(f"Reading: {mapping.source}")
        df = pd.read_csv(mapping.source)

        # Ensure output directory exists
        mapping.destination.parent.mkdir(parents=True, exist_ok=True)

        # Write Parquet with gzip compression
        logger.info(f"Writing: {mapping.destination}")
        df.to_parquet(mapping.destination, compression="gzip", index=False)

        result.success = True
        result.rows = len(df)
        result.columns = len(df.columns)

        logger.info(f"  -> {result.rows:,} rows, {result.columns} columns")

    except Exception as e:
        result.success = False
        result.error = str(e)
        logger.error(f"Error converting {mapping.source.name}: {e}")

    return result


def print_summary(summary: ConversionSummary):
    """Print a human-readable summary of the conversion."""
    print("\n" + "=" * 70)
    print("PREPARE PROCESSED DATA - SUMMARY")
    print("=" * 70)

    if summary.end_time:
        duration = (summary.end_time - summary.start_time).total_seconds()
        print(f"\nDuration: {duration:.2f} seconds")

    print(f"\nFiles processed: {len(summary.results)}")
    print(f"  Successful: {summary.successful}")
    print(f"  Skipped:    {summary.skipped}")
    print(f"  Failed:     {summary.failed}")
    print(f"\nTotal rows: {summary.total_rows:,}")

    print("\n" + "-" * 70)
    print("FILE DETAILS")
    print("-" * 70)

    for result in summary.results:
        if result.skipped:
            status = "SKIPPED"
        elif result.success:
            status = "OK"
        else:
            status = "FAILED"

        print(f"\n{result.destination.name}: {status}")
        if result.success:
            print(f"  Source: {result.source}")
            print(f"  Rows: {result.rows:,}, Columns: {result.columns}")
        elif result.error:
            print(f"  Error: {result.error}")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point for prepare processed data script."""
    parser = argparse.ArgumentParser(
        description="Convert raw CSV data files to Parquet format for the projection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all CSV files to Parquet
  python 00_prepare_processed_data.py

  # Preview what would be done
  python 00_prepare_processed_data.py --dry-run

  # Skip files that already exist
  python 00_prepare_processed_data.py --skip-existing

  # Use custom data directory
  python 00_prepare_processed_data.py --data-dir /path/to/data
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually converting files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the destination",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing files (default behavior)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Base data directory (default: project_root/data)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PREPARE PROCESSED DATA - North Dakota Population Projections")
    logger.info("=" * 70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info("")

    # Get file mappings
    mappings = get_file_mappings(args.data_dir)
    logger.info(f"Found {len(mappings)} files to process")

    # Validate source files exist
    valid_mappings, missing_mappings = validate_source_files(mappings)

    if missing_mappings:
        logger.warning(f"Missing {len(missing_mappings)} source files:")
        for mapping in missing_mappings:
            logger.warning(f"  - {mapping.source}")

    if not valid_mappings:
        logger.error("No valid source files found. Exiting.")
        return 1

    logger.info(f"Processing {len(valid_mappings)} valid source files")
    logger.info("")

    # Process each file
    summary = ConversionSummary()

    for mapping in valid_mappings:
        logger.info(f"Processing: {mapping.description}")
        result = convert_csv_to_parquet(
            mapping,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        summary.add_result(result)

    # Add missing files as failed results
    for mapping in missing_mappings:
        summary.add_result(
            ConversionResult(
                source=mapping.source,
                destination=mapping.destination,
                success=False,
                error="Source file not found",
            )
        )

    summary.finalize()

    # Print summary
    print_summary(summary)

    # Return appropriate exit code
    if summary.failed > 0:
        logger.error(f"Completed with {summary.failed} failures")
        return 1
    else:
        logger.info("All files processed successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
