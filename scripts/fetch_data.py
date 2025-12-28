#!/usr/bin/env python3
"""
Data Fetch Script for North Dakota Population Projections.

This script reads the data_sources.yaml manifest and fetches data files from
sibling repositories on the local machine. For files that don't exist locally,
it reports what's missing and provides download instructions.

This is part of the data management strategy defined in ADR-016.

Usage:
    # Fetch all available data from sibling repos
    python scripts/fetch_data.py

    # Dry run (show what would be fetched without copying)
    python scripts/fetch_data.py --dry-run

    # Fetch specific category only
    python scripts/fetch_data.py --category geographic
    python scripts/fetch_data.py --category population

    # Force re-fetch even if destination exists
    python scripts/fetch_data.py --force

    # Verbose output
    python scripts/fetch_data.py --verbose

    # List available data sources
    python scripts/fetch_data.py --list

Examples:
    # Initial setup on a new machine
    python scripts/fetch_data.py --dry-run  # See what would be fetched
    python scripts/fetch_data.py            # Fetch available data

    # Update specific data category
    python scripts/fetch_data.py --category population --force
"""

import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import os

import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FetchResult:
    """Result of attempting to fetch a single data source."""

    source_name: str
    category: str
    status: str = "pending"  # "fetched", "exists", "not_found", "error", "skipped"
    source_path: Optional[Path] = None
    destination_path: Optional[Path] = None
    message: str = ""
    rows_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    missing_columns: List[str] = field(default_factory=list)


@dataclass
class FetchReport:
    """Summary report of all fetch operations."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    results: List[FetchResult] = field(default_factory=list)
    dry_run: bool = False

    def add_result(self, result: FetchResult) -> None:
        """Add a fetch result to the report."""
        self.results.append(result)

    def finalize(self) -> None:
        """Mark the report as complete."""
        self.end_time = datetime.now()

    @property
    def fetched_count(self) -> int:
        """Number of files successfully fetched."""
        return sum(1 for r in self.results if r.status == "fetched")

    @property
    def exists_count(self) -> int:
        """Number of files that already existed."""
        return sum(1 for r in self.results if r.status == "exists")

    @property
    def not_found_count(self) -> int:
        """Number of files not found locally."""
        return sum(1 for r in self.results if r.status == "not_found")

    @property
    def error_count(self) -> int:
        """Number of files with errors."""
        return sum(1 for r in self.results if r.status == "error")


# =============================================================================
# Core Functions
# =============================================================================


def load_data_sources(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the data sources manifest from YAML.

    Args:
        config_path: Path to data_sources.yaml (default: config/data_sources.yaml)

    Returns:
        Dictionary containing all data source definitions
    """
    if config_path is None:
        config_path = project_root / "config" / "data_sources.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Data sources manifest not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def expand_path(path_str: str) -> Path:
    """
    Expand environment variables and ~ in a path string.

    Args:
        path_str: Path string potentially containing ${HOME} or ~

    Returns:
        Expanded Path object
    """
    # Expand ${VAR} style environment variables
    expanded = os.path.expandvars(path_str)
    # Expand ~ to home directory
    expanded = os.path.expanduser(expanded)
    return Path(expanded)


def find_source_file(source_paths: List[str]) -> Optional[Path]:
    """
    Find the first existing source file from a list of potential paths.

    Args:
        source_paths: List of potential source file paths

    Returns:
        Path to the first existing file, or None if none found
    """
    for path_str in source_paths:
        path = expand_path(path_str)
        if path.exists():
            return path
    return None


def validate_columns(
    file_path: Path, required_columns: List[str], verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that a data file contains required columns.

    Args:
        file_path: Path to the data file
        required_columns: List of column names that must be present
        verbose: Whether to print verbose output

    Returns:
        Tuple of (valid: bool, missing_columns: List[str])
    """
    if not required_columns:
        return True, []

    try:
        # Import pandas here to avoid loading it if not needed
        import pandas as pd

        # Read just the header based on file type
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path, nrows=0)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
            # For parquet, we have the full file, but just check columns
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(file_path, nrows=0)
        else:
            if verbose:
                print(f"  Warning: Cannot validate columns for {suffix} files")
            return True, []

        actual_columns = set(df.columns)
        missing = [col for col in required_columns if col not in actual_columns]

        return len(missing) == 0, missing

    except Exception as e:
        if verbose:
            print(f"  Warning: Could not validate columns: {e}")
        return True, []  # Assume valid if we can't check


def get_file_info(file_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Get file size and row count for a data file.

    Args:
        file_path: Path to the data file

    Returns:
        Tuple of (row_count, file_size_bytes)
    """
    file_size = file_path.stat().st_size if file_path.exists() else None
    row_count = None

    try:
        import pandas as pd

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            # Count lines (minus header)
            with open(file_path, "r") as f:
                row_count = sum(1 for _ in f) - 1
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
            row_count = len(df)
    except Exception:
        pass

    return row_count, file_size


def copy_file(
    source_path: Path, dest_path: Path, force: bool = False, verbose: bool = False
) -> bool:
    """
    Copy a file from source to destination.

    Args:
        source_path: Source file path
        dest_path: Destination file path
        force: Overwrite if destination exists
        verbose: Print verbose output

    Returns:
        True if copy was successful
    """
    # Create destination directory if needed
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        if verbose:
            print(f"  Destination exists, skipping (use --force to overwrite)")
        return False

    try:
        shutil.copy2(source_path, dest_path)
        return True
    except Exception as e:
        if verbose:
            print(f"  Error copying file: {e}")
        return False


def fetch_data_source(
    name: str,
    category: str,
    source_config: Dict[str, Any],
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> FetchResult:
    """
    Fetch a single data source.

    Args:
        name: Name of the data source
        category: Category (geographic, population, etc.)
        source_config: Configuration dict for this source
        dry_run: If True, don't actually copy files
        force: If True, overwrite existing files
        verbose: If True, print detailed output

    Returns:
        FetchResult with the outcome
    """
    result = FetchResult(source_name=name, category=category)

    # Get destination path
    dest_rel = source_config.get("destination", "")
    if not dest_rel:
        result.status = "error"
        result.message = "No destination path configured"
        return result

    dest_path = project_root / dest_rel
    result.destination_path = dest_path

    # Check if destination already exists
    if dest_path.exists() and not force:
        result.status = "exists"
        result.message = "Destination file already exists"
        row_count, file_size = get_file_info(dest_path)
        result.rows_count = row_count
        result.file_size_bytes = file_size

        # Validate columns
        required_cols = source_config.get("required_columns", [])
        valid, missing = validate_columns(dest_path, required_cols, verbose)
        result.missing_columns = missing

        return result

    # Try to find source file
    source_paths = source_config.get("source_paths", [])
    source_path = find_source_file(source_paths)

    if source_path is None:
        result.status = "not_found"
        external_url = source_config.get("external_url", "N/A")
        result.message = f"Not available locally. Download from: {external_url}"
        return result

    result.source_path = source_path

    # Dry run - just report what would be done
    if dry_run:
        result.status = "skipped"
        result.message = f"Would copy from {source_path}"
        row_count, file_size = get_file_info(source_path)
        result.rows_count = row_count
        result.file_size_bytes = file_size
        return result

    # Copy the file
    if verbose:
        print(f"  Copying from {source_path}")

    if copy_file(source_path, dest_path, force=force, verbose=verbose):
        result.status = "fetched"
        result.message = f"Copied from {source_path}"
        row_count, file_size = get_file_info(dest_path)
        result.rows_count = row_count
        result.file_size_bytes = file_size

        # Validate columns
        required_cols = source_config.get("required_columns", [])
        valid, missing = validate_columns(dest_path, required_cols, verbose)
        result.missing_columns = missing

        if missing:
            result.message += f" (Warning: missing columns: {missing})"
    else:
        result.status = "error"
        result.message = "Failed to copy file"

    return result


def fetch_all_data(
    config: Dict[str, Any],
    categories: Optional[List[str]] = None,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> FetchReport:
    """
    Fetch all data sources from the manifest.

    Args:
        config: Data sources configuration
        categories: List of categories to fetch (None = all)
        dry_run: If True, don't actually copy files
        force: If True, overwrite existing files
        verbose: If True, print detailed output

    Returns:
        FetchReport with all results
    """
    report = FetchReport(dry_run=dry_run)

    data_sources = config.get("data_sources", {})
    all_categories = list(data_sources.keys())

    # Filter to requested categories
    if categories:
        fetch_categories = [c for c in categories if c in all_categories]
        invalid = [c for c in categories if c not in all_categories]
        if invalid:
            print(f"Warning: Unknown categories ignored: {invalid}")
            print(f"Valid categories: {all_categories}")
    else:
        fetch_categories = all_categories

    # Process each category
    for category in fetch_categories:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Category: {category.upper()}")
            print("=" * 60)

        sources = data_sources.get(category, {})

        for name, source_config in sources.items():
            if verbose:
                print(f"\n{name}:")
                desc = source_config.get("description", "").strip().split("\n")[0]
                print(f"  {desc}")

            result = fetch_data_source(
                name=name,
                category=category,
                source_config=source_config,
                dry_run=dry_run,
                force=force,
                verbose=verbose,
            )

            report.add_result(result)

            if verbose:
                status_icon = {
                    "fetched": "[OK]",
                    "exists": "[SKIP]",
                    "not_found": "[MISSING]",
                    "error": "[ERROR]",
                    "skipped": "[DRY-RUN]",
                }.get(result.status, "[?]")
                print(f"  Status: {status_icon} {result.message}")

    report.finalize()
    return report


def list_data_sources(config: Dict[str, Any]) -> None:
    """
    Print a formatted list of all data sources.

    Args:
        config: Data sources configuration
    """
    data_sources = config.get("data_sources", {})

    print("\n" + "=" * 80)
    print("DATA SOURCES MANIFEST")
    print("=" * 80)

    for category, sources in data_sources.items():
        print(f"\n{category.upper()}")
        print("-" * 40)

        for name, source_config in sources.items():
            desc = source_config.get("description", "").strip().split("\n")[0]
            dest = source_config.get("destination", "N/A")
            source_paths = source_config.get("source_paths", [])

            # Check availability
            source_path = find_source_file(source_paths)
            dest_path = project_root / dest if dest != "N/A" else None
            dest_exists = dest_path.exists() if dest_path else False

            if dest_exists:
                status = "[OK]"
            elif source_path:
                status = "[AVAILABLE]"
            else:
                status = "[MUST DOWNLOAD]"

            print(f"\n  {name}: {status}")
            print(f"    {desc[:70]}")
            print(f"    Destination: {dest}")

            if not source_path and not dest_exists:
                url = source_config.get("external_url", "N/A")
                print(f"    Download from: {url}")


def print_report(report: FetchReport) -> None:
    """
    Print a summary report of fetch operations.

    Args:
        report: FetchReport to print
    """
    print("\n" + "=" * 80)
    print("DATA FETCH SUMMARY")
    print("=" * 80)

    if report.dry_run:
        print("\n[DRY RUN - No files were actually copied]")

    duration = (
        (report.end_time - report.start_time).total_seconds()
        if report.end_time
        else 0
    )

    print(f"\nStart Time: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"\nResults:")
    print(f"  Fetched:    {report.fetched_count}")
    print(f"  Exists:     {report.exists_count}")
    print(f"  Not Found:  {report.not_found_count}")
    print(f"  Errors:     {report.error_count}")

    # Group by status
    not_found = [r for r in report.results if r.status == "not_found"]
    errors = [r for r in report.results if r.status == "error"]
    missing_cols = [r for r in report.results if r.missing_columns]

    if not_found:
        print("\n" + "-" * 80)
        print("DATA FILES NOT AVAILABLE LOCALLY (must be downloaded):")
        print("-" * 80)
        for result in not_found:
            print(f"\n  {result.category}/{result.source_name}")
            print(f"    {result.message}")

    if errors:
        print("\n" + "-" * 80)
        print("ERRORS:")
        print("-" * 80)
        for result in errors:
            print(f"\n  {result.category}/{result.source_name}")
            print(f"    {result.message}")

    if missing_cols:
        print("\n" + "-" * 80)
        print("VALIDATION WARNINGS (missing required columns):")
        print("-" * 80)
        for result in missing_cols:
            print(f"\n  {result.category}/{result.source_name}")
            print(f"    Missing columns: {result.missing_columns}")

    print("\n" + "=" * 80 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for the fetch script."""
    parser = argparse.ArgumentParser(
        description="Fetch data files from sibling repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all available data
  python scripts/fetch_data.py

  # Dry run to see what would be fetched
  python scripts/fetch_data.py --dry-run

  # Fetch only geographic data
  python scripts/fetch_data.py --category geographic

  # Force re-fetch existing files
  python scripts/fetch_data.py --force

  # List all data sources
  python scripts/fetch_data.py --list

Data Categories:
  geographic  - County, place, and metro area reference data
  population  - Base population estimates and PUMS microdata
  fertility   - Age-specific fertility rates (SEER/CDC)
  mortality   - Life tables and survival rates (SEER/CDC)
  migration   - IRS county-to-county flows
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without copying files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        metavar="NAME",
        help="Fetch only specified category (can be repeated)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all data sources and exit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to data_sources.yaml (default: config/data_sources.yaml)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_data_sources(args.config)

        # List mode
        if args.list:
            list_data_sources(config)
            return 0

        # Fetch data
        print("=" * 80)
        print("DATA FETCH - North Dakota Population Projections")
        print("=" * 80)
        print(f"Config: {args.config or 'config/data_sources.yaml'}")
        print(f"Dry Run: {args.dry_run}")
        print(f"Force: {args.force}")
        print(f"Categories: {args.categories or 'all'}")

        report = fetch_all_data(
            config,
            categories=args.categories,
            dry_run=args.dry_run,
            force=args.force,
            verbose=args.verbose,
        )

        print_report(report)

        # Return non-zero if there were errors or missing required data
        if report.error_count > 0:
            return 1

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
