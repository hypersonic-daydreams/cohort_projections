#!/usr/bin/env python3
"""
Pre-commit hook: Data Manifest Enforcement

Ensures that any new data files added to the repository are documented
in the cohort_projections_meta PostgreSQL database before the commit is allowed.

This hook:
1. Identifies staged files in data/raw/ or data/processed/
2. Queries the database for documented locations
3. Blocks the commit if undocumented data files are found
4. Provides clear guidance on how to update the manifest

Usage:
    This hook is called automatically by pre-commit.
    Manual: python scripts/hooks/check_data_manifest.py

Exit codes:
    0 - All data files are documented (or no data files staged)
    1 - Undocumented data files found (commit blocked)
"""

import re
import subprocess
import sys
from pathlib import Path

# Database configuration
DB_NAME = "cohort_projections_meta"

# Fallback to markdown if database not available
FALLBACK_MANIFEST = Path("data/DATA_MANIFEST.md")


def get_staged_files() -> list[str]:
    """Get list of staged files from git."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f.strip() for f in result.stdout.splitlines() if f.strip()]


def is_data_file(filepath: str) -> bool:
    """Check if file is in a data directory that should be tracked."""
    data_patterns = [
        r"^data/raw/",
        r"^data/processed/",
        r"^sdc_2024_replication/.*/data/",
    ]
    return any(re.match(pattern, filepath) for pattern in data_patterns)


def get_data_extensions() -> set[str]:
    """Extensions that indicate data files (vs scripts/docs)."""
    return {
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".json",
        ".dta",  # Stata
        ".sav",  # SPSS
        ".rds",  # R
        ".feather",
        ".pkl",
        ".pickle",
        ".h5",
        ".hdf5",
        ".nc",  # NetCDF
        ".pdf",  # Data PDFs (reports, tables)
        ".txt",  # Sometimes data
        ".tsv",
        ".dat",
    }


def should_check_file(filepath: str) -> bool:
    """Determine if this file should be checked against the manifest."""
    if not is_data_file(filepath):
        return False

    # Check extension
    ext = Path(filepath).suffix.lower()
    return ext in get_data_extensions()


def get_documented_locations_from_db() -> set[str] | None:
    """
    Query PostgreSQL for documented data locations.

    Returns None if database is not available (triggers fallback).
    """
    try:
        import psycopg2
    except ImportError:
        return None

    try:
        conn = psycopg2.connect(dbname=DB_NAME)
        cur = conn.cursor()

        cur.execute("SELECT location FROM data_sources")
        locations = {row[0].rstrip("/") for row in cur.fetchall()}

        cur.close()
        conn.close()
        return locations

    except psycopg2.OperationalError:
        # Database not available
        return None
    except psycopg2.ProgrammingError:
        # Table doesn't exist
        return None


def get_documented_locations_from_markdown() -> set[str]:
    """
    Fallback: Extract documented paths from markdown manifest.

    ONLY extracts paths from **Location** table fields.
    """
    if not FALLBACK_MANIFEST.exists():
        return set()

    manifest = FALLBACK_MANIFEST.read_text()
    documented = set()

    # Parse Location table rows
    location_matches = re.findall(r"\|\s*\*\*Location\*\*\s*\|\s*`([^`]+)`", manifest)
    for loc in location_matches:
        loc = loc.rstrip("/")
        if loc.startswith(("data/", "sdc_2024_replication/")):
            documented.add(loc)

    return documented


def get_documented_locations() -> tuple[set[str], str]:
    """
    Get documented locations, preferring database over markdown.

    Returns (locations, source) where source is 'database' or 'markdown'.
    """
    # Try database first
    db_locations = get_documented_locations_from_db()
    if db_locations is not None:
        return db_locations, "database"

    # Fallback to markdown
    md_locations = get_documented_locations_from_markdown()
    return md_locations, "markdown"


def check_file_documented(filepath: str, documented_paths: set[str]) -> bool:
    """
    Check if a file is covered by the manifest.

    A file is considered documented if its parent directory matches
    a documented location.
    """
    path = Path(filepath)

    # Check parent directories
    for parent in path.parents:
        parent_str = str(parent)
        if parent_str in documented_paths:
            return True

    # Check if any documented path is a prefix
    return any(filepath.startswith((doc_path + "/", doc_path)) for doc_path in documented_paths)


def format_guidance(undocumented: list[str], source: str) -> str:
    """Format helpful guidance for updating the manifest."""
    guidance = [
        "",
        "=" * 70,
        "DATA MANIFEST ENFORCEMENT - Commit Blocked",
        "=" * 70,
        "",
        f"Source: {source}",
        "",
        "The following data files are not documented:",
        "",
    ]

    for f in undocumented:
        guidance.append(f"  - {f}")

    if source == "database":
        guidance.extend(
            [
                "",
                "To fix this, add the data source to PostgreSQL:",
                "",
                "    psql -d cohort_projections_meta",
                "",
                "    INSERT INTO data_sources (",
                "        name, source_organization, format, location,",
                "        temporal_basis, category",
                "    ) VALUES (",
                "        'Your Data Source Name',",
                "        'Organization Name',",
                "        'csv',  -- or parquet, excel_xlsx, etc.",
                "        'data/raw/your_category/',",
                "        'calendar_year',  -- or fiscal_year, rolling_5year",
                "        'census_population'  -- or refugee_immigration, etc.",
                "    );",
                "",
                "Then regenerate the markdown docs:",
                "    python scripts/db/generate_manifest_docs.py",
                "",
            ]
        )
    else:
        guidance.extend(
            [
                "",
                "To fix this, update data/DATA_MANIFEST.md with:",
                "",
                "1. Add the data source to the appropriate category section",
                "2. Include the **Location** field with the file/directory path",
                "3. Document the **Temporal Basis** (FY, CY, or other)",
                "4. Update the **Changelog** at the bottom",
                "",
                "NOTE: Consider migrating to PostgreSQL for better manifest management:",
                "    python scripts/db/migrate_from_markdown.py",
                "",
            ]
        )

    guidance.append("=" * 70)
    return "\n".join(guidance)


def main() -> int:
    """Main hook logic."""
    # Get staged files
    try:
        staged = get_staged_files()
    except subprocess.CalledProcessError as e:
        print(f"Error getting staged files: {e}", file=sys.stderr)
        return 1

    # Filter to data files
    data_files = [f for f in staged if should_check_file(f)]

    if not data_files:
        return 0

    # Get documented locations
    documented, source = get_documented_locations()

    if not documented:
        print(
            f"Warning: No documented locations found ({source}). "
            "Cannot validate data file documentation.",
            file=sys.stderr,
        )
        print(format_guidance(data_files, source))
        return 1

    # Check each data file
    undocumented = [f for f in data_files if not check_file_documented(f, documented)]

    if undocumented:
        print(format_guidance(undocumented, source))
        return 1

    # All files documented
    print(f"✓ {len(data_files)} data file(s) verified against manifest ({source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
