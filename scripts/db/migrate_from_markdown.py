#!/usr/bin/env python3
"""
Migrate data manifest from markdown to PostgreSQL.

This script:
1. Parses the existing DATA_MANIFEST.md
2. Creates the cohort_projections_meta database if it doesn't exist
3. Applies the schema
4. Populates the data_sources table with parsed entries

Usage:
    python scripts/db/migrate_from_markdown.py [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "DATA_MANIFEST.md"
SCHEMA_PATH = PROJECT_ROOT / "scripts" / "db" / "schema.sql"

# Database configuration
DB_NAME = "cohort_projections_meta"


def parse_manifest() -> list[dict]:
    """
    Parse the markdown manifest into structured data.

    Returns list of dicts with keys matching data_sources columns.
    """
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    content = MANIFEST_PATH.read_text()
    sources = []

    # Map markdown temporal basis to enum values
    temporal_map = {
        "fiscal year": "fiscal_year",
        "calendar year": "calendar_year",
        "tax year": "tax_year",
        "5-year rolling": "rolling_5year",
        "point-in-time": "point_in_time",
        "census intercensal": "intercensal",
    }

    # Parse each data source section
    # Format: #### X.X Source Name followed by table
    import re

    sections = re.split(r"\n#### \d+\.\d+ ", content)[1:]  # Skip before first ####

    for section in sections:
        lines = section.strip().split("\n")
        if not lines:
            continue

        name = lines[0].strip()
        source_data = {"name": name}

        # Parse table rows
        for line in lines:
            if "|" not in line or "---" in line:
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 2:
                continue

            key = parts[0].replace("**", "").strip().lower()
            value = parts[1].replace("`", "").strip()

            if key == "source":
                source_data["source_organization"] = value
            elif key == "format":
                # Map to enum
                fmt_lower = value.lower()
                if "csv" in fmt_lower:
                    source_data["format"] = "csv"
                elif "parquet" in fmt_lower:
                    source_data["format"] = "parquet"
                elif "xlsx" in fmt_lower:
                    source_data["format"] = "excel_xlsx"
                elif "xls" in fmt_lower:
                    source_data["format"] = "excel_xls"
                elif "stata" in fmt_lower or "dta" in fmt_lower:
                    source_data["format"] = "stata_dta"
                elif "pdf" in fmt_lower:
                    source_data["format"] = "pdf"
                elif "json" in fmt_lower:
                    source_data["format"] = "json"
                else:
                    source_data["format"] = "other"
            elif key == "temporal basis":
                # Extract temporal basis
                for md_key, db_val in temporal_map.items():
                    if md_key in value.lower():
                        source_data["temporal_basis"] = db_val
                        break
                else:
                    source_data["temporal_basis"] = "calendar_year"  # default
            elif key == "years available":
                source_data["years_available"] = value
            elif key == "location":
                source_data["location"] = value
            elif key == "processing script":
                source_data["processing_script"] = value
            elif key == "alignment notes":
                source_data["alignment_notes"] = value

        # Only add if we have required fields
        if all(k in source_data for k in ["name", "location", "format"]):
            # Set defaults
            source_data.setdefault("temporal_basis", "calendar_year")
            source_data.setdefault("source_organization", "Unknown")
            source_data.setdefault("category", "other")

            # Infer category from location
            loc = source_data["location"].lower()
            if "immigration" in loc or "refugee" in loc:
                source_data["category"] = "refugee_immigration"
            elif "population" in loc:
                source_data["category"] = "census_population"
            elif "fertility" in loc or "mortality" in loc:
                source_data["category"] = "vital_statistics"
            elif "migration" in loc:
                source_data["category"] = "migration"
            elif "geographic" in loc:
                source_data["category"] = "geographic"
            elif "sdc" in loc or "projection" in loc:
                source_data["category"] = "projections_source"

            sources.append(source_data)
            print(f"  Parsed: {source_data['name']}")

    return sources


def create_database():
    """Create the database if it doesn't exist."""
    # Connect to default postgres database
    conn = psycopg2.connect(dbname="postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
    exists = cur.fetchone()

    if not exists:
        print(f"Creating database: {DB_NAME}")
        cur.execute(f"CREATE DATABASE {DB_NAME}")
    else:
        print(f"Database {DB_NAME} already exists")

    cur.close()
    conn.close()


def apply_schema():
    """Apply the SQL schema to the database."""
    print(f"Applying schema from {SCHEMA_PATH}")

    # Use psql for schema application (handles CREATE TYPE etc better)
    result = subprocess.run(
        ["psql", "-d", DB_NAME, "-f", str(SCHEMA_PATH)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Check if it's just "already exists" errors (idempotent)
        if "already exists" in result.stderr:
            print("  Schema already applied (idempotent)")
        else:
            print(f"Schema error: {result.stderr}")
            return False

    return True


def insert_sources(sources: list[dict], dry_run: bool = False):
    """Insert parsed sources into the database."""
    if dry_run:
        print("\n[DRY RUN] Would insert the following sources:")
        for s in sources:
            print(f"  - {s['name']} ({s['location']})")
        return

    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()

    insert_sql = """
    INSERT INTO data_sources (
        name, source_organization, format, location,
        temporal_basis, years_available, alignment_notes,
        processing_script, category
    ) VALUES (
        %(name)s, %(source_organization)s, %(format)s, %(location)s,
        %(temporal_basis)s, %(years_available)s, %(alignment_notes)s,
        %(processing_script)s, %(category)s
    )
    ON CONFLICT (name) DO UPDATE SET
        source_organization = EXCLUDED.source_organization,
        format = EXCLUDED.format,
        location = EXCLUDED.location,
        temporal_basis = EXCLUDED.temporal_basis,
        years_available = EXCLUDED.years_available,
        alignment_notes = EXCLUDED.alignment_notes,
        processing_script = EXCLUDED.processing_script,
        category = EXCLUDED.category,
        updated_at = CURRENT_TIMESTAMP
    """

    for source in sources:
        # Ensure all fields exist
        source.setdefault("years_available", None)
        source.setdefault("alignment_notes", None)
        source.setdefault("processing_script", None)

        try:
            cur.execute(insert_sql, source)
            print(f"  Inserted/updated: {source['name']}")
        except Exception as e:
            print(f"  ERROR inserting {source['name']}: {e}")

    conn.commit()
    cur.close()
    conn.close()


def insert_changelog(version: str = "1.0.0"):
    """Insert initial changelog entry."""
    conn = psycopg2.connect(dbname=DB_NAME)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO manifest_changelog (version, changes)
        VALUES (%s, %s)
        ON CONFLICT DO NOTHING
        """,
        (version, "Initial migration from DATA_MANIFEST.md"),
    )

    conn.commit()
    cur.close()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate manifest to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, don't insert")
    args = parser.parse_args()

    print("=" * 60)
    print("Migrating DATA_MANIFEST.md to PostgreSQL")
    print("=" * 60)

    # Step 1: Parse markdown
    print("\n[1/4] Parsing markdown manifest...")
    sources = parse_manifest()
    print(f"  Found {len(sources)} data sources")

    if args.dry_run:
        print("\n[DRY RUN] Parsed sources:")
        for s in sources:
            print(f"  - {s['name']}")
            print(f"      Location: {s.get('location', 'N/A')}")
            print(f"      Temporal: {s.get('temporal_basis', 'N/A')}")
            print(f"      Category: {s.get('category', 'N/A')}")
        return

    # Step 2: Create database
    print("\n[2/4] Creating database...")
    create_database()

    # Step 3: Apply schema
    print("\n[3/4] Applying schema...")
    if not apply_schema():
        print("Failed to apply schema")
        sys.exit(1)

    # Step 4: Insert data
    print("\n[4/4] Inserting data sources...")
    insert_sources(sources)
    insert_changelog()

    print("\n" + "=" * 60)
    print("Migration complete!")
    print(f"Database: {DB_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
