#!/usr/bin/env python3
"""
Generate DATA_MANIFEST.md from PostgreSQL database.

This script queries the cohort_projections_meta database and generates
a human-readable markdown document. This is the canonical way to update
the manifest documentation after database changes.

Usage:
    python scripts/db/generate_manifest_docs.py
    python scripts/db/generate_manifest_docs.py --output docs/DATA_MANIFEST.md

The generated file is auto-generated and should not be edited directly.
Edit the database instead and re-run this script.
"""

import argparse
import datetime
from pathlib import Path

import psycopg2

# Database configuration
DB_NAME = "cohort_projections_meta"

# Output path
DEFAULT_OUTPUT = Path("data/DATA_MANIFEST.md")


def get_connection():
    """Get database connection."""
    return psycopg2.connect(dbname=DB_NAME)


def fetch_sources_by_category(conn) -> dict:
    """Fetch all data sources grouped by category."""
    cur = conn.cursor()

    cur.execute("""
        SELECT
            id, name, short_name, description,
            category, source_organization, source_url,
            format, location, temporal_basis,
            years_available, reference_date,
            alignment_notes, processing_script
        FROM data_sources
        ORDER BY category, name
    """)

    columns = [
        "id",
        "name",
        "short_name",
        "description",
        "category",
        "source_organization",
        "source_url",
        "format",
        "location",
        "temporal_basis",
        "years_available",
        "reference_date",
        "alignment_notes",
        "processing_script",
    ]

    sources_by_category = {}
    for row in cur.fetchall():
        source = dict(zip(columns, row, strict=False))
        cat = source["category"]
        if cat not in sources_by_category:
            sources_by_category[cat] = []
        sources_by_category[cat].append(source)

    cur.close()
    return sources_by_category


def fetch_changelog(conn) -> list[dict]:
    """Fetch changelog entries."""
    cur = conn.cursor()
    cur.execute("""
        SELECT version, change_date, changes, changed_by
        FROM manifest_changelog
        ORDER BY change_date DESC, id DESC
        LIMIT 20
    """)

    entries = []
    for version, change_date, changes, changed_by in cur.fetchall():
        entries.append(
            {
                "version": version,
                "date": change_date,
                "changes": changes,
                "changed_by": changed_by,
            }
        )

    cur.close()
    return entries


def fetch_temporal_alignments(conn) -> list[dict]:
    """Fetch temporal alignment rules."""
    cur = conn.cursor()
    cur.execute("""
        SELECT
            sa.name AS source_a,
            sb.name AS source_b,
            ta.alignment_issue,
            ta.handling_strategy
        FROM temporal_alignments ta
        JOIN data_sources sa ON ta.source_a_id = sa.id
        JOIN data_sources sb ON ta.source_b_id = sb.id
        ORDER BY sa.name, sb.name
    """)

    alignments = []
    for source_a, source_b, issue, strategy in cur.fetchall():
        alignments.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "issue": issue,
                "strategy": strategy,
            }
        )

    cur.close()
    return alignments


def format_temporal_basis(basis: str) -> str:
    """Format temporal basis for display."""
    mapping = {
        "fiscal_year": "**FISCAL YEAR (Oct 1 - Sep 30)**",
        "calendar_year": "**CALENDAR YEAR**",
        "tax_year": "**TAX YEAR / CALENDAR YEAR**",
        "rolling_5year": "**CALENDAR YEAR (5-year rolling)**",
        "point_in_time": "**Point-in-time**",
        "intercensal": "**CENSUS INTERCENSAL PERIODS**",
    }
    return mapping.get(basis, basis)


def format_format(fmt: str) -> str:
    """Format data format for display."""
    mapping = {
        "csv": "CSV",
        "parquet": "Parquet",
        "excel_xlsx": "Excel (.xlsx)",
        "excel_xls": "Excel (.xls)",
        "stata_dta": "Stata (.dta)",
        "json": "JSON",
        "pdf": "PDF",
        "spss_sav": "SPSS (.sav)",
        "r_rds": "R (.rds)",
        "other": "Other",
    }
    return mapping.get(fmt, fmt)


def format_category_header(cat: str) -> str:
    """Format category as section header."""
    mapping = {
        "refugee_immigration": "REFUGEE/IMMIGRATION DATA",
        "census_population": "CENSUS POPULATION DATA",
        "vital_statistics": "VITAL STATISTICS",
        "migration": "MIGRATION DATA",
        "geographic": "GEOGRAPHIC REFERENCE",
        "projections_source": "SDC 2024 PROJECTION SOURCE",
        "other": "OTHER DATA SOURCES",
    }
    return mapping.get(cat, cat.upper().replace("_", " "))


def generate_markdown(
    sources_by_category: dict,
    changelog: list[dict],
    alignments: list[dict],
) -> str:
    """Generate the full markdown document."""
    lines = [
        "# Data Manifest with Temporal Alignment Metadata",
        "",
        f"**Generated**: {datetime.datetime.now(tz=datetime.UTC).date().isoformat()}",
        "**Purpose**: Comprehensive inventory of all data sources with temporal (FY/CY) alignment information",
        "",
        "> **Note**: This file is auto-generated from the `cohort_projections_meta` PostgreSQL database.",
        "> Do not edit directly. Instead, update the database and run:",
        "> ```",
        "> python scripts/db/generate_manifest_docs.py",
        "> ```",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This project uses multiple data sources with **different temporal bases**:",
        "",
        "| Temporal Basis | Description | Impact |",
        "|----------------|-------------|--------|",
        "| **Fiscal Year (FY)** | Oct 1 - Sep 30 | Misaligns with Census CY by 3-9 months |",
        "| **Calendar Year (CY)** | Jan 1 - Dec 31 | Primary temporal basis for projections |",
        "| **Tax Year** | Same as CY | Reflects prior year income |",
        "| **5-Year Rolling** | ACS estimates | Centered on final year |",
        "",
        "---",
        "",
        "## Data Sources by Category",
        "",
    ]

    # Category order
    category_order = [
        "refugee_immigration",
        "census_population",
        "vital_statistics",
        "migration",
        "geographic",
        "projections_source",
        "other",
    ]

    section_num = 1
    for cat in category_order:
        sources = sources_by_category.get(cat, [])
        if not sources:
            continue

        lines.append(f"### {section_num}. {format_category_header(cat)}")
        lines.append("")

        subsection_num = 1
        for source in sources:
            lines.append(f"#### {section_num}.{subsection_num} {source['name']}")
            lines.append("")
            lines.append("| Attribute | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| **Source** | {source['source_organization']} |")
            lines.append(f"| **Format** | {format_format(source['format'])} |")
            lines.append(
                f"| **Temporal Basis** | {format_temporal_basis(source['temporal_basis'])} |"
            )

            if source["years_available"]:
                lines.append(f"| **Years Available** | {source['years_available']} |")

            lines.append(f"| **Location** | `{source['location']}` |")

            if source["processing_script"]:
                lines.append(f"| **Processing Script** | `{source['processing_script']}` |")

            if source["alignment_notes"]:
                lines.append(f"| **Alignment Notes** | {source['alignment_notes']} |")

            lines.append("")
            lines.append("---")
            lines.append("")
            subsection_num += 1

        section_num += 1

    # Temporal alignment matrix
    if alignments:
        lines.append("## Temporal Alignment Matrix")
        lines.append("")
        lines.append("| Source A | Source B | Alignment Issue | Handling Strategy |")
        lines.append("|----------|----------|-----------------|-------------------|")
        for a in alignments:
            lines.append(f"| {a['source_a']} | {a['source_b']} | {a['issue']} | {a['strategy']} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Changelog
    lines.append("## Changelog")
    lines.append("")
    lines.append("| Date | Version | Changes |")
    lines.append("|------|---------|---------|")
    for entry in changelog:
        lines.append(f"| {entry['date']} | {entry['version']} | {entry['changes']} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*This manifest is auto-generated. Update the database, not this file.*")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate DATA_MANIFEST.md from PostgreSQL")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Generating DATA_MANIFEST.md from PostgreSQL")
    print("=" * 60)

    # Connect to database
    print(f"\nConnecting to {DB_NAME}...")
    conn = get_connection()

    # Fetch data
    print("Fetching data sources...")
    sources = fetch_sources_by_category(conn)
    total_sources = sum(len(s) for s in sources.values())
    print(f"  Found {total_sources} sources across {len(sources)} categories")

    print("Fetching changelog...")
    changelog = fetch_changelog(conn)
    print(f"  Found {len(changelog)} changelog entries")

    print("Fetching temporal alignments...")
    alignments = fetch_temporal_alignments(conn)
    print(f"  Found {len(alignments)} alignment rules")

    conn.close()

    # Generate markdown
    print("\nGenerating markdown...")
    markdown = generate_markdown(sources, changelog, alignments)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    print(f"  Written to: {args.output}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
