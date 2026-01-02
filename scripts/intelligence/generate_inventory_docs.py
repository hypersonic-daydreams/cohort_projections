#!/usr/bin/env python3
"""
Generate Repository Inventory Docs
==================================

Generates a markdown inventory of the codebase from the 'code_inventory' table.
This file serves as the primary map for AI agents to understand the repository structure.

Usage:
    python scripts/intelligence/generate_inventory_docs.py
"""

from pathlib import Path

import psycopg2

DB_NAME = "cohort_projections_meta"
OUTPUT_FILE = Path("REPOSITORY_INVENTORY.md")


def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME)


def generate_inventory():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch summary stats
    cur.execute("SELECT status, COUNT(*) FROM code_inventory GROUP BY status")
    stats = dict(cur.fetchall())

    # Fetch all active files grouped by function
    cur.execute("""
        SELECT function_tag, filepath, description
        FROM code_inventory
        WHERE status = 'active'
        ORDER BY function_tag, filepath
    """)
    rows = cur.fetchall()

    conn.close()

    # Organize by tag
    by_tag: dict[str, list[tuple[str, str]]] = {}
    for tag, path, desc in rows:
        if tag not in by_tag:
            by_tag[tag] = []
        by_tag[tag].append((path, desc))

    # Generate Markdown
    lines = [
        "# Repository Intelligence Inventory",
        "",
        "**Auto-generated from PostgreSQL. Do not edit.**",
        "",
        "## Overview",
        "",
        f"- **Active Files**: {stats.get('active', 0)}",
        f"- **Deprecated**: {stats.get('deprecated', 0)}",
        f"- **Archived**: {stats.get('archived', 0)}",
        "",
        "## Active Codebase",
        "",
    ]

    # Order tags logically
    order = ["source_code", "data_artifact", "configuration", "automation", "documentation", "test"]

    for tag in order:
        if tag not in by_tag:
            continue

        files = by_tag[tag]
        lines.append(f"### {tag.replace('_', ' ').title()} ({len(files)})")
        lines.append("")
        lines.append("| File | Description |")
        lines.append("|------|-------------|")

        for path, desc in files:
            desc_str = (desc or "").replace("|", r"\|").strip()
            lines.append(f"| `[{path}]({path})` | {desc_str} |")

        lines.append("")

    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"Generated {OUTPUT_FILE} ({len(rows)} active files)")


if __name__ == "__main__":
    generate_inventory()
