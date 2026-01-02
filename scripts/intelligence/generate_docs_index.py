#!/usr/bin/env python3
"""
Generate Documentation Index
============================

Generates a unified `docs/INDEX.md` from the repository intelligence database.
Organizes documentation by function and currency status.

Usage:
    python scripts/intelligence/generate_docs_index.py
"""

from pathlib import Path

import psycopg2

DB_NAME = "cohort_projections_meta"
OUTPUT_FILE = Path("docs/INDEX.md")


def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME)


def generate_index():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch all documentation links
    cur.execute("""
        SELECT
            ci.filepath,
            ci.description,
            ci.function_tag,
            dl.doc_filepath,
            dl.relationship_type,
            dl.currency_status
        FROM documentation_links dl
        JOIN code_inventory ci ON dl.code_id = ci.id
        WHERE ci.status = 'active'
        ORDER BY ci.function_tag, ci.filepath
    """)
    rows = cur.fetchall()

    conn.close()

    # Organize by function -> list of docs
    by_function = {}

    for row in rows:
        filepath, desc, tag, doc_path, rel_type, status = row

        if tag not in by_function:
            by_function[tag] = []

        by_function[tag].append(
            {"code": filepath, "desc": desc, "doc": doc_path, "type": rel_type, "status": status}
        )

    # Generate Markdown
    lines = [
        "# Unified Documentation Index",
        "",
        "**Auto-generated from PostgreSQL. Do not edit.**",
        "",
        "## Overview",
        "This index provides a centralized view of all documentation linked to active code.",
        "",
    ]

    # Order tags logically
    order = ["source_code", "data_artifact", "configuration", "automation", "documentation", "test"]

    for tag in order:
        if tag not in by_function:
            continue

        items = by_function[tag]
        lines.append(f"## {tag.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Code File | Documentation | Type | Status |")
        lines.append("|-----------|---------------|------|--------|")

        for item in items:
            code_link = f"[`{item['code']}`](../{item['code']})"
            doc_link = f"[`{Path(item['doc']).name}`](../{item['doc']})"
            status_icon = "🟢" if item["status"] == "current" else "🔴"

            lines.append(
                f"| {code_link} | {doc_link} | {item['type']} | {status_icon} {item['status']} |"
            )

        lines.append("")

    # Ensure docs dir exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"Generated {OUTPUT_FILE} ({len(rows)} links)")


if __name__ == "__main__":
    generate_index()
