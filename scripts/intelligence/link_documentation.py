#!/usr/bin/env python3
"""
Documentation Linker
====================

Scans the repository to infer relationships between code and documentation.
Populates the 'documentation_links' table.

Heuristics:
1. README.md in the same directory -> User Guide / Overview
2. Explicit "See: [path]" or "Docs: [path]" strings in descriptions/comments (TODO)
3. Matching filename structure (e.g., FOO.py <-> docs/FOO.md) (TODO)

Usage:
    python scripts/intelligence/link_documentation.py
"""

from pathlib import Path

import psycopg2

DB_NAME = "cohort_projections_meta"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME)


def link_documentation():
    conn = get_db_connection()
    cur = conn.cursor()

    print("Inferring documentation links...")

    # 1. Link README.md in same directory
    # Find all active files
    cur.execute("SELECT id, filepath FROM code_inventory WHERE status = 'active'")
    files = cur.fetchall()

    new_links = 0

    for file_id, rel_path_str in files:
        rel_path = Path(rel_path_str)
        parent_dir = rel_path.parent

        # Check for README.md in same dir
        readme_candidates = [
            parent_dir / "README.md",
            parent_dir / "readme.md",
            parent_dir / "Readme.md",
        ]

        found_doc = None
        for candidate in readme_candidates:
            if (PROJECT_ROOT / candidate).exists():
                found_doc = str(candidate)
                break

        # Avoid self-linking (if the file itself is the README)
        if found_doc and found_doc != rel_path_str:
            # Check if link already exists to avoid duplicates
            cur.execute(
                """
                INSERT INTO documentation_links (code_id, doc_filepath, relationship_type)
                SELECT %s, %s, 'directory_readme'
                WHERE NOT EXISTS (
                    SELECT 1 FROM documentation_links
                    WHERE code_id = %s AND doc_filepath = %s
                )
            """,
                (file_id, found_doc, file_id, found_doc),
            )

            if cur.rowcount > 0:
                new_links += 1

    conn.commit()
    conn.close()

    print(f"Linked {new_links} files to their directory READMEs.")


if __name__ == "__main__":
    link_documentation()
