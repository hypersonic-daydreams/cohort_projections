#!/usr/bin/env python3
"""
Archive Manager
===============

Moves files marked as 'deprecated' in the PostgreSQL database to the 'archive/' directory.
Preserves original directory structure to avoid filename collisions.

Usage:
    python scripts/intelligence/archive_manager.py
"""

import argparse
import shutil
from pathlib import Path

import psycopg2

# Configuration
DB_NAME = "cohort_projections_meta"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = PROJECT_ROOT / "archive"


def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME)


def archive_files(dry_run: bool = False):
    conn = get_db_connection()
    cur = conn.cursor()

    # Find files that are deprecated in DB but still exist in main tree
    cur.execute("""
        SELECT filepath FROM code_inventory
        WHERE status = 'deprecated'
    """)
    candidates = [row[0] for row in cur.fetchall()]

    moved_count = 0

    print(f"Checking {len(candidates)} deprecated files for archiving...")

    for rel_path in candidates:
        source_path = PROJECT_ROOT / rel_path

        # Skip if already moved (source doesn't exist)
        if not source_path.exists():
            continue

        # Determine destination
        dest_path = ARCHIVE_ROOT / rel_path

        if dry_run:
            print(f"[DRY RUN] Move {rel_path} -> archive/{rel_path}")
        else:
            print(f"Archiving: {rel_path}")

            # Ensure parent dir exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.move(str(source_path), str(dest_path))
                moved_count += 1

                # Update DB to reflect 'archived' status (optional, but good for clarity)
                # Actually, if we move it, the scanner will mark it as missing next time?
                # No, scanner checks if file exists. If it doesn't, it marks 'deprecated'.
                # We should probably have an 'archived' status.

                cur.execute(
                    """
                    UPDATE code_inventory
                    SET status = 'archived', updated_at = NOW()
                    WHERE filepath = %s
                """,
                    (rel_path,),
                )

            except Exception as e:
                print(f"Error moving {rel_path}: {e}")

    conn.commit()
    conn.close()

    if not dry_run:
        print(f"\nArchived {moved_count} files.")
        if moved_count > 0:
            print(f"Files moved to: {ARCHIVE_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not move files")
    args = parser.parse_args()

    archive_files(dry_run=args.dry_run)
