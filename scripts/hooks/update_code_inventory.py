#!/usr/bin/env python3
"""
Pre-commit Hook: Update Code Inventory
======================================

Incrementally updates the PostgreSQL 'code_inventory' table based on staged changes.
- Added/Modified files -> Upsert to DB as 'active' with updated timestamp
- Deleted files -> Mark as 'deprecated' (or 'gone')

This ensures the database stays in sync with the file system without requiring a full scan.
"""

import subprocess
import sys
from pathlib import Path

import psycopg2

# Reuse logic from scanner if possible, or duplicate for speed/independence
# Duplicating minimal logic to keep hook fast and standalone
DB_NAME = "cohort_projections_meta"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_db_connection():
    try:
        return psycopg2.connect(dbname=DB_NAME)
    except psycopg2.OperationalError:
        return None


def get_staged_changes():
    """Returns lists of (added/modified, deleted) files."""
    # Added or Modified
    result_am = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True,
        text=True,
    )
    added_modified = [f for f in result_am.stdout.splitlines() if f.strip()]

    # Deleted
    result_d = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=D"],
        capture_output=True,
        text=True,
    )
    deleted = [f for f in result_d.stdout.splitlines() if f.strip()]

    return added_modified, deleted


def determine_file_type(filepath):
    return Path(filepath).suffix.lstrip(".") or "txt"


def update_inventory():
    conn = get_db_connection()
    if not conn:
        print(
            "Warning: Could not connect to cohort_projections_meta DB. Skipping inventory update."
        )
        return 0

    cur = conn.cursor()

    added_modified, deleted = get_staged_changes()

    if not added_modified and not deleted:
        return 0

    print(f"Inventory Update: {len(added_modified)} added/mod, {len(deleted)} deleted")

    # Process Added/Modified
    for rel_path in added_modified:
        # We don't extract full description here to keep it fast,
        # or we could do a quick read. Let's do quick read.
        try:
            # Note: The file is staged, so it exists on disk usually, unless it's a rename weirdness
            # But pre-commit runs on the working tree mostly?
            # Actually pre-commit stashes changes not in index.
            # So the file on disk should match what is being committed?
            # Not strictly true for partial add, but close enough for metadata.

            fpath = PROJECT_ROOT / rel_path
            if fpath.exists():
                # Simple function tag heuristic
                func_tag = "source_code"
                if "test" in rel_path:
                    func_tag = "test"
                elif "docs" in rel_path:
                    func_tag = "documentation"

                cur.execute(
                    """
                    INSERT INTO code_inventory (filepath, file_type, function_tag, status, last_validated, updated_at)
                    VALUES (%s, %s, %s, 'active', NOW(), NOW())
                    ON CONFLICT (filepath) DO UPDATE SET
                        status = 'active',
                        last_validated = NOW(),
                        updated_at = NOW();
                """,
                    (rel_path, determine_file_type(rel_path), func_tag),
                )
        except Exception as e:
            print(f"Warning: Failed to update {rel_path}: {e}")

    # Process Deleted
    for rel_path in deleted:
        # Check if it was moved to archive (which would appear as a delete here + add elsewhere)
        # If it's a pure delete, mark deprecated using Type II logic (valid_to is handled by audit triggers or defaulting?)
        # For now, just setting status = 'deprecated'.
        cur.execute(
            """
            UPDATE code_inventory
            SET status = 'deprecated', updated_at = NOW()
            WHERE filepath = %s
        """,
            (rel_path,),
        )

    conn.commit()
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(update_inventory())
