#!/usr/bin/env python3
"""
Repository Intelligence Scanner
===============================

Scans the repository to populate the 'code_inventory' table in PostgreSQL.
Features:
- Respects .gitignore via 'pathspec'
- Extracts docstrings/summaries from files
- Registers new files as 'active'
- Updates 'last_validated' timestamp
- Detects deleted files and marks them as 'deprecated' (or 'missing')

Usage:
    python scripts/intelligence/scan_repository.py
"""

import argparse
import ast
import os
from pathlib import Path

import pathspec
import psycopg2

# Configuration
DB_NAME = "cohort_projections_meta"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_db_connection():
    return psycopg2.connect(dbname=DB_NAME)


def load_gitignore_spec(root: Path) -> pathspec.PathSpec:
    """Load .gitignore patterns."""
    gitignore = root / ".gitignore"
    patterns = []
    if gitignore.exists():
        with open(gitignore) as f:
            patterns = f.readlines()

    # Always ignore .git
    patterns.append(".git/")

    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def extract_description(file_path: Path) -> str | None:
    """Extract docstring or top comments from a file."""
    try:
        content = file_path.read_text(errors="ignore")

        if file_path.suffix == ".py":
            try:
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    return docstring.strip().split("\n")[0]  # First line only
            except SyntaxError:
                pass

        # Fallback: Read first few lines for comments
        lines = content.splitlines()
        for line in lines[:5]:
            line = line.strip()
            if line.startswith(("#", "//")):
                return line.lstrip("#/ ").strip()

    except Exception:
        return None

    return None


def determine_function_tag(file_path: Path) -> str:
    """Heuristic to determine function tag."""
    path_str = str(file_path).lower()

    if "test" in path_str:
        return "test"
    if "scripts" in path_str:
        return "automation"
    if "docs" in path_str:
        return "documentation"
    if "data/processed" in path_str:
        return "data_artifact"
    if "config" in path_str or file_path.suffix in [".toml", ".yaml", ".json"]:
        return "configuration"

    return "source_code"


def scan_repository(dry_run: bool = False):
    conn = get_db_connection()
    cur = conn.cursor()

    spec = load_gitignore_spec(PROJECT_ROOT)

    print(f"Scanning {PROJECT_ROOT}...")

    found_files = set()

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Determine relative path for .gitignore matching
        rel_root = Path(root).relative_to(PROJECT_ROOT)

        # Filter directories in-place
        dirs[:] = [d for d in dirs if not spec.match_file(str(rel_root / d))]

        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(PROJECT_ROOT)
            rel_path_str = str(rel_path)

            if spec.match_file(rel_path_str):
                continue

            found_files.add(rel_path_str)
            description = extract_description(file_path)
            func_tag = determine_function_tag(file_path)
            file_type = file_path.suffix.lstrip(".") or "txt"

            if not dry_run:
                # Upsert file record
                cur.execute(
                    """
                    INSERT INTO code_inventory (filepath, file_type, function_tag, status, last_validated, description)
                    VALUES (%s, %s, %s, 'active', NOW(), %s)
                    ON CONFLICT (filepath) DO UPDATE SET
                        last_validated = NOW(),
                        description = COALESCE(EXCLUDED.description, code_inventory.description),
                        updated_at = NOW()
                    RETURNING id, status;
                """,
                    (rel_path_str, file_type, func_tag, description),
                )

                cur.fetchone()

    # Detect deleted files
    if not dry_run:
        print("\nChecking for deleted files...")
        cur.execute("SELECT filepath FROM code_inventory WHERE status = 'active'")
        db_files = {row[0] for row in cur.fetchall()}

        missing_files = db_files - found_files

        for missing in missing_files:
            print(f"  Marking as missing: {missing}")
            cur.execute(
                """
                UPDATE code_inventory
                SET status = 'deprecated', updated_at = NOW()
                WHERE filepath = %s
            """,
                (missing,),
            )

    conn.commit()
    conn.close()

    print(f"\nScan complete. Found {len(found_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Do not modify database")
    args = parser.parse_args()

    scan_repository(dry_run=args.dry_run)
