import sys
from pathlib import Path

import psycopg2

DB_NAME = "cohort_projections_meta"


def apply_migration():
    try:
        conn = psycopg2.connect(dbname=DB_NAME)
        conn.autocommit = True
        cur = conn.cursor()

        migration_file = Path("scripts/migrations/001_add_governance_inventory.sql")
        print(f"Applying {migration_file}...")

        cur.execute(migration_file.read_text())

        print("Migration successful.")
        conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    apply_migration()
