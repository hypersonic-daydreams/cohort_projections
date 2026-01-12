"""Pre-commit hook: best-effort repository inventory refresh.

This wrapper is intentionally conservative: if the metadata database is not available
in the current environment, the hook exits successfully (and prints a short message)
instead of blocking commits.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Update repository code inventory (best effort).")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: .)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    scan_script = project_root / "scripts" / "intelligence" / "scan_repository.py"
    if not scan_script.exists():
        return 0

    # Heuristic: if no DB env vars are set, do not attempt to run the scanner.
    db_env_vars = ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER"]
    if not any(os.getenv(k) for k in db_env_vars):
        return 0

    try:
        completed = subprocess.run(
            [sys.executable, str(scan_script)],
            cwd=str(project_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return 0

    # Do not hard-fail pre-commit on inventory refresh; treat it as best-effort.
    return 0 if completed.returncode == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
