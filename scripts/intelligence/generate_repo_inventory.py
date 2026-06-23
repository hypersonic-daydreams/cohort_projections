#!/usr/bin/env python3
"""
Generate a repo-scoped inventory by querying the workspace ledger (SCAFFOLD).

Created: 2026-06-23
Author: Claude Code (Opus 4.8)
Decision: Implements Decision 2 of docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md.
Spec: docs/plans/ledger-inventory-refactor.md (Status: planned).

Purpose
-------
This script replaces the dead bespoke "Repository Intelligence System" (the
``cohort_projections_meta`` PostgreSQL DB and the stale ``REPOSITORY_INVENTORY.md``,
whose generator chain is orphaned). Instead of re-implementing a repo scanner, it
sources the inventory from the **workspace ledger** (``workspace_silver.ledger``),
filtered to this repo, using ``asset_status`` + ``v_stale_assets`` for the
current-vs-archived signal.

This file is a SCAFFOLD. The live ledger query is intentionally NOT wired up yet:
per docs/plans/ledger-inventory-refactor.md the work is blocked on confirming the
ledger service is live on this machine AND already holds fresh ``cohort_projections``
rows. Until then the script only PROBES the ledger and degrades gracefully:

  * ledger UNREACHABLE -> print a clear message and exit 0 (never crash a commit/build)
  * ledger REACHABLE   -> emit a date-stamped, advisory PLACEHOLDER output file

Deferred-wiring note
--------------------
The real query (filter ``repo = 'cohort_projections'`` -- the ledger's repo column
holds the SHORT name, not ``demography/cohort_projections``; classify each asset via
``ledger.asset_status`` + ``ledger.v_stale_assets``; aggregate by ``asset_type`` and
status) is deferred to the unblocked pass. Correctness of the live query is OUT OF
SCOPE here. See the spec's task list, steps 3-7.

Method
------
1. Probe the ledger in preference order, each guarded so a missing dependency or a
   down service is a soft failure: (a) HTTP API at http://127.0.0.1:8765,
   (b) ``workspace ledger`` CLI on PATH, (c) read-only psycopg2 SELECT (optional dep).
2. If no method reaches the ledger, print guidance and exit 0.
3. If any method reaches it, write a placeholder advisory markdown file noting it is
   ledger-sourced and date-stamped per the per-machine caveat.

Key design decisions
--------------------
- **Dependency-light**: stdlib only on the happy path. ``requests`` and ``psycopg2``
  are optional and import-guarded; their absence just skips that probe method.
- **Never hard-fail**: the ledger is per-machine and may be down
  (docs/REFERENCE-cross-machine-db-state.md); a missing service must not break a
  commit or build. Hence exit 0 on unreachable.
- **Advisory stamp**: any emitted snapshot is marked machine-specific and dated,
  because Postgres state is not bisynced across machines.
- **Consumer-only**: this repo reads the ledger; it never writes to or owns it.

Inputs
------
- The workspace ledger (``workspace_silver.ledger``), owned by the command-center
  repo. Reached via HTTP (127.0.0.1:8765), the ``workspace ledger`` CLI, or a
  read-only Postgres connection. No local input files.

Output
------
- docs/REPO_INVENTORY.generated.md
    Written ONLY when the ledger is reachable. In this scaffold pass it is a
    date-stamped advisory PLACEHOLDER, not a real inventory.

Usage
-----
    python scripts/intelligence/generate_repo_inventory.py
"""

from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

# --- Constants -------------------------------------------------------------

# The ledger's `repo` COLUMN holds the short name (see SPEC-scanner-cohort-projections.md);
# the relative-path form `demography/cohort_projections` is `repo_path`/REPO_REL_PATH.
LEDGER_REPO = "cohort_projections"
LEDGER_API = "http://127.0.0.1:8765"
PROBE_PATH = "/find?q=__inventory_probe__"
HTTP_TIMEOUT_S = 3.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "docs" / "REPO_INVENTORY.generated.md"


# --- Ledger probes (each returns a short reachability note or None) ---------


def _probe_http() -> str | None:
    """Probe the ledger HTTP API. Return a note if reachable, else None."""
    url = f"{LEDGER_API}{PROBE_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S):  # noqa: S310 (localhost only)
            return f"HTTP API at {LEDGER_API}"
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _probe_cli() -> str | None:
    """Probe the `workspace ledger` CLI. Return a note if reachable, else None."""
    if shutil.which("workspace") is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 (fixed argv, no shell)
            ["workspace", "ledger", "query", "health"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0:
        return "workspace ledger CLI"
    return None


def _probe_db() -> str | None:
    """Probe a read-only Postgres connection to the ledger. Optional dependency."""
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        conn = psycopg2.connect(dbname="workspace_silver", connect_timeout=3)
    except Exception:  # noqa: BLE001 (any connection failure is a soft skip)
        return None
    try:
        with conn.cursor() as cur:
            # Existence check only; the real SELECT is deferred (see module docstring).
            cur.execute("SELECT to_regclass('ledger.assets')")
            row = cur.fetchone()
        reachable = bool(row and row[0])
    except Exception:  # noqa: BLE001
        reachable = False
    finally:
        conn.close()
    return "Postgres workspace_silver.ledger" if reachable else None


def reach_ledger() -> str | None:
    """Try each access method in preference order. Return the first that works."""
    for probe in (_probe_http, _probe_cli, _probe_db):
        note = probe()
        if note is not None:
            return note
    return None


# --- Output ----------------------------------------------------------------


def _placeholder_markdown(source_note: str, today: str) -> str:
    """Build the date-stamped advisory placeholder body."""
    return f"""# Repository Inventory (ledger-sourced) — PLACEHOLDER

> Generated from workspace ledger on {today} — advisory, may be machine-specific.
> Source reached via: {source_note}
> Repo filter: `repo = '{LEDGER_REPO}'`

**This is a scaffold placeholder, not a real inventory.** The live ledger query is
deferred until the ledger service is confirmed live on this machine and indexing this
repo. See `docs/plans/ledger-inventory-refactor.md` (Status: planned) for the spec and
task list, and `scripts/intelligence/generate_repo_inventory.py` for the generator.

When wired up, this file will list this repo's assets classified current-vs-archived
via `ledger.asset_status` + `ledger.v_stale_assets`, aggregated by `asset_type`. It is
an advisory navigation aid only; `DEVELOPMENT_TRACKER.md` and `AGENTS.md` remain
canonical.
"""


def main() -> int:
    """Probe the ledger; emit a placeholder if reachable, else no-op. Always exit 0."""
    source_note = reach_ledger()

    if source_note is None:
        print(
            "Workspace ledger not reachable on this machine "
            f"(tried HTTP {LEDGER_API}, the `workspace ledger` CLI, and a read-only "
            "Postgres connection to workspace_silver.ledger).\n"
            "No inventory written. This is expected when the ledger service is down; "
            "the ledger is per-machine and not bisynced.\n"
            "See docs/plans/ledger-inventory-refactor.md (Status: planned) to unblock."
        )
        return 0

    today = _dt.datetime.now(tz=_dt.UTC).date().isoformat()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(_placeholder_markdown(source_note, today), encoding="utf-8")
    print(
        f"Ledger reachable via {source_note}.\n"
        f"Wrote advisory PLACEHOLDER to {OUTPUT_FILE.relative_to(PROJECT_ROOT)} "
        f"(stamped {today}).\n"
        "NOTE: scaffold only — the live ledger query is not yet wired up "
        "(see docs/plans/ledger-inventory-refactor.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
