# Repository Intelligence Scripts

| Attribute | Value |
|-----------|-------|
| **Status** | transitional — bespoke system retired 2026-06-23; ledger-backed replacement is a scaffold |
| **Last Updated** | 2026-06-23 |

## What changed (2026-06-23)

The bespoke per-repo "Repository Intelligence System" — a local `cohort_projections_meta`
PostgreSQL database plus auto-generated `REPOSITORY_INVENTORY.md` and `docs/INDEX.md` — was
**retired**. It had gone stale (the inventory claimed 0 archived files while 200+ existed) and was
already broken: three of its referenced scripts (`scan_repository.py`, `archive_manager.py`,
`generate_inventory_docs.py`) no longer existed.

Per Decision 2 of [`docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md`](../../docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md),
the repo inventory is being refactored onto the **workspace ledger** (`workspace_silver.ledger`),
which already indexes this repo and tracks asset status (current / stale / archived). See the
implementation spec: [`docs/plans/ledger-inventory-refactor.md`](../../docs/plans/ledger-inventory-refactor.md).

Removed in the retirement: `REPOSITORY_INVENTORY.md`, `docs/INDEX.md`,
`generate_docs_index.py`, `link_documentation.py`, the `update-code-inventory` pre-commit hook, and the
`update-code-inventory` console entry point. The `codebase_catalog` package and its **separate**
`check-data-manifest` hook are retained.

## Scripts

- **`generate_repo_inventory.py`** — scaffold generator that queries the workspace ledger
  (HTTP API → `workspace ledger` CLI → read-only Postgres) for assets scoped to this repo and writes a
  date-stamped, advisory `docs/REPO_INVENTORY.generated.md`. It **no-ops with exit 0** if the ledger is
  unreachable. The generated file is gitignored (machine-specific). The live query is not yet wired for
  correctness — see the spec for the remaining work.

## For navigation

Until the ledger generator is finished, use the hand-maintained map:
[`docs/NAVIGATION.md`](../../docs/NAVIGATION.md) (directory map + entry-point hierarchy),
[`DEVELOPMENT_TRACKER.md`](../../DEVELOPMENT_TRACKER.md) (current state), and
[`docs/ARCHIVE_MANIFEST.md`](../../docs/ARCHIVE_MANIFEST.md) (what is archived).
