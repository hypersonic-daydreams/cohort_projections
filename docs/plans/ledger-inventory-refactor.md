# Ledger-Sourced Repository Inventory — Refactor Spec

| Attribute | Value |
|-----------|-------|
| **Status** | **planned** — blocked on confirming the workspace ledger service is live and already indexing this repo (see "Blocking precondition"). |
| **Scope** | `demography/cohort_projections` — repository inventory / navigation tooling only |
| **Date** | 2026-06-23 |
| **Author** | Claude Code (Opus 4.8) |
| **Decision** | Implements Decision 2 of `docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md` ("refactor onto the workspace ledger, do not repair the bespoke `cohort_projections_meta` system"). |
| **Supersedes** | The `scripts/intelligence/` `cohort_projections_meta` inventory chain (already dead — three of its five scripts are missing). |

---

## 1. Problem

The repo's bespoke "Repository Intelligence System" is **dead and duplicative**, and its
flagship output is **stale and actively wrong**:

- **The bespoke system cannot self-repair.** `scripts/intelligence/README.md` references five
  scripts; **three no longer exist** (`scan_repository.py`, `archive_manager.py`,
  `generate_inventory_docs.py`). Nothing can repopulate the local `cohort_projections_meta`
  PostgreSQL database or regenerate `REPOSITORY_INVENTORY.md`. The generator chain is orphaned.
- **`REPOSITORY_INVENTORY.md` (~291 KB, last generated 2026-01-01) is wrong.** It claims
  ~1,919 active / **0 archived** / 0 deprecated files when git actually tracks **826** files and
  **204 archived** files exist on disk. It lists `.mypy_cache` / `.ruff_cache` as "active source
  code" and is too large for an agent to usefully read. This is the single biggest trust hazard in
  the repo.
- **`docs/INDEX.md`** (auto-generated, "Do not edit," last committed 2026-02-27) is sparse, has no
  generated-date stamp, and omits whole subtrees.

Both stale outputs (`REPOSITORY_INVENTORY.md`, `docs/INDEX.md`) are being **removed** in a separate
retirement pass (Decision 2 implementation). This document specifies what **replaces** them.

## 2. Target

Generate a **repo-scoped inventory** by **querying the workspace ledger** rather than
re-implementing a scanner:

- Query `workspace_silver.ledger` filtered to this repo (see §4 on the `repo` identifier).
- Drive the **current-vs-archived signal** directly from `ledger.asset_status` plus
  `ledger.v_stale_assets`, so the artifact can **never again** claim "0 archived."
- Emit a small, human-readable, **date-stamped advisory** markdown file
  (`docs/REPO_INVENTORY.generated.md`) — not a 1,900-line dump. It is a *navigation aid*, not a
  source of truth; `DEVELOPMENT_TRACKER.md` and `AGENTS.md` remain canonical.

The scaffold generator is `scripts/intelligence/generate_repo_inventory.py`. In this pass it is a
**stub**: it probes the ledger and, if reachable, emits a placeholder; if unreachable, it exits 0
with a clear message. The live query is wired up only after the blocking precondition clears.

## 3. Blocking precondition (why Status: planned)

Per `docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md` Decision 2 and
`docs/REFERENCE-cross-machine-db-state.md`, the ledger was **not reachable on this machine at audit
time** (service down; `workspace` not on PATH; `http://127.0.0.1:8765` did not answer). Before the
generator is wired to a real query, confirm **both**:

1. The ledger service is **running** on this machine (HTTP API answers on `127.0.0.1:8765`, or the
   `workspace ledger` CLI is on PATH), and
2. It already holds **fresh `cohort_projections` rows** (run the scanner / `workspace ledger scan`
   for this repo and confirm a recent `scan_runs` entry).

Until both hold, this spec stays `planned` and the generator stays a stub that no-ops gracefully.

## 4. Ledger primitives to use

The ledger is owned by the workspace command-center repo (`src/workspace_command_center/ledger/`);
this repo only **reads** it. Reference docs: `~/workspace/docs/workspace-ledger/` (notably
`AGENT_LOOKUP.md`, `REFERENCE-architecture.md`, `SPEC-scanner-cohort-projections.md`).

**Repo identifier caveat (load-bearing):** in the ledger schema the `repo` **column value for this
repo is the short name `cohort_projections`** (per `SPEC-scanner-cohort-projections.md`), while the
relative *path* form `demography/cohort_projections` appears as `repo_path` / `REPO_REL_PATH`. Filter
on `repo = 'cohort_projections'`; do **not** filter `repo = 'demography/cohort_projections'` or you
will silently get zero rows.

**Tables / views:**

- `ledger.assets` — one row per asset (location, `repo`, `asset_type`, `size_bytes`, `row_count`,
  `last_scanned_at`). The base inventory.
- `ledger.asset_status` — presence / integrity / activity status per asset. Source of the
  current-vs-archived classification.
- `ledger.v_stale_assets` — assets inactive beyond the staleness threshold (~30 days). **Directly**
  drives the "stale / likely-archived" column. This is the concept the bespoke system faked.
- `ledger.v_*` derived views generally (e.g. `v_agent_inventory`, `v_coverage_health`,
  `v_lineage_active`) — prefer these read-only views over raw joins where one fits.
- `ledger.sources`, `ledger.lineage`, `ledger.scan_runs` — provenance, lineage edges, and the
  most-recent-scan timestamp (use the latest `scan_runs` row to stamp the report and to confirm
  freshness per §3).

**Access methods (in preference order):**

1. **HTTP API** — `http://127.0.0.1:8765` (e.g. `GET /find?q=<artifact>`). Dependency-light;
   preferred for the generator.
2. **CLI** — `workspace ledger query {coverage,gaps,health,...}`, `workspace ledger find <artifact>
   --repo cohort_projections`, `workspace ledger scan`. Use when on PATH.
3. **Read-only `SELECT`** against the `ledger.v_*` views in `workspace_silver` (psycopg2),
   guarded by try/except. Last resort; never write.

No re-implemented scanning. The generator only **queries**.

## 5. Caveats to honor

- **Consumer / ownership:** this repo is a **consumer** of the ledger. The ledger is owned by the
  command-center repo. Do not add scanners, migrations, or write paths here; do not assume write
  access. (`AGENTS.md` / root `CLAUDE.md`: per-repo repos read the ledger, they do not own it.)
- **Per-machine, not bisynced:** Postgres databases — including `workspace_silver` / the ledger —
  are **per-machine rebuilds, not bisynced** (`docs/REFERENCE-cross-machine-db-state.md`). Any
  committed markdown snapshot **must** carry a stamp like:
  `Generated from workspace ledger on <YYYY-MM-DD> — advisory, may be machine-specific.`
  The repo must **not hard-depend** on the ledger being up: missing service → graceful no-op, never
  a failed commit or a broken build.
- **Advisory, not canonical:** the generated file is a navigation aid. `DEVELOPMENT_TRACKER.md`
  (current state) and `AGENTS.md` (agent instructions) remain authoritative.

## 6. Task list (ordered)

1. **[done, this pass]** Land the stub `scripts/intelligence/generate_repo_inventory.py` that
   probes the ledger (HTTP then CLI then DB), no-ops gracefully when unreachable, and emits a
   date-stamped advisory placeholder when reachable.
2. **[done, separate pass]** Remove the stale outputs `REPOSITORY_INVENTORY.md` and
   `docs/INDEX.md` from the working tree and git tracking; disable the dead
   `update-code-inventory` pre-commit hook and CLI entrypoint; fix the orphaned
   `scripts/intelligence/README.md` references.
3. **[blocked]** Confirm the §3 precondition: ledger service live on this machine **and** fresh
   `cohort_projections` rows present (recent `scan_runs`).
4. **[blocked]** Wire the live query: filter `repo = 'cohort_projections'`; classify each asset
   current vs. archived/stale via `asset_status` + `v_stale_assets`; aggregate counts by
   `asset_type` and status.
5. **[blocked]** Render a **concise** advisory markdown to `docs/REPO_INVENTORY.generated.md` with
   the per-machine stamp from §5 and the latest `scan_runs` timestamp. Keep it short and readable —
   no cache dirs, no per-file dump of the whole tree.
6. **[blocked]** Decide regeneration cadence (manual `python scripts/intelligence/generate_repo_inventory.py`
   on demand, or a maintenance task). Do **not** re-add an always-running pre-commit hook that
   blocks commits when the ledger is down.
7. **[blocked]** Update `scripts/intelligence/README.md` to describe only the surviving,
   ledger-backed tooling; reference this spec.
8. **[deferred]** If/when stable, link `docs/REPO_INVENTORY.generated.md` from `docs/NAVIGATION.md`
   as the machine-generated companion to the hand-maintained map.

## 7. Out of scope

- Repairing the `cohort_projections_meta` database or its missing scripts (Decision 2 retires it).
- Re-implementing repository scanning, archive management, or doc-linking in this repo.
- Owning, schema-changing, or writing to the ledger.
- Cross-machine reconciliation of ledger contents (per-machine by design).
