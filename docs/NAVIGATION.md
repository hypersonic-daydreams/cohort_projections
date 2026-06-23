# Repository Navigation Map

| Attribute | Value |
|-----------|-------|
| **Status** | current |
| **Last Updated** | 2026-06-23 |
| **Owner** | Repository maintainers |
| **Supersedes** | `docs/INDEX.md` (auto-generated, sparse, retired) |

A hand-maintained directory map for humans and agents. It answers "where does X
live, and is it current?" without opening many files. It is the navigational
replacement for the retired auto-generated `docs/INDEX.md`.

This map names directories and points at the authority for currency; it does
**not** itself track project status. For that, follow the entry-point hierarchy
below.

---

## Entry-Point Hierarchy

Read in this order. Each layer is more detailed and more authoritative for its
scope than the one before it.

| Order | Document | Role |
|-------|----------|------|
| 1 | `README.md` | Orientation — what the project is, how to run it. |
| 2 | `CLAUDE.md` | Claude Code quick-reference (commands, key rules). Wrapper over AGENTS.md. |
| 3 | `AGENTS.md` | **Canonical agent instructions** — workspace purpose, safety rules, conventions. Authoritative for *how to work in this repo*. |
| 4 | `DEVELOPMENT_TRACKER.md` | **Canonical current state** — the source of truth for *what work remains / what is done*. Answer "what is the status of X?" from here first. |

> The retired root file `REPOSITORY_INVENTORY.md` is **not** part of this set.
> It was auto-generated, stale, and is being removed; do not trust it.

**Currency authority:** `DEVELOPMENT_TRACKER.md` is the single source of truth
for active vs. complete work. Planning documents are classified (current /
supporting / historical) in `docs/plans/README.md`. Archived material is
enumerated in `docs/ARCHIVE_MANIFEST.md` under the rules in `ARCHIVE_STRATEGY.md`.

---

## Top-Level Directories

| Path | Purpose | Status-source pointer | Notes |
|------|---------|-----------------------|-------|
| `cohort_projections/` | Main Python package — the projection engine and library code. | `DEVELOPMENT_TRACKER.md` (Current Snapshot) | Implementation lives here; runners live in `scripts/`. The library-vs-runner split is the key structural relationship. |
| `scripts/` | Executable runners, pipelines, and tooling (CLI layer over the library). | `DEVELOPMENT_TRACKER.md` | 18 subdirs; see the scripts subtable below. |
| `tests/` | Test suite (unit + integration), mirroring the package tree. | `docs/guides/test-suite-reference.md` | Baseline test counts recorded in `DEVELOPMENT_TRACKER.md` and `MEMORY` notes. |
| `config/` | YAML configuration (projection, evaluation, observatory, brand). | `config/projection_config.yaml` header / ADRs | Never hard-code paths or rates — read from here. |
| `libs/` | Extracted internal packages (`codebase_catalog`, `evidence_review`, `project_utils`). Second package root alongside `cohort_projections/`. | (per-lib) | Support tooling, not the projection engine. |
| `examples/` | 7 standalone demo scripts showing library usage. | (self-evident) | Illustrative only; not part of the production pipeline. |
| `data/` | All data inputs, intermediates, and outputs (gitignored; synced via rclone). | per-subdir README + `DATA_SOURCE_NOTES.md` | ~2.8 GB; see the data subtable below. |
| `docs/` | Documentation tree (governance, guides, methodology, plans, reviews). | `DEVELOPMENT_TRACKER.md` + `docs/plans/README.md` | See the docs subtable below. |
| `scratch/` | Transient working files: experiments, citations, figure drafts, and `scratch/archive/`. | none — by definition transient | No preservation guarantee. Auto-cleanable. |
| `data_archives/` | **Deprecated/superseded data outputs** (gitignored, on-disk only). Renamed from `archived/`. | `docs/ARCHIVE_MANIFEST.md` | One of three archive zones (see `ARCHIVE_STRATEGY.md`). Each subdir carries a `README_STALE.md`. |
| `observatory_wireframe/` | Legacy Observatory UI mockup. | `docs/ARCHIVE_MANIFEST.md` | Being archived (Decision 4, hygiene proposal). Not a live design artifact. |
| `logs/`, `htmlcov/` | Generated run logs and coverage HTML. | n/a | Gitignored, regenerable, safe to delete. |
| `.venv/`, `.git/`, `.mypy_cache/`, `.ruff_cache/`, `.uv_cache/`, `.pytest_cache/`, `cohort_projections.egg-info/` | Tooling/cache directories. | n/a | Gitignored, regenerable. Skip when scanning the repo. |

### Root files

| Path | Purpose |
|------|---------|
| `README.md` | Orientation entry point (hierarchy level 1). |
| `CLAUDE.md` | Claude Code quick-reference (level 2). |
| `AGENTS.md` | Canonical agent instructions (level 3). |
| `DEVELOPMENT_TRACKER.md` | Canonical current state (level 4). |
| `ARCHIVE_STRATEGY.md` | The one archival convention (three zones, lifecycle). |
| `pyproject.toml`, `uv.lock` | Packaging and pinned dependencies. |
| `REPOSITORY_INVENTORY.md` | **Retired** — stale auto-generated inventory, slated for removal. Do not trust. |

---

## `scripts/` Subdirectories

| Path | Purpose |
|------|---------|
| `scripts/pipeline/` | Numbered end-to-end pipeline stages (`01_…` → `03_export_results.py`); numeric-prefix + symlink convention. |
| `scripts/projections/` | Projection-run entry points (e.g. `run_all_projections.py`). |
| `scripts/data/` | Canonical data fetch/build (e.g. Census GQ data). |
| `scripts/data_processing/` | Exploratory/one-off data processing (distinct from canonical `data/`). |
| `scripts/exports/` | Output builders (detail workbooks, provisional workbook, marketing docx, pyramid explorer, explainer figures). |
| `scripts/analysis/` | Experiment harness, Observatory CLI/dashboard, benchmarking entry points. |
| `scripts/backtesting/` | Walk-forward / rolling-origin validation runners. |
| `scripts/validation/` | Output validation checks. |
| `scripts/db/`, `scripts/migrations/` | Database setup and schema migrations. |
| `scripts/intelligence/` | Legacy inventory/index tooling (being refactored onto the workspace ledger; see hygiene proposal Decision 2). |
| `scripts/maintenance/` | Repository maintenance utilities. |
| `scripts/reviews/` | Review-generation helpers. |
| `scripts/hooks/`, `scripts/setup/`, `scripts/testing/`, `scripts/windows/` | Git hooks, environment setup, test helpers, Windows-specific scripts. |

---

## `docs/` Subdirectories

| Path | Purpose | Status-source pointer | Notes |
|------|---------|-----------------------|-------|
| `docs/governance/adrs/` | Architecture Decision Records (`NNN-kebab.md`). | each ADR's own Status field | Authority for *why* the system is built as it is. See `docs/work-registry.md`. |
| `docs/governance/sops/` | Standard Operating Procedures (`SOP-NNN-…md`). | each SOP header | Process rules (data docs, prose voice, benchmarking). |
| `docs/governance/plans/`, `docs/governance/reports/` | Package-extraction plans and governance reports. | `DEVELOPMENT_TRACKER.md` | |
| `docs/plans/` | Plans, backlogs, feature-idea docs; includes the PUB-2026 release-handoff subtree. | `docs/plans/README.md` (current/supporting/historical) | Consult the README before treating any plan as open work. |
| `docs/guides/` | How-to guides (observatory, testing, compute runtime, etc.). | (self-evident) | |
| `docs/methodology.md` and `methodology_*` | Canonical projection methodology and comparisons. | the file's own header | Update whenever formulas/rates/data sources change. |
| `docs/reviews/` | Dated reviews, approval gates, AI-analysis packages. | `DEVELOPMENT_TRACKER.md` links | Large flat dir; being regrouped (hygiene Decision 6). |
| `docs/postmortems/` | Incident/bug postmortems. | (dated) | |
| `docs/reference/` | Reference material and crosswalks. | (self-evident) | |
| `docs/reports/`, `docs/research/`, `docs/analysis/` | Generated reports, research notes, analysis writeups. | (dated / per-file) | |
| `docs/archive/` | **Deprecated/superseded documentation** (git-tracked). One of three archive zones. | `docs/archive/README.md` + `docs/ARCHIVE_MANIFEST.md` | |
| `docs/NAVIGATION.md` | This map. | — | |
| `docs/naming-conventions.md` | Filename/metadata conventions. | — | |
| `docs/work-registry.md` | The work-ID schemes (ADR/SOP/PP/CF/PUB). | — | |
| `docs/ARCHIVE_MANIFEST.md` | Enumeration of all archived material across the three zones. | — | |
| `docs/INDEX.md` | **Retired** — sparse auto-generated index; replaced by this file. | — | |

---

## `data/` Subdirectories

`data/` is gitignored and synced between machines via rclone. Inputs flow from
`raw/` → `processed/`/`interim/` → `projections/` (runs) → `exports/`
(deliverables).

| Path | Purpose | Status-source pointer | Notes |
|------|---------|-----------------------|-------|
| `data/raw/` | Source inputs by domain (census, enrollment, fertility, mortality, migration, housing, geographic, immigration, population, nd_sdc_2024_projections). | `DATA_SOURCE_NOTES.md` | Update notes when adding files. **Source vaults inside are not archives** — see below. |
| `data/processed/` | Cleaned/derived inputs (PEP components, GQ data, etc.). | `docs/methodology.md` | Consumed by the engine. |
| `data/interim/` | Intermediate transforms between raw and processed. | — | |
| `data/projections/` | Projection-run outputs by scenario. | `data/projections/README.md` (when present) + `README_STALE.md` markers | `baseline/` = **current locked public run**; `high_growth/`, `restricted_growth/` are STALE and slated for `data_archives/`. See subtable below. |
| `data/exports/` | Public/provisional deliverables (Excel, CSV, HTML) generated from projection runs. | `data/exports/README.md` (when present) | Kept separate from `projections/` by Decision 7. Only the latest dated set is canonical. |
| `data/analysis/` | Benchmark history, experiment logs, evaluation artifacts. | `DEVELOPMENT_TRACKER.md` (Observatory rows) | Largest subtree. |
| `data/backtesting/` | Walk-forward / rolling-origin validation outputs. | `DEVELOPMENT_TRACKER.md` | |
| `data/output/` | Misc generated outputs. | — | |
| `data/metadata/` | Run metadata (`projection_run_*.json`). | — | Maps a run to its config/date. |

### `data/projections/` scenario directories

| Path | Classification | Notes |
|------|----------------|-------|
| `data/projections/baseline/` | **CURRENT — locked public run** | The PUB-2026 public release (`m2026r1`, locked config). The canonical scenario. |
| `data/projections/CBO/` | Input-reference | CBO-adjusted inputs feeding the baseline (ADR-065). |
| `data/projections/sensitivity_20260611/`, `sensitivity_refintl_corrected_20260615/` | Internal what-if | Sensitivity runs; not public. |
| `data/projections/methodology_comparison/` | Internal reference | SDC/naive comparison outputs. |
| `data/projections/exports/` | Derived | Geospatial and other run-derived exports (`geojson/` is STALE — has `README_STALE.md`). |
| `data/projections/high_growth/`, `data/projections/restricted_growth/` | **STALE — being archived** | Pre-ADR-065/066 outputs. Carry `README_STALE.md`. Moving to `data_archives/` (Decision 1). |

---

## Source Vaults — Not Archives

Immutable reference data inside `data/raw/` is **never** to be deleted and is
**not** part of the archive taxonomy:

- `data/raw/immigration/rpc_vault/` (renamed from `rpc_archives/`)
- `data/raw/nd_sdc_2024_projections/source_files/vault/` (renamed from `backup/`)

Per `ARCHIVE_STRATEGY.md`, these are source vaults — immutable reference data.
An agent must never treat them as expendable archive content.
