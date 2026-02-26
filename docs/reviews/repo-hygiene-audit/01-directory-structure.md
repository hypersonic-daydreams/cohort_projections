# Directory Structure Audit

**Date:** 2026-02-26
**Scope:** Full repository at `cohort_projections/`
**Auditor:** Claude Code (automated)

---

## Executive Summary

The repository contains **10,478 files** across **928 directories** (excluding caches and virtual environments). Only **653 files** are tracked in git; the remaining ~9,800 are gitignored data, outputs, and artifacts synced via rclone.

The dominant structural issue is the `sdc_2024_replication/` subtree, which alone accounts for **7,797 files** and **702 directories** (75% of all files, 76% of all directories), deeply nested up to **13 levels**. The core projection system itself (`cohort_projections/`, `scripts/`, `tests/`, `config/`) is well-organized, but the broader repository suffers from scattered one-off directories at the root level, naming inconsistencies between script directories, and three separate locations for SDC-related data.

---

## Current State

### Top-Level Directory Tree with File Counts

```
cohort_projections/              (repo root)
|
|-- .claude/commands/             1 file   (Claude Code custom commands)
|-- 2025_popest_data/            46 files  (PEP release analysis, NOT tracked in git)
|   +-- charts/                  31 files  (PNG/PDF charts)
|-- cohort_projections/          48 files  (Python source package -- THE CORE)
|   |-- config/                   2 .py    (race mappings)
|   |-- core/                     5 .py    (cohort engine, fertility, mortality, migration)
|   |-- data/
|   |   |-- fetch/                3 .py    (Census API, vital stats fetchers)
|   |   |-- load/                 3 .py    (base population, census loaders)
|   |   +-- process/             10 .py    (rate processing, residual migration)
|   |-- geographic/               3 .py    (geography loader, multi-geography)
|   |-- output/                   4 .py    (reports, visualizations, writers)
|   |   +-- templates/            0 files  (empty, .gitkeep only)
|   +-- utils/                    5 .py    (config, BigQuery, validation)
|-- config/                       3 files  (YAML configs: projection, data sources, branding)
|-- data/                      2,071 files (ALL data -- raw, processed, projections, exports)
|   |-- exports/                765 files  (deliverable workbooks + CSVs)
|   |   |-- baseline/           173 files  (county/state x csv/excel + summaries)
|   |   |-- high_growth/        226 files
|   |   |-- low_growth/         111 files  ** STALE -- see Issues #3 **
|   |   |-- restricted_growth/  226 files
|   |   |-- CBO/                  0 files  (empty)
|   |   |-- methodology_comparison/  0 files (empty)
|   |   +-- packages/            10 files  (timestamped ZIP deliverables)
|   |-- interim/                  5 files  (OCR PDFs for immigration)
|   |-- metadata/                 4 files  (.gitkeep, SQL schema, migrations)
|   |-- output/                  27 files  (visualizations + reports)
|   |-- processed/              134 files  (parquet/CSV rates and populations)
|   |   |-- immigration/
|   |   |   |-- analysis/        42 files
|   |   |   +-- rates/            7 files  ** DUPLICATE data -- see Issues #6 **
|   |   |-- migration/           10 files
|   |   |-- mortality/            2 files
|   |   |-- sdc_2024/            11 files
|   |   +-- ves/                 32 files  (vital events statistics)
|   |-- projections/            569 files  (parquet/CSV/JSON projection results)
|   |   |-- baseline/           193 files  (county/ + state/ + metadata/)
|   |   |-- high_growth/        183 files
|   |   |-- restricted_growth/  179 files
|   |   |-- CBO/                  4 files
|   |   +-- methodology_comparison/  9 files
|   +-- raw/                    564 files  (source data, synced via rclone)
|       |-- census/               0 files  ** EMPTY subdirs -- see Issues #8 **
|       |   |-- acs/              (empty)
|       |   |-- decennial/        (empty)
|       |   +-- pep/              (empty)
|       |-- census_bureau_methodology/  7 files
|       |-- fertility/           15 files
|       |-- geographic/           4 files
|       |-- immigration/        404 files  (DHS, ACS, refugee data, deeply nested)
|       |   +-- dhs_naturalizations/  226 files (20 fiscal-year subdirs!)
|       |-- migration/           12 files
|       |-- mortality/           44 files
|       |-- nd_sdc_2024_projections/  55 files  ** OVERLAP -- see Issues #6 **
|       +-- population/          11 files
|-- docs/                       273 files  (documentation)
|   |-- analysis/                 2 files
|   |-- archive/                 12 files  (old session summaries)
|   |-- governance/
|   |   |-- adrs/                62 files  (Architecture Decision Records)
|   |   |   |-- 020-reports/     84 files  ** BLOATED -- see Issues #5 **
|   |   |   +-- 021-reports/     44 files  ** BLOATED -- see Issues #5 **
|   |   |-- plans/                2 files
|   |   |-- reports/              2 files
|   |   |-- sops/                 7 files  (SOPs + templates)
|   |   +-- templates/            0 files  (empty)
|   |-- guides/                   6 files
|   |-- plans/                    5 files  ** OVERLAP with governance/plans/ **
|   |-- postmortems/              1 file
|   |-- reference/                1 file
|   |-- reports/                  2 files  ** OVERLAP with governance/reports/ **
|   |-- research/                 2 files
|   +-- reviews/                 29 files  (methodology reviews, sanity checks)
|-- examples/                     7 files  (usage examples for the package)
|-- htmlcov/                      (test coverage HTML -- gitignored)
|-- journal_article_pdfs/        28 files  ** MISPLACED -- see Issues #1 **
|-- libs/                        15 files  (3 internal packages)
|   |-- codebase_catalog/
|   |-- evidence_review/
|   +-- project_utils/
|-- logs/                        24 files  (runtime logs -- gitignored)
|-- scratch/                      9 files  (experiments, citations -- gitignored)
|-- scripts/                     64 files  (runnable scripts)
|   |-- data/                    14 files  ** NAMING OVERLAP -- see Issues #4 **
|   |-- data_processing/          4 files  ** NAMING OVERLAP -- see Issues #4 **
|   |-- db/                       5 files  (SQL schemas, migration tools)
|   |-- exports/                  3 files  (workbook builders, methodology text)
|   |-- hooks/                    2 files  (git hooks)
|   |-- intelligence/             3 files  (doc indexing tools)
|   |-- maintenance/              2 files  (link updater, migration applier)
|   |-- migrations/               1 file   (SQL migration)
|   |-- pipeline/                10 files  ** NUMBERING COLLISION -- see Issues #4 **
|   |-- projections/              3 files
|   +-- setup/                    4 files  (project initialization)
|-- sdc_2024_replication/     7,797 files  ** DOMINANT subtree -- see Issues #2 **
|   |-- citation_management/      8 files
|   |-- data/                     5 files  ** TRIPLICATED -- see Issues #6 **
|   |-- data_immigration_policy/ 48 files
|   |-- data_updated/             6 files  ** TRIPLICATED -- see Issues #6 **
|   |-- output/                  12 files
|   |-- revisions/               19 files
|   +-- scripts/              7,760 files
|       |-- database/             7 files
|       +-- statistical_analysis/  7,741 files
|           |-- journal_article/  7,269 files (output: 6,076; claim_review: 1,022)
|           |-- figures/         184 files
|           |-- results/         138 files
|           +-- (modules, concordance, archive, etc.)
+-- tests/                       56 files
    |-- test_config/              2 files
    |-- test_core/                6 files
    |-- test_data/               13 files
    |-- test_geographic/          3 files
    |-- test_integration/         4 files
    |-- test_output/              4 files
    |-- test_statistical/         6 files
    |-- test_tools/               1 file
    |-- test_utils/               6 files
    +-- unit/                     9 files
```

### Top-Level Root Files (22 files)

| File | Tracked | Status |
|------|---------|--------|
| `.env` | No | Gitignored (secrets) |
| `.env.example` | Yes | Appropriate |
| `.envrc` | Yes | direnv config |
| `.gitignore` | Yes | Appropriate |
| `.pre-commit-config.yaml` | Yes | Appropriate |
| `.python-version` | Yes | Appropriate |
| `AGENTS.md` | Yes | AI agent guidance |
| `CLAUDE.md` | Yes | Claude Code quick-ref |
| `DEVELOPMENT_TRACKER.md` | Yes | Project status |
| `README.md` | Yes | Appropriate |
| `REPOSITORY_INVENTORY.md` | Yes | Code inventory (291 KB!) |
| `pyproject.toml` | Yes | Appropriate |
| `uv.lock` | Yes | Appropriate |
| `RCLONE_TEST` | **Yes** | **Should not be tracked** |
| `.coverage` | No | Gitignored |
| `chatgpt_feedback_on_v0.9.md` | No | Stray, gitignored by `*.md` pattern? No -- actually NOT ignored by pattern, but not tracked |
| `formula_audit_article-0.9-production_20260112_205726.md` | No | Stray |
| `ingest.log` | No | Gitignored |
| `ingest_full.log` | No | Gitignored |
| `ingest_nohup.log` | No | Gitignored |
| `ward_county_nd_population_2008_2024.xlsx` | No | Gitignored by `*.xlsx` |

### Disk Usage (non-cache, non-.git)

| Directory | Size |
|-----------|------|
| `data/exports/` | 1.2 GB |
| `data/raw/` | 308 MB |
| `sdc_2024_replication/.../journal_article/output/` | 263 MB |
| `data/projections/` | 68 MB |
| `journal_article_pdfs/` | 33 MB |
| `logs/` | 17 MB |
| **Total repo (excl. caches)** | **~2.0 GB** |

---

## Issues Found

### Issue #1: Root-Level Clutter (Orphaned/Misplaced Directories)

**Severity: Medium**

Several top-level directories do not belong at the repository root:

| Directory/File | Problem | Recommendation |
|----------------|---------|----------------|
| `2025_popest_data/` (46 files) | One-time PEP release analysis with charts, Word docs, and a SQLite database. Not tracked in git. Has no connection to the projection pipeline. | Move to `scratch/2025_popest_analysis/` or a separate repo |
| `journal_article_pdfs/` (28 files) | Compiled PDF versions of the journal article. Not tracked in git. Belongs with `sdc_2024_replication/`. | Move to `sdc_2024_replication/article_pdfs/` or delete (reproducible from source) |
| `RCLONE_TEST` | A bisync test sentinel file. **Tracked in git** but serves no code purpose. | Remove from git tracking |
| `chatgpt_feedback_on_v0.9.md` | ChatGPT review notes, not gitignored, just happens to not be staged. | Move to `scratch/` or `sdc_2024_replication/revisions/` |
| `formula_audit_article-0.9-production_20260112_205726.md` | A one-off audit artifact. | Move to `sdc_2024_replication/revisions/` or `scratch/` |
| `ward_county_nd_population_2008_2024.xlsx` | A single data file at the repo root. | Move to `data/raw/population/` |
| `ingest*.log` (3 files) | Stray log files at root, despite `logs/` directory existing. | Already gitignored; delete or move to `logs/` |

### Issue #2: `sdc_2024_replication/` Dominates the Repository

**Severity: High**

This subtree contains **7,797 files** (75% of all files) and **702 directories** (76% of all directories). Only **271 files** are tracked in git; the remaining ~7,500 are gitignored outputs, figures, and versioned article builds synced via rclone.

Key concerns:

- **Maximum nesting depth is 13 levels** (e.g., `.../journal_article/output/versions/production/article-0.9.9-.../results/seed_sweeps/.../seed_51/`)
- The `journal_article/output/` directory alone holds **6,076 files** across **40 production version directories**, many containing duplicate result sets from seed sweeps
- There is a **recursive phantom directory**: `sdc_2024_replication/scripts/statistical_analysis/sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo/` -- an empty nested copy of the parent path
- **50+ empty directories** within this subtree (empty seed sweep folders, empty revision outputs)
- The `claim_review/` directory holds **1,022 files** of generated claim-review artifacts

This is effectively an **independent project** sharing a monorepo. Its structure should be managed independently or extracted.

### Issue #3: Stale `low_growth` Scenario in Exports

**Severity: Low**

`data/exports/low_growth/` contains **111 files** (last modified 2026-02-17), but `data/projections/low_growth/` **does not exist**. The current scenario set is `baseline`, `high_growth`, and `restricted_growth`. The `low_growth` name appears to be a deprecated predecessor of `restricted_growth`.

Similarly:
- `data/exports/CBO/` exists but is **completely empty** (0 files)
- `data/exports/methodology_comparison/` exists but is **completely empty** (0 files)

These are stale remnants of earlier scenario configurations.

### Issue #4: `scripts/` Has Overlapping and Inconsistently Named Subdirectories

**Severity: Medium**

Three distinct concerns are split across confusingly named directories:

1. **`scripts/data/`** (14 files) -- Data fetching, building, and ingesting scripts (e.g., `download_census_pep.py`, `build_nd_fertility_rates.py`, `ingest_stcoreview.py`)
2. **`scripts/data_processing/`** (4 files) -- PEP data extraction and processing (e.g., `extract_pep_county_migration.py`, `process_pep_rates.py`)
3. **`scripts/pipeline/`** (10 files) -- Numbered pipeline steps that also do data processing (e.g., `01_process_demographic_data.py`, `01_compute_residual_migration.py`)

The distinction between `scripts/data/` and `scripts/data_processing/` is unclear. Both contain data transformation scripts. Meanwhile, `scripts/pipeline/` has a **numbering collision**: two scripts share the `01_` prefix:
- `01_process_demographic_data.py`
- `01_compute_residual_migration.py`

Additionally:
- `scripts/projections/` (3 files) overlaps conceptually with `scripts/pipeline/02_run_projections.py`
- `scripts/intelligence/` (3 files) is an unusual name for documentation tooling
- `scripts/maintenance/` (2 files) and `scripts/migrations/` (1 file) could be consolidated
- There is a **test file in the wrong location**: `scripts/data/test_ves_extraction.py` should be in `tests/`

### Issue #5: ADR Report Directories Are Bloated

**Severity: Low**

Two ADRs have large report subdirectories that break the otherwise flat ADR structure:

- `docs/governance/adrs/020-reports/` -- **84 files** (agent reports, CSV data, figures, ChatGPT review packages, phase plans)
- `docs/governance/adrs/021-reports/` -- **44 files** (data, figures, phase plans, results)

These contain CSV data files, PNG figures, and nested subdirectories (PHASE_A, PHASE_B, chatgpt_review_package) that do not belong alongside ADR markdown files. ADRs should be lightweight decision records, not research project archives.

### Issue #6: SDC Data Is Triplicated Across Three Locations

**Severity: Medium**

The same five rate files (`adjustment_factors_by_county.csv`, `base_population_by_county.csv`, `fertility_rates_by_county.csv`, `migration_rates_by_county.csv`, `survival_rates_by_county.csv`) exist in three places:

1. `sdc_2024_replication/data/` (original, 5 files)
2. `sdc_2024_replication/data_updated/` (updated version, 6 files with MANIFEST)
3. `data/processed/immigration/rates/` (another copy, 7 files with MANIFEST and JSON metadata)

This creates confusion about which is canonical and risks silent divergence.

Additionally, SDC-related raw data is split between:
- `data/raw/nd_sdc_2024_projections/` (55 files) -- the SDC's published projection inputs
- `data/processed/sdc_2024/` (11 files) -- processed SDC rates
- `sdc_2024_replication/data/` -- replication-specific rate files

### Issue #7: `docs/` Has Duplicate Category Directories

**Severity: Low**

Documentation categories exist at two levels:

| Category | Location 1 | Location 2 |
|----------|-----------|-----------|
| Plans | `docs/plans/` (5 files) | `docs/governance/plans/` (2 files) |
| Reports | `docs/reports/` (2 files) | `docs/governance/reports/` (2 files) |
| Templates | `docs/governance/templates/` (empty) | `docs/governance/sops/templates/` (3 files) |

The `docs/governance/` prefix suggests a formal governance artifact, but the split creates ambiguity about where new documents should go.

### Issue #8: Empty Placeholder Directories

**Severity: Low**

Several directories were created in anticipation of future data but remain empty:

- `data/raw/census/acs/` -- empty
- `data/raw/census/decennial/` -- empty
- `data/raw/census/pep/` -- empty (Census PEP data actually lives in `data/raw/population/`)
- `data/exports/CBO/summaries/` -- empty
- `data/exports/methodology_comparison/summaries/` -- empty
- `docs/governance/templates/` -- empty
- `docs/governance/adrs/020-reports/PHASE_B/PLANNING/` -- empty
- `docs/governance/adrs/021-reports/phase_b_plans/` -- empty
- `cohort_projections/output/templates/` -- empty

### Issue #9: `data/output/` vs `data/exports/` Confusion

**Severity: Low**

Two separate output directories exist with unclear delineation:

- `data/output/` (27 files) -- Contains visualizations (PNG) and reports (HTML, JSON, Markdown)
- `data/exports/` (765 files) -- Contains deliverable workbooks and CSVs organized by scenario

The distinction appears to be internal-use outputs vs external deliverables, but this is not documented anywhere and the naming does not make the intent clear.

### Issue #10: Duplicate CSV/Parquet File Pairs in `data/processed/`

**Severity: Low**

Twelve datasets exist in both `.csv` and `.parquet` format:

- `pep_county_components_2000_2024` (.csv + .parquet)
- `pep_county_components_2000_2025` (.csv + .parquet)
- `survival_rates` (.csv + .parquet)
- 9 files in `data/processed/immigration/analysis/` (.csv + .parquet)

The CSV copies appear to be convenience exports. If parquet is the canonical format, the CSVs add unnecessary disk usage and maintenance burden.

---

## Recommendations

### R1: Extract `sdc_2024_replication/` to a Separate Repository (High Priority)

This subtree is an independent research project (a journal article replication study) that shares no code with the core projection engine. It accounts for 75% of all files and 76% of all directories.

**Action:** Move `sdc_2024_replication/` to its own repository (e.g., `sdc-2024-replication`). If it must remain, add a top-level README note explaining it is a companion project, and aggressively prune the 40+ production output versions and 50+ empty directories.

### R2: Consolidate `scripts/` Under a Clear Naming Convention (Medium Priority)

Merge overlapping script directories and adopt a clearer taxonomy:

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `scripts/data/` + `scripts/data_processing/` | `scripts/data/` | Both do data acquisition/transformation |
| `scripts/pipeline/` | `scripts/pipeline/` | Keep, but fix numbering collision |
| `scripts/projections/` | `scripts/projections/` | Keep (standalone projection runners) |
| `scripts/exports/` | `scripts/exports/` | Keep |
| `scripts/db/` + `scripts/migrations/` | `scripts/db/` | Both are database-related |
| `scripts/maintenance/` + `scripts/intelligence/` | `scripts/tools/` | Both are repo maintenance utilities |
| `scripts/hooks/` | `scripts/hooks/` | Keep |
| `scripts/setup/` | `scripts/setup/` | Keep |

Fix the pipeline numbering collision:
- `01_process_demographic_data.py` (general demographic processing)
- `01_compute_residual_migration.py` (specific residual migration)
- `01b_compute_convergence.py`
- `01c_compute_mortality_improvement.py`

Suggested renumbering:
```
00_prepare_processed_data.py          (unchanged)
01_process_demographic_data.py        (unchanged)
02_compute_residual_migration.py      (was 01_)
03_compute_convergence.py             (was 01b_)
04_compute_mortality_improvement.py   (was 01c_)
05_run_projections.py                 (was 02_)
06_export_results.py                  (was 03_)
run_complete_pipeline.sh              (unchanged)
```

### R3: Clean Up Root-Level Clutter (Medium Priority)

- Remove `RCLONE_TEST` from git tracking (`git rm --cached RCLONE_TEST`)
- Move `ward_county_nd_population_2008_2024.xlsx` to `data/raw/population/`
- Delete or move stray root `.log` files to `logs/`
- Move `chatgpt_feedback_on_v0.9.md` and `formula_audit_article-0.9-production_20260112_205726.md` to `scratch/` or `sdc_2024_replication/revisions/`
- Move `2025_popest_data/` to `scratch/2025_popest_analysis/` or a separate repo
- Move `journal_article_pdfs/` to `sdc_2024_replication/article_pdfs/`
- Add `2025_popest_data/` to `.gitignore` explicitly (currently ignored by being untracked but not by pattern)

### R4: Resolve SDC Data Triplication (Medium Priority)

Establish a single canonical location for the SDC rate files. Options:

- **Option A:** Keep only `data/processed/immigration/rates/` (the version with MANIFEST and metadata), and symlink or document the path in `sdc_2024_replication/`.
- **Option B:** Keep only `sdc_2024_replication/data/` and `data_updated/`, and remove the copy in `data/processed/immigration/rates/`.

Whichever is chosen, delete the other copies and document the canonical location.

### R5: Flatten ADR Report Directories (Low Priority)

Move the large ADR report subdirectories out of the ADR folder:

```
docs/governance/adrs/020-reports/  -->  docs/research/adr-020-extended-time-series/
docs/governance/adrs/021-reports/  -->  docs/research/adr-021-immigration-durability/
```

Or archive them to a separate location entirely. ADR files themselves should remain as single markdown documents.

### R6: Consolidate `docs/` Categories (Low Priority)

- Merge `docs/plans/` into `docs/governance/plans/`
- Merge `docs/reports/` into `docs/governance/reports/`
- Remove the empty `docs/governance/templates/` directory
- Consider whether `docs/analysis/`, `docs/reference/`, `docs/postmortems/`, and `docs/research/` (each with 1-2 files) should be consolidated under a single `docs/technical/` or left as-is for future growth

### R7: Remove Stale and Empty Directories (Low Priority)

- Delete `data/exports/low_growth/` (111 stale files from a renamed scenario)
- Delete `data/exports/CBO/` and `data/exports/methodology_comparison/` (empty)
- Delete `data/raw/census/` and its empty subdirectories (acs, decennial, pep) -- Census data lives elsewhere
- Delete the phantom recursive directory `sdc_2024_replication/scripts/statistical_analysis/sdc_2024_replication/`
- Remove empty `docs/governance/adrs/020-reports/PHASE_B/PLANNING/`
- Remove empty `docs/governance/adrs/021-reports/phase_b_plans/`
- Remove empty `cohort_projections/output/templates/` (or add a template)

### R8: Clarify `data/output/` vs `data/exports/` (Low Priority)

Rename for clarity:
- `data/output/` --> `data/visualizations/` (it only contains viz PNGs and generated reports)
- Or merge into `data/exports/visualizations/`

Document the distinction in a `data/README.md`.

### R9: Standardize on Parquet, Remove Duplicate CSVs (Low Priority)

For the 12 files that exist as both `.csv` and `.parquet`:
- Keep `.parquet` as the canonical format (already the standard for core pipeline files)
- Remove the `.csv` duplicates, or add a script that generates CSVs on demand

---

## Proposed New Structure

The following shows the recommended directory layout after applying R1-R9. Changes are marked with `[NEW]`, `[MOVED]`, or `[REMOVED]`.

```
cohort_projections/                    (repo root)
|
|-- .claude/commands/                  (unchanged)
|-- cohort_projections/                (Python source package -- unchanged)
|   |-- config/
|   |-- core/
|   |-- data/
|   |   |-- fetch/
|   |   |-- load/
|   |   +-- process/
|   |-- geographic/
|   |-- output/                        [REMOVED empty templates/ subdir]
|   +-- utils/
|-- config/                            (YAML configs -- unchanged)
|-- data/
|   |-- exports/                       (deliverable workbooks + CSVs)
|   |   |-- baseline/
|   |   |-- high_growth/
|   |   |-- restricted_growth/
|   |   +-- packages/
|   |   [REMOVED: low_growth/, CBO/, methodology_comparison/]
|   |-- interim/                       (unchanged)
|   |-- metadata/                      (unchanged)
|   |-- processed/                     (unchanged, but deduplicate CSV/parquet)
|   |   |-- immigration/
|   |   |   +-- analysis/
|   |   |   [REMOVED: rates/ -- canonical copy lives in sdc repo or single location]
|   |   |-- migration/
|   |   |-- mortality/
|   |   |-- sdc_2024/
|   |   +-- ves/
|   |-- projections/                   (unchanged)
|   |-- raw/                           (unchanged, but remove empty census/ tree)
|   |   |-- census_bureau_methodology/
|   |   |-- fertility/
|   |   |-- geographic/
|   |   |-- immigration/
|   |   |-- migration/
|   |   |-- mortality/
|   |   |-- nd_sdc_2024_projections/
|   |   +-- population/               [RECEIVES ward_county xlsx from root]
|   +-- visualizations/               [RENAMED from data/output/]
|-- docs/
|   |-- analysis/
|   |-- archive/
|   |-- governance/
|   |   |-- adrs/                      [FLAT -- no more 020-reports/ or 021-reports/]
|   |   |-- plans/                     [ABSORBS docs/plans/]
|   |   |-- reports/                   [ABSORBS docs/reports/]
|   |   +-- sops/
|   |-- guides/
|   |-- postmortems/
|   |-- reference/
|   |-- research/                      [RECEIVES ADR report archives]
|   |   |-- adr-020-extended-time-series/  [MOVED from adrs/020-reports/]
|   |   +-- adr-021-immigration-durability/ [MOVED from adrs/021-reports/]
|   +-- reviews/
|-- examples/                          (unchanged)
|-- libs/                              (unchanged)
|   |-- codebase_catalog/
|   |-- evidence_review/
|   +-- project_utils/
|-- logs/                              (unchanged, gitignored)
|-- scratch/                           (gitignored working space)
|   |-- 2025_popest_analysis/          [MOVED from root 2025_popest_data/]
|   |-- citations/
|   |-- experiments/
|   +-- fig11_variants/
|-- scripts/
|   |-- data/                          [ABSORBS scripts/data_processing/]
|   |-- db/                            [ABSORBS scripts/migrations/]
|   |-- exports/
|   |-- hooks/
|   |-- pipeline/                      [RENUMBERED: 00-06 sequential]
|   |-- projections/
|   |-- setup/
|   +-- tools/                         [MERGED: intelligence/ + maintenance/]
+-- tests/                             (unchanged)
    |-- test_config/
    |-- test_core/
    |-- test_data/
    |-- test_geographic/
    |-- test_integration/
    |-- test_output/
    |-- test_statistical/
    |-- test_tools/
    |-- test_utils/
    +-- unit/

[EXTRACTED to separate repo: sdc_2024_replication/]
[REMOVED from root: RCLONE_TEST, journal_article_pdfs/, 2025_popest_data/]
[REMOVED from root: stray .log files, stray .md files, stray .xlsx]
```

### On Numbered Directory Schemes

A numbered directory scheme (e.g., `01_raw_data/`, `02_processing/`) was considered for the `data/` hierarchy. **This is not recommended** for this project because:

1. The existing `data/raw/` -> `data/processed/` -> `data/projections/` -> `data/exports/` naming already communicates the data flow clearly
2. Numbered schemes create friction when inserting new stages
3. The `scripts/pipeline/` directory already uses numbering for its execution order, which is the right place for it

However, the pipeline renumbering (R2) to use sequential numbers `00`-`06` instead of the current `00, 01, 01, 01b, 01c, 02, 03` is strongly recommended.

---

## Summary of Findings

| Category | Count |
|----------|-------|
| Total files (excl. caches) | 10,478 |
| Git-tracked files | 653 |
| Total directories (excl. caches) | 928 |
| Empty directories | 50+ |
| Maximum nesting depth | 13 levels |
| Issues identified | 10 |
| High priority recommendations | 1 (extract sdc_2024_replication) |
| Medium priority recommendations | 3 |
| Low priority recommendations | 5 |

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-26 |
| **Audit Type** | Directory Structure |
| **Sequence** | 01 of repo-hygiene-audit series |
