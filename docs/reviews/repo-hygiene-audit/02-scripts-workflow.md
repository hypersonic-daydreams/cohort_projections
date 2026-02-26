# Scripts & Workflow Organization Audit

| Attribute | Value |
|-----------|-------|
| **Audit Date** | 2026-02-26 |
| **Scope** | `scripts/` directory, pipeline orchestration, workflow completeness |
| **Auditor** | Claude Code |
| **Status** | Complete |

---

## 1. Current Script Inventory

### 1.1 `scripts/pipeline/` -- Core Pipeline Steps (8 files + README + shell runner)

| File | Purpose | Called by `run_complete_pipeline.sh`? |
|------|---------|--------------------------------------|
| `00_prepare_processed_data.py` | Convert raw CSV files to Parquet format (1:1 conversion) | **No** |
| `01_process_demographic_data.py` | Orchestrate processing of fertility, survival, and migration rates from raw data | **Yes** (Step 1/3) |
| `01_compute_residual_migration.py` | Run residual migration pipeline (PEP data, 5 inter-censal periods, dampening) | **No** |
| `01b_compute_convergence.py` | Compute time-varying migration rates via 5-10-5 convergence interpolation | **No** |
| `01c_compute_mortality_improvement.py` | Compute ND-adjusted mortality improvement projections from national data | **No** |
| `02_run_projections.py` | Run cohort-component projections for all geographies/scenarios (~80KB, largest file) | **Yes** (Step 2/3) |
| `03_export_results.py` | Export projection results to CSV/Excel, create summary tables, packages | **Yes** (Step 3/3) |
| `run_complete_pipeline.sh` | Shell wrapper that runs steps 01, 02, 03 sequentially | N/A (is the runner) |
| `README.md` | Pipeline documentation (detailed, 390 lines) | N/A |
| `__init__.py` | Package docstring listing the three main steps | N/A |

### 1.2 `scripts/data/` -- Data Ingestion & Building (11 files)

| File | Purpose |
|------|---------|
| `download_census_pep.py` | Download historical PEP data from Census FTP server (ADR-034 Phase 1) |
| `convert_popest_to_parquet.py` | Convert Census PEP CSV to Parquet 1:1 (ADR-034 Phase 2) |
| `archive_popest_raw_by_vintage.py` | Archive raw staging files into ZIP by vintage (ADR-034 Phase 3) |
| `extract_popest_docs.py` | Extract PDF documentation and metadata from PEP files (ADR-034 Phase 4) |
| `build_popest_postgres.py` | Build Postgres analytics layer from PEP Parquet files (ADR-034 Phase 5) |
| `view_census_catalog.py` | Display contents of the Census PEP data catalog |
| `ingest_stcoreview.py` | Parse Census stcoreview Vintage 2025 Excel into Parquet |
| `ingest_ves_data.py` | Ingest ND Vital Event Summary PDFs into structured Parquet |
| `test_ves_extraction.py` | Test/validate PDF table extraction from VES PDFs |
| `build_nd_fertility_rates.py` | Build ND-specific ASFR from CDC WONDER data (ADR-053 Part A) |
| `build_nd_survival_rates.py` | Build ND-adjusted survival rates from NVSR state life tables (ADR-053 Part B) |
| `build_race_distribution_from_census.py` | Build age-sex-race distributions from Census full-count data (ADR-044/047/048) |
| `fetch_census_gq_data.py` | Build group quarters population by county/age/sex (ADR-055) |

### 1.3 `scripts/data_processing/` -- PEP Migration Analysis (4 files)

| File | Purpose |
|------|---------|
| `extract_pep_county_migration.py` | Extract/harmonize county-level net migration from PEP 2000-2024 (ADR-035 Phase 1) |
| `extract_pep_county_migration_with_metadata.py` | Enhanced version with PostgreSQL metadata tracking (ADR-035 Phase 1+) |
| `analyze_pep_regimes.py` | Classify counties and compute regime-weighted migration statistics (ADR-035 Phase 2) |
| `process_pep_rates.py` | Process PEP data into age/sex/race migration rate tables (ADR-035 Phase 3) |

### 1.4 `scripts/exports/` -- Export Workbooks (3 files)

| File | Purpose |
|------|---------|
| `_methodology.py` | Shared constants (methodology text, scenario definitions, labels) |
| `build_detail_workbooks.py` | Build per-scenario detail workbooks (state + 8 regions + 53 counties) |
| `build_provisional_workbook.py` | Build multi-sheet provisional workbook for leadership review |

### 1.5 `scripts/projections/` -- Standalone Projection Runners (2 files)

| File | Purpose |
|------|---------|
| `run_pep_projections.py` | Run PEP-based projections for all 53 counties (ADR-035 Phase 5); wraps `02_run_projections.py` via importlib |
| `run_sdc_2024_comparison.py` | Run SDC 2024 methodology projections and compare with baseline |

### 1.6 `scripts/setup/` -- One-Time Setup Scripts (3 files)

| File | Purpose |
|------|---------|
| `01_initialize_project.py` | Create directory structure, validate environment |
| `02_test_bigquery_connection.py` | Verify BigQuery credentials and explore available datasets |
| `03_explore_census_data.py` | Query known Census tables in BigQuery |

### 1.7 `scripts/db/` -- Database Management (4 files + 1 SQL schema)

| File | Purpose |
|------|---------|
| `migrate_from_markdown.py` | Migrate data manifest from Markdown to PostgreSQL |
| `generate_manifest_docs.py` | Generate DATA_MANIFEST.md from PostgreSQL database |
| `backup_manifest_db.sh` | Backup/restore cohort_projections_meta database |
| `schema.sql` | Database schema for manifest metadata |
| `census_pep_metadata_schema.sql` | Census PEP-specific metadata schema |

### 1.8 `scripts/intelligence/` -- Documentation Intelligence (2 files)

| File | Purpose |
|------|---------|
| `link_documentation.py` | Scan repository to infer code-to-documentation relationships |
| `generate_docs_index.py` | Generate unified docs/INDEX.md from intelligence database |

### 1.9 `scripts/maintenance/` -- Repository Maintenance (2 files)

| File | Purpose |
|------|---------|
| `apply_migration.py` | Apply SQL migrations to the metadata database |
| `update_links.py` | Update file path links across Markdown/Python files |

### 1.10 `scripts/hooks/` -- Git Hook Scripts (2 files)

| File | Purpose |
|------|---------|
| `rclone_reminder.sh` | Pre-commit hook: reminds about data sync protocol (always passes) |
| `methodology_doc_check.sh` | Pre-commit hook: warns when core files change without methodology.md update (soft warning) |

### 1.11 `scripts/migrations/` -- SQL Migrations (1 file)

| File | Purpose |
|------|---------|
| `001_add_governance_inventory.sql` | SQL migration to add governance inventory tables |

### 1.12 `scripts/` Root-Level Scripts (7 files)

| File | Purpose |
|------|---------|
| `fetch_data.py` | Fetch data files from sibling repositories using `data_sources.yaml` manifest (ADR-016) |
| `check_test_coverage.py` | Identify orphaned tests and untested production modules |
| `validate_data.py` | Validate all processed data files, generate validation report |
| `process_nd_migration.py` | Process IRS county-to-county migration data for ND counties |
| `extract_sdc_fertility_rates.py` | Extract fertility rates from SDC 2024 Excel workbooks |
| `run_integration_test.py` | Run end-to-end integration test for Cass County |
| `generate_visualizations_and_reports.py` | Generate population pyramids, trend charts, and summary reports |
| `generate_article_pdf.py` | Generate PDF from immigration policy article markdown |
| `bisync.sh` | Wrapper for rclone bisync (data sync with Google Drive) |
| `setup_rclone_bisync.sh` | One-time setup for rclone bisync |

### 1.13 Other Executable Scripts Outside `scripts/`

| File | Purpose |
|------|---------|
| `examples/fetch_census_data.py` | Example: fetching Census data via API |
| `examples/process_migration_example.py` | Example: processing migration rates with synthetic data |
| `sdc_2024_replication/scripts/` | Separate replication study scripts (20+ files, own scope) |

---

## 2. Pipeline Analysis

### 2.1 Intended Pipeline Workflow

The `run_complete_pipeline.sh` documents a simple 3-step pipeline:

```
Step 1/3: 01_process_demographic_data.py --all      (fertility, survival, migration)
Step 2/3: 02_run_projections.py --all                (all geographies, all scenarios)
Step 3/3: 03_export_results.py --all                 (CSV, Excel, packages)
```

### 2.2 Actual Pipeline Workflow (as evolved)

In practice, a full pipeline run requires significantly more steps that are **not captured** in the shell runner:

```
Phase 0: Data Acquisition (manual / ad-hoc)
    scripts/data/download_census_pep.py
    scripts/data/ingest_stcoreview.py
    scripts/data/ingest_ves_data.py
    scripts/data/fetch_census_gq_data.py
    scripts/data/build_nd_fertility_rates.py
    scripts/data/build_nd_survival_rates.py
    scripts/data/build_race_distribution_from_census.py
    scripts/fetch_data.py

Phase 0.5: CSV-to-Parquet Preparation
    scripts/pipeline/00_prepare_processed_data.py

Phase 1a: Old-style rate processing (the "01" that the shell runner calls)
    scripts/pipeline/01_process_demographic_data.py --all

Phase 1b: Residual migration (separate step, same "01" prefix!)
    scripts/pipeline/01_compute_residual_migration.py

Phase 1c: Convergence interpolation
    scripts/pipeline/01b_compute_convergence.py

Phase 1d: Mortality improvement
    scripts/pipeline/01c_compute_mortality_improvement.py

Phase 2: Run projections
    scripts/pipeline/02_run_projections.py --all

Phase 3: Export results (pipeline CSV/parquet export)
    scripts/pipeline/03_export_results.py --all

Phase 3b: Export workbooks (separate from pipeline!)
    scripts/exports/build_detail_workbooks.py
    scripts/exports/build_provisional_workbook.py
```

### 2.3 Missing Pipeline Steps

The following scripts are required for a complete end-to-end run but are **not integrated into the pipeline runner**:

1. **`00_prepare_processed_data.py`** -- Step 00 exists but `run_complete_pipeline.sh` skips it entirely, jumping straight to Step 01.

2. **`01_compute_residual_migration.py`**, **`01b_compute_convergence.py`**, **`01c_compute_mortality_improvement.py`** -- These three data-processing scripts were added after the original pipeline was designed. The shell runner only calls `01_process_demographic_data.py`, not these. They must be run manually in sequence.

3. **All data acquisition scripts** (`scripts/data/`) -- No pipeline step fetches or builds raw data. This is understandable (data acquisition is often manual), but there is no documented checklist or data-readiness validation.

4. **Export workbooks** (`scripts/exports/build_detail_workbooks.py`, `scripts/exports/build_provisional_workbook.py`) -- These are disconnected from `03_export_results.py`. The pipeline export step produces CSV/Excel in a different format from these workbooks.

### 2.4 Ghost Reference: `scripts/projections/run_all_projections.py`

This file is referenced in **10+ documentation files** including:
- `CLAUDE.md` (line 32)
- `AGENTS.md` (line 81)
- `README.md` (line 70)
- `DEVELOPMENT_TRACKER.md` (line 861)
- `docs/NAVIGATION.md` (line 18)
- `docs/REPOSITORY_HYGIENE_IMPLEMENTATION_PLAN.md`

**The file does not exist.** It was either renamed, removed, or never created. The closest equivalents are `scripts/pipeline/02_run_projections.py` (the pipeline version) or `scripts/pipeline/run_complete_pipeline.sh` (the full pipeline runner). Documentation across the repository points users to a command that will fail.

---

## 3. Issues Found

### 3.1 Numbering Collision in Pipeline

Two files share the `01_` prefix with completely different purposes:

| File | Purpose |
|------|---------|
| `01_process_demographic_data.py` | Process raw fertility/survival/migration (original) |
| `01_compute_residual_migration.py` | Compute PEP residual migration rates (added later) |

The sub-steps `01b_compute_convergence.py` and `01c_compute_mortality_improvement.py` use letter suffixes, but `01_compute_residual_migration.py` does not. This creates ambiguity about what "Step 1" actually is. The shell runner calls `01_process_demographic_data.py` but ignores the residual migration step entirely.

**Severity: High** -- An operator following the pipeline README or shell runner would miss critical data processing steps.

### 3.2 Parallel Directory Structures for Data Processing

Data processing scripts exist in two separate directories:

- **`scripts/data/`** -- 13 scripts for data ingestion and building
- **`scripts/data_processing/`** -- 4 scripts for PEP migration analysis

These serve overlapping domains (both process Census PEP data into rates). The split appears accidental -- `scripts/data/` grew organically while `scripts/data_processing/` was created for the ADR-035 PEP migration work. There is no `__init__.py` in `scripts/data_processing/`.

**Severity: Medium** -- Confusing for anyone trying to find where data processing happens.

### 3.3 Hard-Coded Paths

Three scripts contain hard-coded absolute paths that violate the project's "NEVER hard-code file paths" rule:

| File | Hard-coded path |
|------|----------------|
| `scripts/data/ingest_stcoreview.py:35` | `Path("/mnt/c/Users/nhaarstad/Downloads/stcoreview_v2025_ND.xlsx")` |
| `scripts/data_processing/extract_pep_county_migration.py:27` | `Path.home() / "workspace/shared-data/census/popest/parquet"` |
| `scripts/data_processing/extract_pep_county_migration_with_metadata.py:36` | `Path.home() / "workspace/shared-data/census/popest/parquet"` |

The `ingest_stcoreview.py` path is especially fragile -- it references a Windows filesystem path via WSL that is machine-specific.

**Severity: Medium** -- These scripts work on the developer's machine but would fail elsewhere.

### 3.4 `scripts/projections/run_pep_projections.py` Imports Pipeline via `importlib`

This script dynamically imports `02_run_projections.py` using `importlib.util.spec_from_file_location` to reuse its functions. This is a code smell -- it means `02_run_projections.py` contains both orchestration logic and reusable library functions that should live in the `cohort_projections` package.

**Severity: Medium** -- Brittle coupling; if `02_run_projections.py` changes its function signatures, `run_pep_projections.py` breaks silently.

### 3.5 Library Code Embedded in Scripts

Several scripts contain substantial logic that should be library code:

| Script | Logic That Should Be in Library |
|--------|--------------------------------|
| `02_run_projections.py` (80KB) | `load_demographic_rates()`, `run_geographic_projections()`, aggregation reconciliation, and the complete county projection loop are reused by `run_pep_projections.py` |
| `03_export_results.py` (39KB) | CSV/Excel export logic, summary statistics generation, package creation |
| `build_detail_workbooks.py` (24KB) | Excel formatting, age-group binning, sheet generation |
| `build_provisional_workbook.py` (24KB) | Excel formatting (near-duplicate of detail workbooks formatting) |
| `extract_pep_county_migration.py` (16KB) | Vintage harmonization, county extraction logic |

**Severity: Medium** -- Makes scripts hard to test, reuse, and maintain.

### 3.6 Duplicate/Overlapping Scripts

| Pair | Overlap |
|------|---------|
| `extract_pep_county_migration.py` vs. `extract_pep_county_migration_with_metadata.py` | Same data extraction with metadata enhancement; the "with metadata" version is a superset |
| `build_detail_workbooks.py` vs. `build_provisional_workbook.py` | Both build Excel workbooks with similar formatting; share `_methodology.py` but duplicate style constants |
| `01_process_demographic_data.py` vs. `01_compute_residual_migration.py` | Both process migration data but via different pipelines (IRS-based vs. PEP residual); naming collision |
| `validate_data.py` (root) vs. validation in `01_process_demographic_data.py` | Both validate processed data files |
| `run_integration_test.py` vs. `pytest tests/` | Manual integration test script alongside the test suite |

**Severity: Low-Medium** -- Not all overlap is bad (some scripts serve different use cases), but the naming collisions and unclear lineage cause confusion.

### 3.7 Root-Level Script Clutter

Seven Python scripts sit at the `scripts/` root level without clear organizational homes:

- `process_nd_migration.py` -- should be in `scripts/data/` or `scripts/data_processing/`
- `extract_sdc_fertility_rates.py` -- should be in `scripts/data/`
- `validate_data.py` -- should be in `scripts/pipeline/` or `scripts/data/`
- `run_integration_test.py` -- should be in `tests/` or `scripts/testing/`
- `generate_visualizations_and_reports.py` -- should be in `scripts/exports/` or `scripts/pipeline/`
- `generate_article_pdf.py` -- should be in `scripts/exports/` or `docs/`
- `check_test_coverage.py` -- should be in `scripts/maintenance/` or `scripts/testing/`

**Severity: Low** -- These work fine where they are, but the flat layout makes it hard to see what is current vs. legacy.

### 3.8 Stale Setup Scripts

The setup scripts reference BigQuery integration that appears unused by the current pipeline:

- `02_test_bigquery_connection.py` -- Tests BigQuery credentials
- `03_explore_census_data.py` -- Explores BigQuery public datasets

The current pipeline uses local Parquet files and Census API calls, not BigQuery. These may be from an earlier architecture.

**Severity: Low** -- Not harmful but adds confusion about what infrastructure the project actually uses.

### 3.9 Database Scripts Without Clear Integration

The `scripts/db/`, `scripts/intelligence/`, and `scripts/maintenance/` directories contain PostgreSQL-dependent scripts that operate on a metadata database (`cohort_projections_meta`). These are:

- Not part of the pipeline
- Not referenced in any pipeline documentation
- Depend on a local PostgreSQL instance that is not mentioned in setup instructions
- Not tested by the test suite

They appear to be a parallel metadata management system that evolved independently.

**Severity: Low** -- Useful tools but disconnected from the main workflow.

### 3.10 `sys.path.insert` Anti-Pattern

Nearly every script manipulates `sys.path` to find the project root:

```python
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

This is fragile and unnecessary if scripts are run from the project root with the virtualenv activated (which `uv` and `pyproject.toml` should handle). The newer pipeline scripts import from `project_utils` (installed as an editable package in `libs/`), which is the right pattern, but they still use `sys.path.insert`.

**Severity: Low** -- Works but indicates the package installation story is incomplete.

---

## 4. Naming Convention Assessment

### 4.1 Pipeline Scripts

The numbered prefix scheme (`00_`, `01_`, `01b_`, `01c_`, `02_`, `03_`) is a good concept but has degraded:

- **Good**: `02_run_projections.py`, `03_export_results.py` are clear
- **Bad**: Two files starting with `01_` that are different steps
- **Bad**: Letter suffixes (`01b`, `01c`) suggest sub-steps but are actually independent pipeline stages

### 4.2 Data Scripts

Naming in `scripts/data/` is generally good but inconsistent in verb choice:

- `download_`, `convert_`, `archive_`, `extract_` -- action-oriented (good)
- `build_` -- used for data construction (good)
- `ingest_` -- used for parsing external files (good)
- `view_` -- used for display (good)
- `test_ves_extraction.py` -- a test script in a non-test directory (confusing)
- `fetch_census_gq_data.py` -- uses `fetch_` while others use `download_` for similar operations

### 4.3 Export Scripts

- `build_detail_workbooks.py` and `build_provisional_workbook.py` -- consistent `build_` prefix
- `_methodology.py` -- underscore prefix correctly signals "private/internal module"

### 4.4 Root-Level Scripts

No consistent naming convention. Mix of `process_`, `extract_`, `validate_`, `run_`, `generate_`, `check_`, `fetch_`.

---

## 5. Recommendations

### 5.1 Fix the Pipeline Runner (Priority: High)

Update `run_complete_pipeline.sh` to call all required steps in the correct order:

```bash
# Step 0: Prepare processed data
python scripts/pipeline/00_prepare_processed_data.py

# Step 1a: Process demographic data (legacy fertility/survival/migration)
python scripts/pipeline/01_process_demographic_data.py --all

# Step 1b: Compute residual migration
python scripts/pipeline/01_compute_residual_migration.py

# Step 1c: Compute convergence interpolation
python scripts/pipeline/01b_compute_convergence.py --all-variants

# Step 1d: Compute mortality improvement
python scripts/pipeline/01c_compute_mortality_improvement.py

# Step 2: Run projections
python scripts/pipeline/02_run_projections.py --all

# Step 3a: Export pipeline results
python scripts/pipeline/03_export_results.py --all

# Step 3b: Build workbooks
python scripts/exports/build_detail_workbooks.py
python scripts/exports/build_provisional_workbook.py
```

### 5.2 Renumber Pipeline Steps (Priority: High)

Eliminate the `01_` collision by renumbering:

| Current | Proposed | Purpose |
|---------|----------|---------|
| `00_prepare_processed_data.py` | `00_prepare_processed_data.py` | CSV to Parquet |
| `01_process_demographic_data.py` | `01_process_demographic_rates.py` | Process fertility/survival/migration from raw |
| `01_compute_residual_migration.py` | `02_compute_residual_migration.py` | PEP residual migration |
| `01b_compute_convergence.py` | `03_compute_convergence.py` | Convergence interpolation |
| `01c_compute_mortality_improvement.py` | `04_compute_mortality_improvement.py` | Mortality improvement |
| `02_run_projections.py` | `05_run_projections.py` | Run projections |
| `03_export_results.py` | `06_export_results.py` | Export results |

This makes execution order unambiguous and eliminates letter suffixes.

### 5.3 Fix the Ghost Reference (Priority: High)

Either:
- **Option A**: Create `scripts/projections/run_all_projections.py` as a thin wrapper around `run_complete_pipeline.sh` or `02_run_projections.py`
- **Option B**: Update all 10+ documentation references to point to `scripts/pipeline/run_complete_pipeline.sh` or `scripts/pipeline/02_run_projections.py`

Option B is more honest and avoids creating another wrapper.

### 5.4 Merge `scripts/data_processing/` into `scripts/data/` (Priority: Medium)

Move the four PEP migration scripts into `scripts/data/` with a `pep_` prefix:

```
scripts/data/pep_extract_county_migration.py
scripts/data/pep_analyze_regimes.py
scripts/data/pep_process_rates.py
```

Delete `extract_pep_county_migration_with_metadata.py` if the non-metadata version is no longer needed, or vice versa.

### 5.5 Extract Library Code from `02_run_projections.py` (Priority: Medium)

Move reusable functions out of the 80KB pipeline script into the `cohort_projections` package:

- `load_demographic_rates()` --> `cohort_projections/data/load/`
- `run_geographic_projections()` --> `cohort_projections/core/`
- Aggregation/reconciliation logic --> `cohort_projections/core/aggregation.py`

This would eliminate the `importlib` hack in `run_pep_projections.py` and make the functions testable.

### 5.6 Organize Root-Level Scripts (Priority: Low)

Move scripts to logical subdirectories:

| Script | Move To |
|--------|---------|
| `process_nd_migration.py` | `scripts/data/` (or archive if superseded by residual migration) |
| `extract_sdc_fertility_rates.py` | `scripts/data/` |
| `validate_data.py` | `scripts/pipeline/` (or integrate into step 00/01) |
| `run_integration_test.py` | `tests/integration/` (or archive if superseded by pytest suite) |
| `generate_visualizations_and_reports.py` | `scripts/exports/` |
| `generate_article_pdf.py` | `scripts/exports/` |
| `check_test_coverage.py` | `scripts/maintenance/` |

### 5.7 Fix Hard-Coded Paths (Priority: Medium)

- `ingest_stcoreview.py`: Replace `Path("/mnt/c/Users/nhaarstad/Downloads/...")` with a required `--input` argument (no default fallback to a machine-specific path)
- `extract_pep_county_migration.py` and `_with_metadata.py`: Use `CENSUS_POPEST_DIR` environment variable (already used by other PEP scripts) instead of `Path.home() / "workspace/..."`.

### 5.8 Evaluate Script Retirement (Priority: Low)

Several scripts appear to be from earlier iterations and may be candidates for archival:

| Script | Reason |
|--------|--------|
| `scripts/setup/02_test_bigquery_connection.py` | BigQuery appears unused by current pipeline |
| `scripts/setup/03_explore_census_data.py` | Same as above |
| `scripts/data/test_ves_extraction.py` | Test script in data directory; convert to pytest or archive |
| `scripts/data_processing/extract_pep_county_migration.py` | Superseded by `_with_metadata` version? |
| `scripts/process_nd_migration.py` | IRS-based migration superseded by PEP residual method |
| `scripts/extract_sdc_fertility_rates.py` | SDC replication data; may belong in `sdc_2024_replication/` |

### 5.9 Add Data Readiness Validation (Priority: Medium)

Create a lightweight `scripts/pipeline/00_validate_inputs.py` (or enhance step 00) that checks:
- All required input files exist
- File formats are correct
- Required config keys are present
- Environment variables (`CENSUS_POPEST_DIR`, etc.) are set

This would prevent pipeline failures mid-run due to missing inputs.

---

## 6. Summary

### What Works Well

- The pipeline concept (numbered steps, shell runner, dry-run support, resume capability) is solid
- `scripts/data/` has good script naming with verb prefixes and ADR cross-references
- `scripts/exports/_methodology.py` correctly centralizes shared constants
- `scripts/hooks/` implements useful soft-warning pre-commit hooks
- Individual script docstrings are generally excellent with usage examples, ADR references, and method descriptions

### What Needs Attention

| Priority | Issue | Effort |
|----------|-------|--------|
| **High** | Pipeline runner skips 4 required data processing steps | 1 hour |
| **High** | Two files share `01_` prefix with different purposes | 2 hours (renumber + update refs) |
| **High** | `run_all_projections.py` referenced in 10+ docs but does not exist | 1 hour |
| **Medium** | Hard-coded paths in 3 scripts | 30 min |
| **Medium** | `02_run_projections.py` (80KB) mixes library code with orchestration | 4-8 hours |
| **Medium** | `scripts/data/` and `scripts/data_processing/` overlap | 1 hour |
| **Low** | 7 scripts at `scripts/` root without clear organization | 1 hour |
| **Low** | Stale BigQuery setup scripts | 30 min |
| **Low** | PostgreSQL metadata system disconnected from pipeline | Documentation only |

**Total script count**: 47 Python files + 6 shell scripts across 10 subdirectories in `scripts/`, plus 2 example files and the `sdc_2024_replication` scripts (separate scope).
