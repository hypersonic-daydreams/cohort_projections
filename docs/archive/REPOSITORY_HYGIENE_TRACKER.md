# Repository Hygiene Implementation Tracker

**Created:** 2026-02-02
**Purpose:** Track implementation of repository hygiene improvements before starting population projection work
**Methodology:** Sub-agents will be dispatched consecutively to complete each task group

---

## Overview

This document tracks progress on repository hygiene improvements identified through comprehensive analysis. Each section is designed to be completed by a sub-agent working autonomously with clear acceptance criteria.

**Overall Progress:** 11/11 task groups completed ✅

| Priority | Task Groups | Status |
|----------|-------------|--------|
| 🔴 Critical | 2 | ✅ Complete |
| 🟠 High | 3 | ✅ Complete |
| 🟡 Medium | 3 | ✅ Complete |
| 🟢 Low | 3 | ✅ Complete |

---

## 🔴 CRITICAL PRIORITY

### Task Group 1: Git Hygiene - Fix Tracking of Generated Files

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1-2 hours
**Dependencies:** None

#### Context for Sub-Agent
Generated PDFs and images are being tracked in git despite commit `124682f` attempting to remove them. The `.gitignore` patterns are broken because they don't use recursive wildcards (`**`). This causes repository bloat and conflicts with the rclone bisync strategy documented in ADR-016.

#### Tasks
- [x] **1.1** Update `.gitignore` with corrected patterns:
  ```gitignore
  # Fix these patterns (add ** for recursion):
  sdc_2024_replication/scripts/statistical_analysis/journal_article/**/*.pdf
  sdc_2024_replication/scripts/statistical_analysis/journal_article/**/*.png
  data/exports/**
  scratch/**
  ```
- [x] **1.2** Untrack files that match new patterns:
  ```bash
  git rm --cached -r sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/
  git rm --cached -r sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/
  git rm --cached data/exports/packages/*.zip
  git rm --cached scratch/citations/*.pdf
  ```
- [x] **1.3** Verify cleanup with: `git ls-files | grep -E '\.(pdf|png)$' | wc -l` (Result: 1, target was < 30)
- [x] **1.4** Commit changes with message: "fix: correct .gitignore patterns for generated files" (Commit: 0b134dc)

#### Acceptance Criteria
- [x] `.gitignore` uses `**` wildcards for nested directories
- [x] No PDFs in `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/` are tracked
- [x] `git status` shows clean working directory after commit
- [x] Pre-commit hooks pass

#### Files to Modify
- `.gitignore`

#### Reference
- Commit `124682f` - Previous cleanup attempt
- ADR-016 - Raw data management strategy

---

### Task Group 2: Remove Empty/Orphaned Directories

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 30 minutes
**Dependencies:** None

#### Context for Sub-Agent
Several empty directories were created as placeholders but never populated. Additionally, some README files were orphaned when their associated scripts were deleted. These create confusion for AI agents exploring the codebase.

#### Tasks
- [x] **2.1** Remove empty package directories (keep if they have meaningful `__init__.py` content):
  - [x] `cohort_projections/models/` - Removed (empty `__init__.py`)
  - [x] `cohort_projections/validation/` - Removed (empty `__init__.py`)
  - [x] `cohort_projections/data/validate/` - Removed (empty `__init__.py`)
- [x] **2.2** Remove empty script directories:
  - [x] `scripts/validation/` - Removed
  - [x] `scripts/data_pipeline/` - Removed
  - [x] `scripts/export/` - Removed
  - [x] `scripts/hooks/` - Kept (contains `rclone_reminder.sh`)
- [x] **2.3** Remove orphaned documentation:
  - [x] `scripts/intelligence/README.md` - Kept (scripts still exist: `generate_docs_index.py`, `link_documentation.py`)
- [x] **2.4** Commit with message: "chore: remove empty directories and orphaned documentation" (Commit: 64baa74)

#### Acceptance Criteria
- [x] No empty `__init__.py` files in package directories
- [x] No empty script directories
- [x] No orphaned README files (README without associated code)
- [x] Pre-commit hooks pass

#### Files Deleted
- `cohort_projections/models/__init__.py` (and directory)
- `cohort_projections/validation/__init__.py` (and directory)
- `cohort_projections/data/validate/__init__.py` (and directory)
- `scripts/validation/__init__.py` (and directory)
- `scripts/data_pipeline/__init__.py` (and directory)
- `scripts/export/__init__.py` (and directory)

---

## 🟠 HIGH PRIORITY

### Task Group 3: Create Scripts Index and Documentation

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1-2 hours
**Dependencies:** Task Group 2 (empty dirs removed)

#### Context for Sub-Agent
AI agents struggle to discover the correct script for a task. The `scripts/` directory has 13 subdirectories and 9+ top-level scripts with no index. Creating a README will dramatically improve discoverability.

#### Tasks
- [x] **3.1** Create `scripts/README.md` with the following sections:
  - Quick Start (link to pipeline)
  - Pipeline Scripts (00, 01, 02, 03 with descriptions)
  - Data Management scripts
  - Testing & QA scripts
  - Visualization & Reporting scripts
  - Setup scripts
  - Maintenance scripts
- [x] **3.2** Create `scripts/DEPENDENCIES.md` showing data flow:
  ```
  fetch_data.py → [00] Prepare → [01] Process → [02] Project → [03] Export
  ```
- [x] **3.3** Audit existing scripts for missing `--help` support; documented 6 scripts needing improvement
- [x] **3.4** Commit with message: "docs: add scripts directory index and dependency documentation"

#### Acceptance Criteria
- [x] `scripts/README.md` exists with categorized script listing
- [x] `scripts/DEPENDENCIES.md` shows clear data flow between scripts
- [x] All pipeline scripts (00-03) are documented with purpose and usage
- [x] Pre-commit hooks pass

#### Files Created
- `scripts/README.md` (11 categorized sections, --help audit included)
- `scripts/DEPENDENCIES.md` (ASCII data flow diagram, stage-by-stage breakdown)

#### Reference
- `scripts/pipeline/README.md` - Existing pipeline documentation
- `CLAUDE.md` - Quick reference format to follow

---

### Task Group 4: Create Environment and Configuration Documentation

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1-2 hours
**Dependencies:** None

#### Context for Sub-Agent
The repository lacks a `.env.example` template and comprehensive configuration documentation. Users don't know what environment variables are expected, and the `projection_config.yaml` structure isn't documented.

#### Tasks
- [x] **4.1** Create `.env.example` with documented variables:
  ```bash
  # Census API (optional but recommended for data fetching)
  CENSUS_API_KEY=

  # Google BigQuery (required for database features)
  GCP_PROJECT_ID=antigravity-sandbox
  GCP_CREDENTIALS_PATH=~/.config/gcloud/cohort-projections-key.json
  ```
- [x] **4.2** Create `docs/guides/configuration-reference.md` documenting:
  - `projection_config.yaml` structure and all options
  - `data_sources.yaml` manifest format
  - `nd_brand.yaml` visualization settings
  - How to override defaults
- [x] **4.3** Create `docs/NAVIGATION.md` - "Where to find X" quick reference
- [x] **4.4** Commit with message: "docs: add environment template and configuration reference"

#### Acceptance Criteria
- [x] `.env.example` exists with all expected variables documented
- [x] Configuration guide covers all three YAML config files
- [x] Navigation document helps users find information quickly
- [x] Pre-commit hooks pass

#### Files Created
- `.env.example` (Census API, GCP, PostgreSQL, SDC settings)
- `docs/guides/configuration-reference.md` (comprehensive config docs)
- `docs/NAVIGATION.md` (task-oriented lookup guide)

#### Reference
- `config/projection_config.yaml` - Main configuration
- `config/data_sources.yaml` - Data source manifest
- `config/nd_brand.yaml` - Brand colors
- `docs/guides/environment-setup.md` - Existing setup guide

---

### Task Group 5: Add Tests for Utils Module

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 2-3 hours
**Dependencies:** None

#### Context for Sub-Agent
The `cohort_projections/utils/` module contains critical utilities but has **zero test coverage**. This includes configuration loading, BigQuery integration, and reproducibility helpers. Tests should mock external dependencies.

#### Tasks
- [x] **5.1** Create `tests/test_utils/` directory structure:
  ```
  tests/test_utils/
  ├── __init__.py
  ├── conftest.py
  ├── test_config_loader.py
  ├── test_bigquery_client.py
  ├── test_demographic_utils.py
  └── test_reproducibility.py
  ```
- [x] **5.2** Write tests for `config_loader.py` (24 tests, 94.81% coverage):
  - Test successful config loading
  - Test missing config file handling
  - Test config caching behavior
- [x] **5.3** Write tests for `bigquery_client.py` (23 tests, 92.42% coverage):
  - Test connection initialization
  - Test query execution
  - Test credential path expansion
- [x] **5.4** Write tests for `demographic_utils.py` (42 tests, 95.06% coverage):
  - Test utility functions directly
- [x] **5.5** Write tests for `reproducibility.py` (19 tests, 96.67% coverage):
  - Test seed management
  - Test execution logging (mock database)
- [x] **5.6** Run `pytest tests/test_utils/ -v` to verify all pass (108 tests passed)
- [x] **5.7** Commit with message: "test: add comprehensive tests for utils module"

#### Acceptance Criteria
- [x] All 4 test files created with meaningful test cases
- [x] Tests use mocking for external dependencies (GCP, database)
- [x] `pytest tests/test_utils/` passes with 0 failures (108 tests)
- [x] Pre-commit hooks pass

#### Files Created
- `tests/test_utils/__init__.py`
- `tests/test_utils/conftest.py`
- `tests/test_utils/test_config_loader.py` (24 tests)
- `tests/test_utils/test_bigquery_client.py` (23 tests)
- `tests/test_utils/test_demographic_utils.py` (42 tests)
- `tests/test_utils/test_reproducibility.py` (19 tests)

#### Reference
- `cohort_projections/utils/` - Source files to test
- `tests/conftest.py` - Existing fixture patterns
- `tests/test_core/` - Example test structure

---

## 🟡 MEDIUM PRIORITY

### Task Group 6: Consolidate Race/Ethnicity Mappings

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1-2 hours
**Dependencies:** Task Group 5 (tests exist for verification)

#### Context for Sub-Agent
Four nearly-identical race/ethnicity mapping dictionaries exist across the codebase. This violates DRY principles and risks inconsistency. They should be consolidated into a single canonical source.

#### Tasks
- [x] **6.1** Create `cohort_projections/config/race_mappings.py` with:
  - Canonical 6-category mapping from AGENTS.md
  - Source-specific aliases (SEER, Census, IRS)
  - Clear documentation of mapping logic
- [x] **6.2** Update imports in affected files:
  - `cohort_projections/data/process/base_population.py`
  - `cohort_projections/data/process/fertility_rates.py`
  - `cohort_projections/data/process/migration_rates.py`
  - `cohort_projections/data/process/survival_rates.py`
- [x] **6.3** Remove duplicate dictionary definitions from source files
- [x] **6.4** Add tests for race mapping consistency (27 tests)
- [x] **6.5** Run full test suite to verify no regressions (927 tests pass)
- [x] **6.6** Commit with message: "refactor: consolidate race/ethnicity mappings into single module" (Commit: aa8a34d)

#### Acceptance Criteria
- [x] Single source of truth for race mappings in `config/race_mappings.py`
- [x] All 4 processing modules import from centralized location
- [x] No duplicate mapping dictionaries in codebase
- [x] Full test suite passes (927 tests)
- [x] Pre-commit hooks pass

#### Files Created
- `cohort_projections/config/__init__.py`
- `cohort_projections/config/race_mappings.py`
- `tests/test_config/__init__.py`
- `tests/test_config/test_race_mappings.py` (27 tests)

#### Files Modified
- `cohort_projections/data/process/base_population.py`
- `cohort_projections/data/process/fertility_rates.py`
- `cohort_projections/data/process/migration_rates.py`
- `cohort_projections/data/process/survival_rates.py`

#### Reference
- AGENTS.md Section 6 - Canonical race/ethnicity categories
- ADR-007 - Race/ethnicity categorization decision

---

### Task Group 7: Fix Dependency Version Inconsistencies

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 30 minutes
**Dependencies:** None

#### Context for Sub-Agent
The `pyproject.toml` has inconsistent version constraints between main dependencies and optional groups. This can cause confusing installation behavior.

#### Tasks
- [x] **7.1** Consolidate version constraints in `pyproject.toml`:
  - `scipy>=1.13.0` (currently inconsistent: 1.16.3 vs 1.10.0)
  - `statsmodels>=0.14.6` (currently inconsistent: 0.14.6 vs 0.14.0)
  - `matplotlib>=3.10.0` (currently inconsistent: 3.10.8 vs 3.7.0)
- [x] **7.2** Remove Black from dev dependencies (Ruff handles formatting)
- [x] **7.3** Consider moving utility dependencies to optional groups:
  - `weasyprint`, `markdown` → `[project.optional-dependencies.pdf_export]`
  - `xlrd` → `[project.optional-dependencies.excel_io]`
- [x] **7.4** Run `uv sync` to regenerate lock file
- [x] **7.5** Verify installation still works: `uv pip install -e ".[dev]"` (927 tests pass)
- [x] **7.6** Commit with message: "chore: consolidate dependency versions and cleanup unused deps" (Commit: 015e92c)

#### Acceptance Criteria
- [x] No duplicate version constraints for same package
- [x] Black removed from dev dependencies
- [x] `uv sync` succeeds without errors
- [x] `pytest` still runs successfully (927 tests pass)
- [x] Pre-commit hooks pass

#### Files Modified
- `pyproject.toml` (consolidated versions, removed Black, added pdf_export and excel_io optional groups)

#### Reference
- Current `pyproject.toml` lines 56-120 (dependencies)
- `uv.lock` - Will be regenerated

---

### Task Group 8: Improve SDC Replication Script Organization

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1 hour
**Dependencies:** Task Group 3 (scripts index exists)

#### Context for Sub-Agent
The `sdc_2024_replication/scripts/` directory has multiple variant runner scripts that cause confusion: `run_replication.py`, `run_both_variants.py`, `run_three_variants.py`, `run_all_variants.py`. Only one entry point should exist.

#### Tasks
- [x] **8.1** Analyze which variant runner is most comprehensive (run_all_variants.py)
- [x] **8.2** Consolidate to single entry point (`run_all_variants.py`) with flags:
  ```bash
  python run_all_variants.py --variant original
  python run_all_variants.py --variant updated
  python run_all_variants.py --all
  ```
- [x] **8.3** Deprecate redundant scripts (add deprecation notice, don't delete yet):
  - `run_replication.py` → `run_all_variants.py --variant original`
  - `run_both_variants.py` → `run_all_variants.py --variant original --variant updated`
  - `run_three_variants.py` → `run_all_variants.py --all`
- [x] **8.4** Create `sdc_2024_replication/GETTING_STARTED.md` with:
  - Purpose of replication package
  - How to run variants
  - Output locations
- [x] **8.5** Commit with message: "refactor: consolidate SDC variant runners and add documentation"

#### Acceptance Criteria
- [x] Single recommended entry point documented
- [x] Deprecated scripts have clear deprecation notices
- [x] GETTING_STARTED.md provides clear onboarding
- [x] Pre-commit hooks pass

#### Files Created
- `sdc_2024_replication/GETTING_STARTED.md`

#### Files Modified
- `sdc_2024_replication/scripts/run_all_variants.py` (added CLI flags: --variant, --all, --list)
- `sdc_2024_replication/scripts/run_replication.py` (deprecation notice)
- `sdc_2024_replication/scripts/run_both_variants.py` (deprecation notice)
- `sdc_2024_replication/scripts/run_three_variants.py` (deprecation notice)
- `sdc_2024_replication/README.md` (updated with entry point docs)

---

## 🟢 LOW PRIORITY

### Task Group 9: Additional Documentation

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 2 hours
**Dependencies:** Task Groups 3, 4

#### Context for Sub-Agent
Several documentation gaps were identified that would help AI agents work more effectively with the codebase.

#### Tasks
- [x] **9.1** Create `docs/guides/data-sources-workflow.md`:
  - Document all data acquisition sources
  - Update procedures for each data type
  - Contact information for data providers
- [x] **9.2** Create `docs/guides/troubleshooting.md`:
  - Common errors and solutions
  - Debugging tips
  - When to ask for help
- [x] **9.3** Create `docs/reference/geographic-hierarchy.md`:
  - FIPS code reference
  - State → County → Place hierarchy
  - Geographic validation rules
- [x] **9.4** Update `DEVELOPMENT_TRACKER.md`:
  - Updated NEXT TASK section for documentation polish focus
  - Marked Repository Hygiene as complete in sprint status
- [x] **9.5** Commit with message: "docs: add data workflow, troubleshooting, and geographic reference guides"

#### Acceptance Criteria
- [x] All 3 new guide documents created
- [x] DEVELOPMENT_TRACKER.md reflects current state
- [x] Pre-commit hooks pass

#### Files Created
- `docs/guides/data-sources-workflow.md` (comprehensive data acquisition guide)
- `docs/guides/troubleshooting.md` (common errors and solutions)
- `docs/reference/geographic-hierarchy.md` (FIPS codes, county/place lists)

#### Files Modified
- `DEVELOPMENT_TRACKER.md` (updated status)
- `docs/guides/README.md` (added links to new guides)

---

### Task Group 10: Test Suite Improvements

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1-2 hours
**Dependencies:** Task Group 5

#### Context for Sub-Agent
The test suite is comprehensive but could benefit from improvements to fixtures, parametrization, and configuration.

#### Tasks
- [x] **10.1** Add type hints to fixture return values in `tests/conftest.py` (15 fixtures typed across 3 files)
- [x] **10.2** Identify 3-5 test functions that could benefit from `@pytest.mark.parametrize` (3 classes converted)
- [x] **10.3** Add `--timeout=300` to pytest configuration in `pyproject.toml`
- [x] **10.4** Add fixture docstrings explaining synthetic data characteristics
- [x] **10.5** Run full test suite to verify improvements (929 tests pass)
- [x] **10.6** Commit with message: "test: improve fixture typing, parametrization, and timeout config"

#### Acceptance Criteria
- [x] Key fixtures have return type annotations (15 fixtures across 3 conftest.py files)
- [x] At least 3 tests converted to parametrized form (TestCreateAgeGroups, TestCalculateSexRatio, TestCalculateGrowthRate)
- [x] Pytest timeout configured (--timeout=300)
- [x] Full test suite passes (929 tests, 72.71% coverage)
- [x] Pre-commit hooks pass

#### Files Modified
- `tests/conftest.py` (type hints, docstrings)
- `tests/test_utils/conftest.py` (type hints)
- `tests/test_statistical/conftest.py` (type hints, docstrings)
- `tests/test_utils/test_demographic_utils.py` (parametrized tests)
- `pyproject.toml` (pytest-timeout, --timeout=300)

---

### Task Group 11: Code Quality Improvements

**Status:** ✅ Completed (2026-02-02)
**Estimated effort:** 1-2 hours
**Dependencies:** Task Groups 2, 6

#### Context for Sub-Agent
Minor code quality improvements to enhance maintainability and catch potential issues earlier.

#### Tasks
- [x] **11.1** Add module-level docstrings to modules that lack them:
  - `cohort_projections/data/validate/__init__.py` (if not deleted)
  - Other modules identified during exploration
- [x] **11.2** Expand Ruff rules in `pyproject.toml`:
  ```toml
  select = [
      # Add these:
      "PERF",   # Performance anti-patterns
      "FURB",   # refurb suggestions
      "LOG",    # logging best practices
  ]
  ```
- [x] **11.3** Run `ruff check --fix` to auto-fix any new violations (fixed 4 PERF401 issues)
- [x] **11.4** Document MyPy strict mode adoption plan in comments:
  ```toml
  # MyPy Strictness Roadmap:
  # Phase 1 (current): Relaxed mode
  # Phase 2: Enable warn_return_any
  # Phase 3: Enable disallow_untyped_defs for core/
  ```
- [x] **11.5** Commit with message: "chore: expand linting rules and document type checking roadmap" (Commit: 839d21b)

#### Acceptance Criteria
- [x] All modules have docstrings
- [x] Extended Ruff rules configured (PERF, FURB, LOG)
- [x] No new Ruff violations (4 fixed)
- [x] MyPy roadmap documented
- [x] Pre-commit hooks pass

#### Files Modified
- `pyproject.toml` (Ruff rules, MyPy roadmap)
- `cohort_projections/__init__.py` (docstring)
- `cohort_projections/data/__init__.py` (docstring)
- `cohort_projections/data/load/base_population_loader.py` (PERF fixes)
- `cohort_projections/geographic/multi_geography.py` (PERF fixes)

---

## Execution Log

Record sub-agent dispatches and completions here:

| Date | Task Group | Agent ID | Status | Notes |
|------|------------|----------|--------|-------|
| 2026-02-02 | Task Group 1 | a3a3d89 | ✅ Complete | 548 files untracked, final PDF/PNG count: 1 |
| 2026-02-02 | Task Group 2 | a5fad70 | ✅ Complete | 6 empty directories removed, kept hooks/ and intelligence/ |
| 2026-02-02 | Task Group 3 | ae6a066 | ✅ Complete | Created README.md and DEPENDENCIES.md with full script index |
| 2026-02-02 | Task Group 4 | acf071d | ✅ Complete | Created .env.example, config reference, and NAVIGATION.md |
| 2026-02-02 | Task Group 5 | a7743b0 | ✅ Complete | 108 tests for utils module, 92-97% coverage per file |
| 2026-02-02 | Task Group 6 | a6999c2 | ✅ Complete | Consolidated race mappings, 27 new tests, 927 total pass |
| 2026-02-02 | Task Group 7 | a8d1d0b | ✅ Complete | Consolidated deps, removed Black, added optional groups |
| 2026-02-02 | Task Group 8 | a883e0c | ✅ Complete | Consolidated SDC runners, added CLI flags, GETTING_STARTED.md |
| 2026-02-02 | Task Group 9 | a0b3b44 | ✅ Complete | Created 3 guide documents, updated DEVELOPMENT_TRACKER |
| 2026-02-02 | Task Group 10 | ac43b0a | ✅ Complete | Fixture typing, parametrized tests, pytest timeout |
| 2026-02-02 | Task Group 11 | abb1e1d | ✅ Complete | Ruff rules expanded, MyPy roadmap, PERF fixes |

---

## Post-Implementation Verification

After all task groups are complete, run this verification checklist:

```bash
# 1. Git hygiene check
git ls-files | grep -E '\.(pdf|png)$' | wc -l  # Should be < 30

# 2. No empty directories
find . -type d -empty -not -path './.git/*' | wc -l  # Should be 0

# 3. Full test suite
pytest tests/ -v

# 4. Pre-commit on all files
pre-commit run --all-files

# 5. Type checking
mypy cohort_projections/

# 6. Documentation links work
# Manually verify links in new documentation

# 7. Environment setup works
# Test on fresh clone or in clean environment
```

---

## Notes

- Sub-agents should read this document before starting work
- Mark tasks complete with `[x]` as they finish
- Update the execution log after each sub-agent completes
- If a task group has blockers, document them in the Notes column
- This document should be deleted or archived after all improvements are implemented

---

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-02-02 |
| **Last Updated** | 2026-02-02 |
| **Status** | ✅ Completed |
| **Owner** | Repository maintainer |
