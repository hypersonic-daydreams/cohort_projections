# Configuration & Data Management Audit

**Audit Date:** 2026-02-26
**Scope:** `config/`, `data/`, `.env*`, `.gitignore`, `pyproject.toml`, and config loading code
**Auditor:** Claude Code (repo-hygiene-audit series)

---

## 1. Configuration Inventory

### 1.1 Files in `config/`

| File | Size | Purpose | Last Modified |
|------|------|---------|---------------|
| `projection_config.yaml` | 16.3 KB | Main projection parameters: scenarios, rates, geography, output, validation, pipeline settings | 2026-02-26 |
| `data_sources.yaml` | 15.2 KB | Data source manifest: documents external data acquisition paths, URLs, column requirements | 2025-12-28 |
| `nd_brand.yaml` | 2.2 KB | ND Dept. of Commerce brand colors for plots, growth-rate color rules, Plotly/Matplotlib themes | 2026-01-27 |

**Assessment:** The config directory is well-organized with clear separation of concerns. Three files serve distinct purposes: runtime projection parameters, data provenance documentation, and visual identity. No config sprawl in this directory.

### 1.2 Environment Configuration

| File | Tracked | Purpose |
|------|---------|---------|
| `.envrc` | Yes | direnv configuration; loads `.env`, sets `VIRTUAL_ENV` and `PROJECT_ROOT` |
| `.env.example` | Yes | Template with all environment variables documented |
| `.env` | No (gitignored) | Actual secrets and local configuration |

**Environment Variables Used:**

| Variable | Where Used | Required? |
|----------|-----------|-----------|
| `CENSUS_API_KEY` | `examples/fetch_census_data.py` | Optional (rate limiting) |
| `CENSUS_POPEST_DIR` | `scripts/data/download_census_pep.py`, `cohort_projections/data/popest_shared.py`, multiple data scripts | Required for PEP archive access |
| `CENSUS_POPEST_PG_DSN` | `scripts/data_processing/extract_pep_county_migration_with_metadata.py`, `scripts/data/build_popest_postgres.py` | Optional (database features) |
| `GCP_PROJECT_ID` / `GCP_CREDENTIALS_PATH` | BigQuery client | Optional |
| `DEMOGRAPHY_DB_*` / `DEMOGRAPHY_DATABASE_URL` | `sdc_2024_replication/scripts/database/db_config.py` | Optional (SDC replication) |
| `SDC_ANALYSIS_DATA_SOURCE` | `sdc_2024_replication/scripts/statistical_analysis/data_loader.py` | Optional |
| `MPLBACKEND` | Test files | Optional (CI/headless) |
| `PGHOST`, `PGPORT`, etc. | `libs/codebase_catalog/` | Optional (code catalog DB) |

**Assessment:** Environment variables are well-documented in `.env.example` with clear section headers and explanations. The `.envrc` pattern with direnv is clean. The separation between `.env` (secrets, gitignored) and `.env.example` (template, tracked) is correct.

### 1.3 Other Configuration Locations

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project metadata, dependencies, tool config (ruff, mypy, pytest, coverage) |
| `.pre-commit-config.yaml` | Pre-commit hook definitions |
| `scripts/exports/_methodology.py` | Shared constants for export scripts (scenario names, methodology text) |

### 1.4 Config Loader Architecture

There are **two** `ConfigLoader` implementations:

1. **`cohort_projections/utils/config_loader.py`** -- Primary loader used by the projection engine. Resolves config directory relative to `__file__`. Has caching, `get_parameter()` for nested key access, and convenience methods for projection, fertility, mortality, and migration configs.

2. **`libs/project_utils/project_utils/config.py`** -- Library version. Resolves config directory from `Path.cwd()`. Adds `_interpolate_env_vars()` for `${VAR}` expansion, dot-notation `get()`, and `get_path()` for relative path resolution.

The main `cohort_projections/utils/__init__.py` imports from both and provides a fallback mechanism:

```python
from .config_loader import ConfigLoader as _LocalConfigLoader
try:
    from project_utils import ConfigLoader as _ConfigLoader
    ...
except ModuleNotFoundError:
    _HAS_PROJECT_UTILS = False
```

---

## 2. Data Directory Assessment

### 2.1 Directory Structure

```
data/
+-- DATA_MANIFEST.md          # Comprehensive temporal alignment metadata (tracked)
+-- DATA_VALIDATION_REPORT.md # Automated validation results (tracked)
+-- README.md                 # Setup guide for new users (tracked)
+-- raw/                      # External source data (NOT in git, synced via rclone)
|   +-- census/               # Census API data (ACS, decennial, PEP)
|   +-- fertility/            # CDC NCHS birth rate data
|   +-- geographic/           # County/place reference files
|   +-- immigration/          # DHS, refugee, ACS migration data (large subtree)
|   +-- migration/            # IRS SOI county-to-county flows
|   +-- mortality/            # CDC life tables
|   +-- nd_sdc_2024_projections/  # SDC replication source files
|   +-- population/           # Census PEP county/state estimates
+-- processed/                # Derived rate tables (NOT in git, synced via rclone)
|   +-- immigration/          # Immigration analysis outputs
|   +-- migration/            # Convergence rates, residual rates
|   +-- mortality/            # Processed survival rates
|   +-- sdc_2024/             # SDC replication processed data
|   +-- ves/                  # Vintage evaluation study
+-- projections/              # Projection outputs (NOT in git)
|   +-- baseline/             # Baseline scenario (county/ + state/ + metadata/)
|   +-- high_growth/          # High growth scenario
|   +-- restricted_growth/    # Restricted growth scenario
|   +-- CBO/                  # CBO methodology comparison
|   +-- methodology_comparison/
+-- exports/                  # Final dissemination packages (NOT in git)
|   +-- baseline/             # county/{csv,excel} + state/{csv,excel} + summaries/
|   +-- high_growth/          # Same structure
|   +-- restricted_growth/    # Same structure
|   +-- packages/             # ZIP archives
+-- interim/                  # Intermediate work (NOT in git)
+-- metadata/                 # Database schemas (NOT in git)
+-- output/                   # Reports and visualizations (NOT in git)
```

**Assessment:** The directory structure follows a clean `raw -> processed -> projections -> exports` pipeline. The hierarchy is logical and the separation between stages is clear.

### 2.2 Git Tracking Policy

The `.gitignore` is comprehensive and well-organized:

- **Data files:** All CSV, Parquet, XLSX, JSON, PDF etc. under `data/raw/`, `data/processed/`, `data/projections/`, `data/exports/`, and `data/interim/` are gitignored.
- **Exceptions:** `.gitkeep` files and `DATA_SOURCE_NOTES.md` / `MANIFEST.md` are explicitly un-ignored (tracked).
- **Documentation tracked:** `data/DATA_MANIFEST.md`, `data/DATA_VALIDATION_REPORT.md`, `data/README.md`, and `data/raw/*/DATA_SOURCE_NOTES.md` are all in git.

This means the data directory structure and documentation travel with the repo, but actual data files do not. Data is synced via rclone (`scripts/bisync.sh`) per ADR-016.

### 2.3 Data Documentation Quality

**DATA_SOURCE_NOTES.md files exist in 4 of 10 raw subdirectories:**

| Directory | Has NOTES? | Quality |
|-----------|-----------|---------|
| `data/raw/fertility/` | Yes | Excellent -- 361 lines, covers all files, CDC WONDER query parameters, race mapping tables, ND-specific processing, validation metrics |
| `data/raw/mortality/` | Yes | Excellent -- 238 lines, covers all 18+ life table files, ND ratio adjustment methodology, column definitions, race mapping, citations |
| `data/raw/population/` | Yes | Excellent -- 469 lines, covers stcoreview, county characteristics, single-year estimates, historical notes on data transitions, validation tables |
| `data/raw/migration/` | Yes | Excellent -- 295 lines, covers PEP primary source and IRS legacy source, column definitions, processing scripts, state-level migration summary, limitations |
| `data/raw/census/` | No | Missing |
| `data/raw/census_bureau_methodology/` | No | Missing |
| `data/raw/geographic/` | No | Missing (referenced in `data_sources.yaml` but no local notes) |
| `data/raw/immigration/` | No | Missing (partially covered by `DATA_MANIFEST.md`) |
| `data/raw/nd_sdc_2024_projections/` | No | Missing |

**Top-level data documentation:**

| Document | Lines | Quality |
|----------|-------|---------|
| `data/DATA_MANIFEST.md` | 446 | Strong -- temporal alignment matrix, FY/CY handling, script cross-references, appendix with file counts |
| `data/DATA_VALIDATION_REPORT.md` | 181 | Good -- automated report, 34/34 checks passing, file-by-file results |
| `data/README.md` | 374 | Good -- data acquisition guide, pipeline steps, alternative sources, citations |

---

## 3. Path Management Assessment

### 3.1 Config-Driven Paths (Good Practice)

The `projection_config.yaml` centralizes most data paths. Examples:

```yaml
base_population:
  single_year_distribution: "data/raw/population/nd_age_sex_race_distribution_single_year.csv"
  county_distributions:
    path: "data/processed/county_age_sex_race_distributions.parquet"
  group_quarters:
    gq_data_path: "data/processed/gq_county_age_sex_2025.parquet"

pipeline:
  data_processing:
    migration:
      pep_input: "data/processed/pep_county_components_2000_2025.parquet"
```

The config uses relative paths from the project root, which is the correct approach.

### 3.2 Fallback Defaults in Code (Acceptable Pattern)

Several source files use config values but provide fallback defaults if config is absent:

| File | Fallback Default |
|------|-----------------|
| `base_population_loader.py:227` | `"data/raw/population/nd_age_sex_race_distribution_single_year.csv"` |
| `base_population_loader.py:571` | `"data/processed/county_age_sex_race_distributions.parquet"` |
| `base_population_loader.py:918` | `"data/processed/gq_county_age_sex_2025.parquet"` |
| `residual_migration.py:1062` | `"data/processed/gq_county_age_sex_historical.parquet"` |
| `residual_migration.py:1160` | `"data/processed/pep_county_components_2000_2025.parquet"` |

These follow a consistent `config.get("key", "fallback/path")` pattern. The fallback paths match what is in `projection_config.yaml`, so they serve as documentation-in-code rather than silent overrides. This is acceptable but creates a maintenance risk: if the config path changes, the fallback in code must also be updated.

### 3.3 Hardcoded Absolute Paths (Issues)

Three files contain hardcoded user-specific absolute paths:

| File | Line | Hardcoded Path | Severity |
|------|------|---------------|----------|
| `scripts/data/ingest_stcoreview.py:35` | `DEFAULT_INPUT = Path("/mnt/c/Users/nhaarstad/Downloads/stcoreview_v2025_ND.xlsx")` | Medium -- ingestion script default, overridable via CLI arg |
| `sdc_2024_replication/update_audit_method.py:4` | `JSON_FILE = "/home/nhaarstad/workspace/demography/cohort_projections/sdc_2024_replication/..."` | Low -- one-off script in replication study |
| `examples/fetch_census_data.py:149` | `Path("/tmp/census_cache")` | Low -- example code, `/tmp/` is universally available |

### 3.4 Sibling Repository References

The `data_sources.yaml` file references sibling repositories via `${HOME}` variables:

```yaml
source_paths:
  - "${HOME}/projects/ndx-econareas/data/processed/reference/popest/2024/..."
  - "${HOME}/projects/popest/data/raw/counties/totals/co-est2024-alldata.csv"
  - "${HOME}/maps/data/raw/pums_person.parquet"
```

These use environment variable interpolation (supported by the `project_utils` ConfigLoader's `_interpolate_env_vars()`), which is appropriate for cross-repo data sharing. The `scripts/fetch_data.py` script handles the actual copying from these paths.

### 3.5 Shared Data Directory

Census PEP data is stored in a shared workspace directory (`~/workspace/shared-data/census/popest/`) accessed via the `CENSUS_POPEST_DIR` environment variable. This is documented in:
- ADR-034 (`docs/governance/adrs/034-census-pep-data-archive.md`)
- `data/README.md`
- `.env.example`

This is well-architected for avoiding data duplication across projects.

---

## 4. Issues Found

### 4.1 Critical Issues

None.

### 4.2 Medium Issues

**M1: Duplicate ConfigLoader implementations.** Two `ConfigLoader` classes exist with different behaviors:
- `cohort_projections/utils/config_loader.py` resolves config relative to `__file__` (reliable)
- `libs/project_utils/project_utils/config.py` resolves relative to `Path.cwd()` (fragile, depends on working directory)
- The `project_utils` version has `_interpolate_env_vars()` and `get_path()` that the local version lacks
- The local version has `get_parameter()` with `*keys` traversal that the library version lacks

This creates confusion about which loader is active and what features are available. The `__init__.py` imports both and has a fallback mechanism, but callers may not know which one they are using.

**M2: Hardcoded path in `ingest_stcoreview.py`.** The `DEFAULT_INPUT` path (`/mnt/c/Users/nhaarstad/Downloads/...`) is WSL-specific and user-specific. While overridable via CLI argument, the default will fail silently on any other machine.

**M3: Version number inconsistencies.** Three different version numbers exist:
- `pyproject.toml`: `version = "0.1.0"`
- `CLAUDE.md`: `Version 2.3.0`
- `cohort_projections/output/__init__.py`: `__version__ = "1.0.0"`
- `data_sources.yaml` metadata: `version: "1.0.0"`
- `DATA_MANIFEST.md`: `Version: 1.0.0`

There is no single source of truth. The `pyproject.toml` version (0.1.0) is stale and misleading given the maturity of the codebase.

**M4: Fallback paths duplicated between config and code.** At least 5 data file paths appear both in `projection_config.yaml` and as string literals in Python source files. If a path changes in config but not in code (or vice versa), the fallback could silently load stale data.

**M5: Missing DATA_SOURCE_NOTES.md in 5 raw subdirectories.** The `census/`, `census_bureau_methodology/`, `geographic/`, `immigration/`, and `nd_sdc_2024_projections/` directories under `data/raw/` lack DATA_SOURCE_NOTES.md files. The project's own SOP-002 and CLAUDE.md both require updating these notes when adding raw data files.

### 4.3 Low Issues

**L1: Hardcoded county FIPS lists in multiple places.** Oil-patch counties (`38105`, `38053`, `38061`, `38025`, `38089`) appear in:
- `projection_config.yaml` (dampening.counties)
- `residual_migration.py` (OIL_COUNTIES constant)
- `pep_regime_analysis.py` (hardcoded dictionary)

College-town counties similarly appear in both config and code. Config should be the single source.

**L2: Hardcoded population data in geography_loader.py fallback.** The `_create_default_county_reference()` function (line 440-509) contains hardcoded population numbers for 10 counties. These are stale point-in-time values that will become increasingly inaccurate.

**L3: Magic numbers in demographic processors.**
- `migration_rates.py:370-378` -- Rogers-Castro model parameters (a1=0.02, alpha1=0.08, etc.) are not configurable
- `survival_rates.py:979-1004` -- Synthetic life table parameters for testing (qx=0.006 infant, qx=0.0005 child, etc.)
- `fertility.py:169` -- `max_plausible_rate = 0.35` (350 births per 1,000 women)
- `fertility_rates.py:484` -- `max_plausible_rate = 0.15` (150 per 1,000) -- different threshold than core module
- `residual_migration.py:522` -- `cap_value = 0.20` (20% migration rate cap) -- hardcoded rather than from config

**L4: GCP credentials path in projection_config.yaml.** The `bigquery.credentials_path` value (`~/.config/gcloud/cohort-projections-key.json`) should arguably be in `.env` rather than in the tracked config file, since credential paths are user-specific. The `.env.example` documents `GCP_CREDENTIALS_PATH` but it is unclear whether the environment variable overrides the config file value.

**L5: `data_sources.yaml` metadata is stale.** The `metadata.updated` field reads `2025-12-28`, but the projection system has undergone significant changes since then (ADR-035 through ADR-055, Vintage 2025 data). The file still documents some data sources that have been superseded (e.g., SEER life tables replaced by CDC NVSR).

**L6: Methodology constants in `scripts/exports/_methodology.py` duplicate config.** Scenario names, horizon length ("30 years"), and descriptions are defined both in `projection_config.yaml` and in this Python module. Changes to one must be manually propagated to the other.

---

## 5. Assessment: Can an AI Agent Understand the Data Flow?

**Overall: Yes, with effort.** The combination of:
- `config/projection_config.yaml` (primary runtime configuration)
- `config/data_sources.yaml` (data acquisition manifest)
- `data/README.md` (setup guide)
- `data/DATA_MANIFEST.md` (temporal alignment details)
- `data/raw/*/DATA_SOURCE_NOTES.md` (per-domain documentation)
- ADR references throughout config comments

...provides enough context for an AI agent to reconstruct the data flow. However:

1. **The pipeline path is not fully explicit in one place.** An agent must trace from `projection_config.yaml` -> processing scripts -> the core engine to understand which files flow where. The `pipeline` section of the config provides partial guidance but does not enumerate every intermediate file.

2. **Config keys do not always map 1:1 to code locations.** For example, `rates.migration.domestic.dampening.counties` in config feeds into `residual_migration.py`, but the code also has its own `OIL_COUNTIES` constant. An agent would need to understand the fallback/override pattern.

3. **The `data_sources.yaml` and `projection_config.yaml` serve different audiences** -- the former is about data acquisition (where to get external files), the latter about runtime behavior. This separation is reasonable but not immediately obvious.

---

## 6. pyproject.toml Quality Assessment

### Metadata

| Field | Value | Assessment |
|-------|-------|-----------|
| name | `cohort_projections` | Good |
| version | `0.1.0` | **Stale** -- should match CLAUDE.md (2.3.0) or be bumped |
| description | Reasonable | Mentions "2025-2045" but projection horizon is now 2025-2055 |
| license | MIT | Appropriate |
| requires-python | `>=3.11` | Good |
| classifiers | 5 classifiers | Good, includes topic and development status |
| keywords | 5 terms | Good domain-specific keywords |
| URLs | Homepage, Docs, Repo, Issues | All point to GitHub |

### Dependencies

- **Core:** 16 direct dependencies -- reasonable for a data science project
- **Optional groups:** `dev`, `viz`, `geo`, `stats`, `bayesian`, `pdf_export`, `excel_io`, `all` -- well-organized
- **Internal packages:** 3 editable path dependencies (`project-utils`, `evidence-review`, `codebase-catalog`) via `[tool.uv.sources]`
- **Duplicate versions:** `matplotlib`, `scipy`, and `statsmodels` appear in both main dependencies and `[stats]` optional group with matching version bounds. This is documented with a comment ("canonical versions - also in optional stats group") but is technically unnecessary since main deps are always installed.

### Tool Configuration

Ruff, mypy, pytest, and coverage are all configured in `pyproject.toml` with detailed settings. The per-file ruff ignores for legacy/research scripts are thorough and well-commented. The mypy configuration acknowledges a phased strictness roadmap.

---

## 7. Recommendations

### High Priority

**R1: Unify ConfigLoader implementations.** Merge the two `ConfigLoader` classes into a single implementation that includes:
- Reliable path resolution (relative to `__file__`, not `cwd()`)
- Environment variable interpolation (`_interpolate_env_vars`)
- Both `get_parameter(*keys)` and dot-notation `get("a.b.c")` access patterns
- The `get_path()` method for path resolution

Deprecate one of the current implementations and update all imports.

**R2: Synchronize version numbers.** Choose a single source of truth for the version number. Options:
- Use `pyproject.toml` as canonical and read it programmatically elsewhere
- Use a `__version__` in `cohort_projections/__init__.py` and reference it from `pyproject.toml` via dynamic versioning
- At minimum, update `pyproject.toml` from `0.1.0` to the actual current version

**R3: Eliminate fallback path duplication.** Replace hardcoded fallback paths in Python code with a helper that reads from config with a clear error if the key is missing. Example:

```python
# Instead of:
path = config.get("gq_data_path", "data/processed/gq_county_age_sex_2025.parquet")

# Use:
path = config_loader.require("base_population.group_quarters.gq_data_path")
```

This makes config the single source of truth and fails loudly if config is misconfigured.

### Medium Priority

**R4: Add missing DATA_SOURCE_NOTES.md files.** Create notes for:
- `data/raw/census/` (ACS, decennial, PEP API data)
- `data/raw/geographic/` (county/place reference files)
- `data/raw/immigration/` (DHS, refugee, ACS migration -- large directory with ~50 files)
- `data/raw/nd_sdc_2024_projections/` (SDC replication source files)
- `data/raw/census_bureau_methodology/` (methodology reference documents)

**R5: Centralize county group lists.** Move oil-patch counties, college-town counties, and reservation counties into `projection_config.yaml` exclusively. Remove the `OIL_COUNTIES` constant from `residual_migration.py` and the hardcoded dictionaries from `pep_regime_analysis.py`. Have code read these lists from config at runtime.

**R6: Fix the hardcoded WSL path in `ingest_stcoreview.py`.** Replace `DEFAULT_INPUT = Path("/mnt/c/Users/nhaarstad/Downloads/...")` with a default that uses `Path.home() / "Downloads"` or requires the user to pass the path explicitly.

**R7: Update `data_sources.yaml` metadata.** Refresh the `metadata.updated` field and review the data source entries for currency. Several sources now have newer vintages available (Vintage 2025 PEP data, 2023 life tables).

### Low Priority

**R8: Extract Rogers-Castro parameters to config.** The migration age-pattern model parameters (a1, alpha1, a2, alpha2, lambda2, c) are standard demographic constants but should be configurable for sensitivity analysis. Add a `rates.migration.domestic.rogers_castro_params` section to the config.

**R9: Reconcile plausibility thresholds.** `fertility.py` uses `max_plausible_rate = 0.35` while `fertility_rates.py` uses `max_plausible_rate = 0.15`. These different thresholds for the same concept (maximum plausible fertility rate) should be unified and made configurable.

**R10: Remove or gate the hardcoded fallback county data.** The `_create_default_county_reference()` function in `geography_loader.py` with hardcoded population numbers for 10 counties should either be removed (requiring proper geographic reference data) or gated behind a `--use-defaults` flag with a clear warning.

**R11: Update `pyproject.toml` description.** Change "2025-2045" to "2025-2055" to match the current 30-year projection horizon.

**R12: Resolve the dual dependency listing.** Remove `matplotlib`, `scipy`, and `statsmodels` from the main `[project.dependencies]` if they truly belong only in the `[stats]` optional group, or remove them from `[stats]` if they are core requirements.

---

## 8. Summary Scorecard

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Config organization | **A** | Three well-separated YAML files with clear purposes |
| Config documentation | **A-** | Comments in YAML, ADR cross-references, `.env.example` |
| Data directory structure | **A** | Clean pipeline stages, proper gitignoring |
| Data documentation | **B+** | 4 of 9 raw subdirectories have notes; top-level docs are strong |
| Path management | **B** | Config-driven with fallbacks; 3 hardcoded absolute paths |
| Version management | **C** | Three different version numbers with no single source of truth |
| Config loader architecture | **B-** | Works but has duplication between two implementations |
| Data flow traceability | **B+** | Reconstructable from config + ADRs but not in one place |
| pyproject.toml quality | **B+** | Well-structured deps and tool config; stale version and description |
| Overall | **B+** | Solid foundation with addressable maintenance debt |

---

| Attribute | Value |
|-----------|-------|
| **Audit Series** | repo-hygiene-audit |
| **Document Number** | 05 |
| **Focus Area** | Configuration & Data Management |
| **Files Examined** | ~35 config, source, and documentation files |
| **Issues Found** | 5 medium, 6 low |
| **Recommendations** | 12 total (3 high, 4 medium, 5 low priority) |
