# Code Organization Audit

**Date:** 2026-02-26
**Scope:** Python package structure, test mirroring, import health, module sizing, dead code
**Auditor:** Claude Code (automated analysis)

---

## 1. Package Structure Map

Total package size: **17,759 lines** across **35 Python files** in 9 subpackages.

```
cohort_projections/                           (20 lines)  __init__.py  - docstring only, no re-exports
|
+-- config/                                   (300 lines total, 2 files)
|   +-- __init__.py                           (45)   re-exports from race_mappings
|   +-- race_mappings.py                      (255)  canonical race/ethnicity constants
|
+-- core/                                     (1,727 lines total, 5 files)
|   +-- __init__.py                           (42)   re-exports all public API
|   +-- cohort_component.py                   (617)  ** CohortComponentProjection class (587 lines)
|   +-- fertility.py                          (253)  birth calculations
|   +-- migration.py                          (471)  migration application engine
|   +-- mortality.py                          (344)  survival application engine
|
+-- data/                                     (119 lines, 2 files)
|   +-- __init__.py                           (11)   docstring only
|   +-- popest_shared.py                      (108)  POPEST archive CLI helpers
|   |
|   +-- fetch/                                (789 lines, 3 files)
|   |   +-- __init__.py                       (11)   re-exports CensusDataFetcher
|   |   +-- census_api.py                     (646)  Census API client
|   |   +-- vital_stats.py                    (132)  VitalStatsFetcher (partial stub)
|   |
|   +-- load/                                 (2,012 lines, 3 files)
|   |   +-- __init__.py                       (49)   re-exports all loaders
|   |   +-- base_population_loader.py         (1,465) *** LARGEST - base pop + GQ + race dist
|   |   +-- census_age_sex_population.py      (498)  PEP/Census vintage loaders
|   |
|   +-- process/                              (7,551 lines, 10 files)
|       +-- __init__.py                       (151)  re-exports 6 of 8 modules
|       +-- base_population.py                (540)  cohort matrix creation
|       +-- convergence_interpolation.py      (623)  ** convergence pipeline
|       +-- example_usage.py                  (286)  *** DEAD CODE - example script
|       +-- fertility_rates.py                (722)  ** SEER fertility processing
|       +-- migration_rates.py                (1,963) *** LARGEST MODULE IN PROJECT
|       +-- mortality_improvement.py          (482)  NP2023 survival improvement
|       +-- pep_regime_analysis.py            (457)  oil/metro county regimes
|       +-- residual_migration.py             (1,309) *** second largest
|       +-- survival_rates.py                 (1,018) ** life table processing
|
+-- geographic/                               (1,617 lines, 3 files)
|   +-- __init__.py                           (81)   re-exports all public API
|   +-- geography_loader.py                   (696)  ** FIPS code loading + defaults
|   +-- multi_geography.py                    (840)  ** projection orchestrator
|
+-- output/                                   (2,467 lines, 4 files + empty templates/)
|   +-- __init__.py                           (76)   re-exports all + stale __version__
|   +-- reports.py                            (858)  ** HTML/text report generation
|   +-- visualizations.py                     (832)  ** matplotlib charts
|   +-- writers.py                            (701)  ** Excel/CSV/shapefile export
|   +-- templates/                            (empty, .gitkeep placeholder)
|
+-- utils/                                    (1,152 lines, 5 files)
|   +-- __init__.py                           (155)  complex: project_utils fallback + BigQuery stub
|   +-- bigquery_client.py                    (356)  BigQuery wrapper
|   +-- config_loader.py                      (129)  YAML config loading
|   +-- demographic_utils.py                  (380)  helper functions (mostly unused)
|   +-- reproducibility.py                    (132)  @log_execution decorator
|
+-- version.py                                (5)    __version__ = "0.1.0" (orphaned)
```

### Size distribution by subpackage

| Subpackage | Lines | Files | Avg lines/file |
|---|---:|---:|---:|
| `data/process/` | 7,551 | 10 | 755 |
| `output/` | 2,467 | 4 | 617 |
| `data/load/` | 2,012 | 3 | 671 |
| `core/` | 1,727 | 5 | 345 |
| `geographic/` | 1,617 | 3 | 539 |
| `utils/` | 1,152 | 5 | 230 |
| `data/fetch/` | 789 | 3 | 263 |
| `config/` | 300 | 2 | 150 |
| `data/` (top-level) | 119 | 2 | 60 |

The `data/process/` subpackage accounts for **42%** of all package code. Two modules alone (`migration_rates.py` at 1,963 and `residual_migration.py` at 1,309) total 3,272 lines, or 18% of the entire package.

---

## 2. Test Structure Map

Total test code: **24,976 lines** across **49 Python files** (1.41x ratio to package code).

```
tests/
+-- conftest.py                               (305)  shared fixtures
+-- __init__.py                               (0)
|
+-- test_config/                              (214 lines, 2 files)
|   +-- test_race_mappings.py                 (213)
|
+-- test_core/                                (3,604 lines, 5 files)
|   +-- test_cohort_component.py              (923)
|   +-- test_fertility.py                     (501)
|   +-- test_migration.py                     (1,114)
|   +-- test_mortality.py                     (579)
|   +-- test_time_varying_engine.py           (487)
|
+-- test_data/                                (8,744 lines, 12 files)
|   +-- test_base_population.py               (655)
|   +-- test_bebr_averaging.py                (739)
|   +-- test_census_api.py                    (303)
|   +-- test_convergence_interpolation.py     (1,118)
|   +-- test_county_race_distributions.py     (1,131)
|   +-- test_fertility_rates.py               (651)
|   +-- test_migration_rates.py               (849)
|   +-- test_mortality_improvement.py         (527)
|   +-- test_pep_migration_rates.py           (772)
|   +-- test_pep_regime_analysis.py           (498)
|   +-- test_residual_migration.py            (1,312)
|   +-- test_survival_rates.py                (779)
|
+-- test_geographic/                          (1,434 lines, 2 files)
|   +-- test_geography_loader.py              (628)
|   +-- test_multi_geography.py               (806)
|
+-- test_integration/                         (1,796 lines, 4 files)
|   +-- test_census_method_validation.py      (569)
|   +-- test_end_to_end.py                    (738)
|   +-- test_pep_pipeline.py                  (488)
|
+-- test_output/                              (1,959 lines, 3 files)
|   +-- test_reports.py                       (610)
|   +-- test_visualizations.py                (755)
|   +-- test_writers.py                       (594)
|
+-- test_statistical/                         (1,631 lines, 5 files)
|   +-- conftest.py                           (231)  adds sdc_2024_replication/ to sys.path
|   +-- test_bayesian_var.py                  (718)
|   +-- test_covariate_anchor.py              (95)
|   +-- test_multistate_placebo.py            (587)
|   +-- test_regime_aware.py                  (580)
|
+-- test_tools/                               (198 lines, 1 file)
|   +-- test_citation_audit.py                (198)
|
+-- test_utils/                               (1,994 lines, 5 files)
|   +-- conftest.py                           (190)
|   +-- test_bigquery_client.py               (432)
|   +-- test_config_loader.py                 (270)
|   +-- test_demographic_utils.py             (785)
|   +-- test_reproducibility.py               (317)
|
+-- unit/                                     (1,918 lines, 9 files)
    +-- test_adr021_modules.py                (1,063)
    +-- test_build_dhs_lpr_panel_variants.py  (71)
    +-- test_duration_figure_table_consistency.py (101)
    +-- test_journal_article_derived_stats.py (186)
    +-- test_journal_article_versioning.py    (120)
    +-- test_module_7_causal_inference.py     (65)
    +-- test_module_8_duration_analysis.py    (134)
    +-- test_popest_archive_and_fingerprints.py (65)
    +-- test_sdc_data_loader.py               (111)
```

### Test-to-package mirroring

| Package subpackage | Test directory | Status |
|---|---|---|
| `config/` | `test_config/` | Mirrored |
| `core/` | `test_core/` | Mirrored |
| `data/` | `test_data/` | Mirrored (flat -- does not separate fetch/load/process) |
| `geographic/` | `test_geographic/` | Mirrored |
| `output/` | `test_output/` | Mirrored |
| `utils/` | `test_utils/` | Mirrored |
| (none) | `test_integration/` | Cross-cutting integration tests (appropriate) |
| (none) | `test_statistical/` | Tests sibling repo `sdc_2024_replication/` |
| (none) | `test_tools/` | Tests for standalone tools |
| (none) | `unit/` | Tests sibling repos (various) |

---

## 3. Issues Found

### 3.1 Oversized Modules (>500 lines)

| Module | Lines | Concern |
|---|---:|---|
| `data/process/migration_rates.py` | **1,963** | 17 functions spanning IRS loading, distribution algorithms, BEBR scenarios, dampening, validation, and PEP rate processing. This is by far the largest module and does too many things. |
| `data/load/base_population_loader.py` | **1,465** | 18 functions/helpers. Mixes three distinct responsibilities: (1) race distribution loading, (2) group quarters population management, and (3) base population assembly for counties/state. |
| `data/process/residual_migration.py` | **1,309** | 14 functions. The `run_residual_migration_pipeline()` function alone is 305 lines. |
| `data/process/survival_rates.py` | **1,018** | 8 functions. More justifiable since all are survival-related, but `process_survival_rates()` at 154 lines and `validate_survival_rates()` at 184 lines are large. |
| `output/reports.py` | **858** | `generate_summary_statistics()` is 247 lines and `_build_html_report()` is 216 lines. |
| `geographic/multi_geography.py` | **840** | `run_multi_geography_projections()` is 252 lines and `run_single_geography_projection()` is 218 lines. |
| `output/visualizations.py` | **832** | 6 plot functions. Reasonable size per function. |
| `data/process/fertility_rates.py` | **722** | 6 functions, well-organized. Borderline but acceptable. |
| `output/writers.py` | **701** | 4 export functions. Acceptable. |
| `geographic/geography_loader.py` | **696** | 11 functions including hardcoded default county/place data (146 lines for `_create_default_nd_counties` + `_create_default_nd_places`). Data should be externalized. |
| `data/fetch/census_api.py` | **646** | Single class. Acceptable. |
| `data/process/convergence_interpolation.py` | **623** | 6 functions. `run_convergence_pipeline()` is 222 lines -- a candidate for decomposition. |
| `core/cohort_component.py` | **617** | Single class (587 lines). Well-structured with clean method decomposition. Acceptable. |

**Priority splits:**
1. `migration_rates.py` (1,963 lines) -- split into `migration_loading.py`, `migration_distribution.py`, `migration_scenarios.py`
2. `base_population_loader.py` (1,465 lines) -- extract `gq_population.py` (~200 lines for GQ functions) and `race_distribution_loader.py` (~400 lines)
3. `residual_migration.py` (1,309 lines) -- extract `residual_adjustments.py` (dampening, college-age, male adjustment, PEP recalibration) from the main pipeline

### 3.2 Dead or Orphaned Code

| File | Lines | Issue |
|---|---:|---|
| `data/process/example_usage.py` | 286 | **Dead code.** Never imported anywhere. An example script that was placed inside the package instead of in `scripts/` or `examples/`. Should be deleted or moved to `examples/`. |
| `version.py` | 5 | **Orphaned.** Defines `__version__ = "0.1.0"` but is never imported by any module. The package version is defined in `pyproject.toml`. |
| `data/fetch/vital_stats.py` | 132 | **Stub/dead code.** `VitalStatsFetcher` has a working `fetch_fertility_rates()` that falls back to mock data, and `fetch_mortality_rates()` is `raise NotImplementedError`. Never imported by any non-test code. The `fetch/__init__.py` does not export it. |
| `output/__init__.py` `__version__` | 1 | **Stale version.** Declares `__version__ = "1.0.0"` which conflicts with both `version.py` (`0.1.0`) and `pyproject.toml` (`0.1.0`). |
| `output/templates/` | 0 | **Empty directory.** Contains only a `.gitkeep`. May be intentional (future use) but has been empty for months. |

### 3.3 Unused Functions in `demographic_utils.py`

Of the 10 functions in `cohort_projections/utils/demographic_utils.py` (380 lines), only **1** is used by non-test code:

| Function | Used in production? | Used in tests? |
|---|---|---|
| `sprague_graduate()` | Yes (`base_population_loader.py`) | Yes |
| `create_age_groups()` | No | Yes |
| `calculate_sex_ratio()` | No | Yes |
| `calculate_dependency_ratio()` | No (duplicate exists in `cohort_component.py`) | Yes |
| `calculate_median_age()` | No (duplicate exists in `cohort_component.py` and `reports.py`) | Yes |
| `interpolate_missing_ages()` | No | Yes |
| `aggregate_race_categories()` | No | Yes |
| `calculate_growth_rate()` | No | Yes |
| `validate_cohort_sums()` | No | Yes |
| `_pad_groups()` | Internal to `sprague_graduate` | N/A |

The unused functions have tests but no production callers. This is a library of demographic utilities written speculatively -- either they should be wired into production code or their tests are exercising dead code.

### 3.4 Duplicate Function Implementations

Three functions are implemented in multiple places:

1. **`calculate_median_age()`** -- defined in:
   - `utils/demographic_utils.py` (standalone function, unused)
   - `core/cohort_component.py` (method `_calculate_median_age`)
   - `output/reports.py` (private function `_calculate_median_age`)

2. **`calculate_dependency_ratio()`** -- defined in:
   - `utils/demographic_utils.py` (standalone function, unused)
   - `core/cohort_component.py` (method `_calculate_dependency_ratio`)

3. **Race mapping re-exports** -- `MIGRATION_RACE_MAP` is:
   - Canonically defined in `config/race_mappings.py`
   - Re-exported via `config/__init__.py`
   - Imported and re-exported in `data/process/migration_rates.py` for backward compatibility
   - Re-exported via `data/process/__init__.py`

   Similarly, `SEER_RACE_ETHNICITY_MAP` in `fertility_rates.py` and `RACE_ETHNICITY_MAP` in `base_population.py` are just aliases for the canonical config constants.

### 3.5 Naming and Placement Issues

| Item | Issue |
|---|---|
| `data/popest_shared.py` | Lives at `data/` top level (not in `data/fetch/`, `data/load/`, or `data/process/`). Only used by 4 CLI scripts in `scripts/data/`. Would be better placed in a `scripts/data/_shared.py` or `data/fetch/popest_shared.py`. |
| `data/process/example_usage.py` | An example script living inside the installable package. Should be `examples/base_population_example.py` or deleted. |
| `scripts/exports/_methodology.py` | Contains methodology text constants shared by export scripts. Uses `from _methodology import ...` (fragile, depends on `sys.path`). Could be `cohort_projections/config/methodology.py` for clean imports. |
| `test_statistical/` and `unit/` | Test sibling repositories (`sdc_2024_replication/`, journal articles) by adding paths to `sys.path`. These tests do not test `cohort_projections` at all and belong in the sibling repos. |
| `data/load/` vs `data/fetch/` | The naming distinction is unclear. `fetch/` implies network I/O, `load/` implies file I/O, but `census_api.py` (in `fetch/`) also loads from local files as a fallback. The boundary is reasonable but could be documented. |

### 3.6 Version String Chaos

There are **four** different version strings in the project:

| Location | Version | Purpose |
|---|---|---|
| `pyproject.toml` | `0.1.0` | The authoritative package version |
| `cohort_projections/version.py` | `0.1.0` | Orphaned, never imported |
| `cohort_projections/output/__init__.py` | `1.0.0` | Stale, contradicts pyproject.toml |
| `CLAUDE.md` | `2.3.0` | Documentation version, semantically different |

---

## 4. Import Health Assessment

### 4.1 Circular Imports

**No circular import cycles detected.** The dependency graph is clean and acyclic.

One near-cycle exists between `residual_migration.py` and `migration_rates.py`, but it is handled correctly via a **lazy import** inside a function body (`_get_rogers_castro_age_group_weights()` at line 592 imports `get_standard_age_migration_pattern` at call time, not module load time).

### 4.2 Dependency Flow

The subpackage-level dependency graph is well-layered:

```
config  <----+
  |          |
  v          |
utils  <-----+---+---+---+
  |          |   |   |   |
  v          |   |   |   |
core  -------+   |   |   |
  |              |   |   |
  v              |   |   |
data/load  ------+   |   |
  |                  |   |
  v                  |   |
data/process  -------+   |
  |                      |
  v                      |
geographic  -------------+
  |
  v
output
```

All subpackages depend on `utils/` (for logging and config). `geographic/` depends on `core/` and `data/load/`, which is appropriate for an orchestration layer. `output/` is a leaf -- it depends only on `utils/`.

There are **no upward imports** (e.g., `core/` does not import from `data/` or `geographic/`).

### 4.3 Import Style Consistency

Two import styles are used, split by subpackage:

| Subpackage | Style | Example |
|---|---|---|
| `core/` | **Relative** | `from ..utils import get_logger_from_config` |
| `output/` | **Relative** | `from ..utils import get_logger_from_config` |
| `data/fetch/vital_stats.py` | **Relative** | `from ...utils import ConfigLoader` |
| `data/process/` | **Absolute** | `from cohort_projections.utils import ...` |
| `data/load/` | **Absolute** | `from cohort_projections.utils import ...` |
| `geographic/` | **Absolute** | `from cohort_projections.utils import ...` |

No single file mixes relative and absolute imports. However, the project would benefit from picking one style. The absolute style is dominant (used in 13 of 18 non-`__init__` modules) and is recommended by PEP 8 for applications.

### 4.4 `__init__.py` Export Quality

| Package | Quality | Notes |
|---|---|---|
| `cohort_projections/__init__.py` | Minimal | Docstring only, no re-exports. Acceptable for a top-level namespace. |
| `config/__init__.py` | Good | Clean `__all__`, re-exports all public API. |
| `core/__init__.py` | Good | Clean `__all__`, re-exports all public API. |
| `data/__init__.py` | Minimal | Docstring only. Sub-subpackages handle their own exports. |
| `data/fetch/__init__.py` | Good | Exports `CensusDataFetcher`. Does not export `VitalStatsFetcher` (appropriate since it is a stub). |
| `data/load/__init__.py` | Good | Exports all public loaders with `__all__`. |
| `data/process/__init__.py` | Bloated | 151 lines. Re-exports **50+ symbols** from 6 modules. Does NOT export `convergence_interpolation` or `residual_migration` -- both are imported directly by consumers. This creates an inconsistency: some data/process modules go through `__init__`, others don't. |
| `geographic/__init__.py` | Good | Clean `__all__` with all public functions. |
| `output/__init__.py` | Good | Clean `__all__`. Has stale `__version__`. |
| `utils/__init__.py` | Complex | 155 lines of conditional imports, fallback classes, and compatibility shims. Works but is harder to maintain. |

---

## 5. Scripts Analysis

Total scripts code: **20,531 lines** across **40 Python files**.

### 5.1 Code That Should Move to the Package

| Script | Lines | Candidate logic |
|---|---:|---|
| `pipeline/02_run_projections.py` | 2,084 | Contains rate-transformation functions (`_transform_fertility_rates`, `_transform_survival_rates`, `_transform_migration_rates`, `expand_5yr_migration_to_engine_format`, `_build_convergence_rate_dicts`, `_build_survival_rates_by_year`) and `aggregate_county_results_to_state()` (148 lines). These are reusable data-shaping functions that bridge data/process output to core engine input. A `cohort_projections/data/transform/` module would be appropriate. |
| `pipeline/03_export_results.py` | 1,151 | Contains generic summary-table generators (`create_total_population_summary`, `create_age_distribution_summary`, etc.) and `generate_data_dictionary()` (152 lines). These could supplement `output/reports.py`. |
| `data/build_race_distribution_from_census.py` | 1,094 | Contains `build_distribution()`, `build_single_year_statewide_distribution()`, and `build_county_distributions()` with validation. These are data processing functions that produce files consumed by `base_population_loader.py`. Could live in `data/process/`. |
| `exports/_methodology.py` | 97 | Shared methodology constants. Using `from _methodology import ...` is fragile. This should be `cohort_projections/config/methodology.py`. |

### 5.2 Scripts Testing Sibling Repos

The following test directories test code that lives **outside** the `cohort_projections` package:

| Test directory | What it tests | Location of tested code |
|---|---|---|
| `tests/test_statistical/` | Bayesian VAR, multistate placebo, regime analysis | `../sdc_2024_replication/scripts/statistical_analysis/` |
| `tests/unit/` (most files) | ADR-021 modules, journal article code, DHS/LPR panels | `../sdc_2024_replication/scripts/` and similar |

These tests use `sys.path.insert()` to reach sibling repository code. They add **3,549 lines** to the test suite (14.2% of total). While running them here may be convenient, they logically belong in the repositories they test.

---

## 6. Separation of Concerns Assessment

| Layer | Subpackage(s) | Assessment |
|---|---|---|
| **Configuration** | `config/`, `utils/config_loader.py` | Clean. Race mappings in `config/`, YAML loading in `utils/`. |
| **Data Loading** | `data/load/`, `data/fetch/` | Adequate. `fetch/` handles API/network sources, `load/` handles file loading. |
| **Data Processing** | `data/process/` | Largest layer (7,551 lines). Contains clear domain modules (fertility, survival, migration) but migration is split across 3 modules (`migration_rates.py`, `residual_migration.py`, `pep_regime_analysis.py`) without clear boundaries. |
| **Core Engine** | `core/` | Clean. The cohort-component engine with clear fertility/mortality/migration separation. |
| **Orchestration** | `geographic/` | Clean. Handles multi-geography projection coordination. |
| **Output** | `output/` | Clean. Reports, visualizations, and file writers. |
| **Utilities** | `utils/` | Adequate but contains speculative code (`demographic_utils.py` functions mostly unused). |

**Missing layer:** There is no explicit `data/transform/` layer for reshaping processed data into engine-ready formats. This logic currently lives in `scripts/pipeline/02_run_projections.py` (rate transformations, age-group expansion, convergence rate dictionary building). The gap forces the pipeline script to contain ~500 lines of reusable data transformation code.

---

## 7. Recommendations

### Priority 1: Remove Dead Code

1. **Delete `data/process/example_usage.py`** (286 lines). It is never imported. If needed for documentation, move to `examples/`.
2. **Delete `version.py`** (5 lines). It is orphaned. The canonical version is in `pyproject.toml`.
3. **Remove `__version__ = "1.0.0"` from `output/__init__.py`**. It contradicts `pyproject.toml`.
4. **Evaluate `data/fetch/vital_stats.py`** (132 lines). `VitalStatsFetcher` is unused. If still needed as a development scaffold, document it. If not, delete it.

### Priority 2: Split Oversized Modules

1. **Split `data/process/migration_rates.py`** (1,963 lines) into:
   - `migration_loading.py` -- `load_irs_migration_data()`, `load_international_migration_data()`
   - `migration_distribution.py` -- `get_standard_age_migration_pattern()`, `distribute_migration_by_age/sex/race()`
   - `migration_scenarios.py` -- `calculate_period_average()`, `calculate_multiperiod_averages()`, `calculate_bebr_scenarios()`, `apply_county_dampening()`, `calculate_interpolated_rates()`
   - Keep `migration_rates.py` as the pipeline entry point with `process_migration_rates()`, `process_pep_migration_rates()`, `validate_migration_data()`

2. **Extract GQ logic from `data/load/base_population_loader.py`** (1,465 lines):
   - Move `get_county_gq_population()`, `get_all_county_gq_populations()`, `clear_gq_cache()`, `_load_gq_data()`, `_expand_gq_to_single_year_ages()`, `_distribute_gq_across_races()`, `_separate_gq_from_base_population()` (~200 lines) to a new `data/load/gq_population.py`.

3. **Decompose `run_residual_migration_pipeline()`** in `residual_migration.py` (305 lines). Extract setup, rate computation, and adjustment phases into named helper functions.

### Priority 3: Consolidate Duplicates

1. **Deduplicate `calculate_median_age()`** -- make `core/cohort_component.py` and `output/reports.py` call `utils/demographic_utils.calculate_median_age()` instead of maintaining private copies.
2. **Remove race-mapping aliases** -- `SEER_RACE_ETHNICITY_MAP` in `fertility_rates.py` and `RACE_ETHNICITY_MAP` in `base_population.py` are trivial aliases for `config/` constants. Import directly from `config/` and deprecate the aliases.
3. **Remove backward-compatibility re-exports** of `MIGRATION_RACE_MAP` from `data/process/migration_rates.py` after checking external consumers.

### Priority 4: Standardize Import Style

Pick absolute imports project-wide (currently dominant at 13 of 18 modules). Update the 5 modules in `core/` and `output/` that use relative imports:
- `core/cohort_component.py`
- `core/fertility.py`
- `core/migration.py`
- `core/mortality.py`
- `output/reports.py`
- `output/visualizations.py`
- `output/writers.py`
- `data/fetch/vital_stats.py`

### Priority 5: Structural Improvements

1. **Create `data/transform/` subpackage** for engine-ready data shaping (extract from `02_run_projections.py`).
2. **Move `data/popest_shared.py`** to `data/fetch/popest_shared.py` or `scripts/data/_popest_shared.py` (it is only used by scripts).
3. **Move `scripts/exports/_methodology.py`** to `cohort_projections/config/methodology.py`.
4. **Externalize hardcoded county/place defaults** from `geographic/geography_loader.py` (146 lines of hardcoded data) into a data file (CSV or YAML in `config/`).
5. **Make `data/process/__init__.py` exports consistent** -- either export all modules (including `convergence_interpolation` and `residual_migration`) or export none and rely on direct imports.

### Priority 6: Test Organization

1. **Move sibling-repo tests** (`tests/test_statistical/` and most of `tests/unit/`) to their respective repositories. This removes ~3,500 lines of unrelated tests.
2. **Add dedicated test files** for `data/load/base_population_loader.py` and `data/load/census_age_sex_population.py` -- currently tested only indirectly via other tests.
3. **Flatten test_data/** or add sub-directories to mirror `data/fetch/`, `data/load/`, `data/process/`. Currently all 12 test files sit flat under `test_data/` even though the source modules span three subpackages.

---

## 8. Summary Metrics

| Metric | Value | Assessment |
|---|---|---|
| Package line count | 17,759 | Moderate |
| Test line count | 24,976 | Good test-to-code ratio (1.41x) |
| Script line count | 20,531 | Heavy; contains reusable logic |
| Circular imports | 0 | Excellent |
| Dead/orphaned modules | 3 (423 lines) | Should be cleaned up |
| Modules >1,000 lines | 3 | Should be split |
| Modules >500 lines | 13 of 28 non-init modules | Heavy upper half |
| Duplicate implementations | 3 functions | Should be consolidated |
| Unused utility functions | 8 of 10 in `demographic_utils.py` | Speculative code |
| Import style consistency | Mixed (absolute dominant) | Should standardize |
| Version string consistency | 4 conflicting values | Should consolidate |
| Sibling-repo tests | ~3,500 lines (14% of tests) | Should relocate |

**Overall assessment:** The package has a sound architectural layering with no circular dependencies and clear separation between core engine, data processing, geographic orchestration, and output. The main structural debts are oversized migration-related modules, dead code, and reusable logic trapped in pipeline scripts. These are typical for a project that has grown organically and are straightforward to address incrementally.
