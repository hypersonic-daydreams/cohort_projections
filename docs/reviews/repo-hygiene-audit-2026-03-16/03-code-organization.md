# Code Organization Audit

**Date:** 2026-03-16
**Scope:** Python package structure, dead code, import hygiene, file sizing, naming, scripts organization, duplication
**Auditor:** Claude Code (automated analysis)
**Prior audit:** 2026-02-26 (docs/reviews/repo-hygiene-audit/04-code-organization.md)

---

## Summary of Findings

| # | Finding | Severity | Status vs Feb Audit |
|---|---------|----------|---------------------|
| 1 | `analysis/` subpackage now dominates the codebase at 21,294 lines (48% of package) | WARNING | New -- growth since Feb |
| 2 | Observatory dashboard has 5 modules over 700 lines; `tab_command_center.py` at 1,464 lines | WARNING | New |
| 3 | `data/process/example_usage.py` still dead code (286 lines) | WARNING | Unfixed from Feb |
| 4 | `data/fetch/vital_stats.py` still a stub, never imported | WARNING | Unfixed from Feb |
| 5 | `_normalize_fips()` duplicated in 4 modules (place_backtest, multicounty_allocation, place_shares, place_projection_orchestrator) | WARNING | New |
| 6 | `validate_*_rates()` / `validate_migration_data()` duplicated across `core/` and `data/process/` (different signatures and scopes) | INFO | Known from Feb |
| 7 | Two `generate_html_report()` functions in `output/reports.py` and `analysis/evaluation/html_report.py` | INFO | Known from Feb |
| 8 | `data/process/migration_rates.py` still 1,963 lines -- largest package module | WARNING | Unfixed from Feb |
| 9 | `data/load/base_population_loader.py` still 1,471 lines | WARNING | Unfixed from Feb |
| 10 | `data/process/__init__.py` at 262 lines, re-exports ~60 symbols inconsistently | INFO | Worsened from Feb |
| 11 | Mixed import styles (relative in core/output, absolute elsewhere) persist | INFO | Unfixed from Feb |
| 12 | `scripts/data/test_ves_extraction.py` is a test file in `scripts/data/` | INFO | New |
| 13 | 8 root-level scripts in `scripts/` should be grouped into subdirectories | INFO | Known from Feb |
| 14 | `scripts/data/` and `scripts/data_processing/` overlap in purpose | INFO | Known from Feb |
| 15 | `scripts/intelligence/link_documentation.py` has two TODO stubs (heuristics 2 & 3) | INFO | Pre-existing |
| 16 | `scripts/pipeline/02_run_projections.py` at 2,103 lines contains reusable transform logic | WARNING | Known from Feb |
| 17 | `version.py` was fixed (now reads from pyproject.toml), `output/__init__.py` stale version removed | -- | Fixed since Feb |
| 18 | `demographic_utils.py` deduplication completed (`calculate_median_age`, `calculate_dependency_ratio` now used from canonical location) | -- | Fixed since Feb |
| 19 | No circular imports detected | -- | Maintained |
| 20 | No star imports in any package module | -- | Maintained |
| 21 | No TODO/FIXME/HACK/XXX comments in `cohort_projections/` package | -- | Clean |

---

## 1. Module Structure

### Package Hierarchy

Total package size: **43,994 lines** across **76 Python files** (up from 17,759 lines / 35 files in Feb).

```
cohort_projections/                              (24 lines)  __init__.py
|
+-- analysis/                                    (21,294 lines, 34 files)  *** LARGEST SUBPACKAGE
|   +-- __init__.py                              (35)   re-exports from benchmarking
|   +-- benchmark_contract.py                    (new)  benchmark data contract + validation
|   +-- benchmarking.py                          (grew) versioned benchmarking + promotion
|   +-- evaluation_policy.py                     (new)  gate/threshold evaluation rules
|   +-- experiment_log.py                        experiment metadata tracking
|   +-- evaluation/                              (5,199 lines, 12 files)
|   |   +-- data_structures.py, metrics.py, etc.
|   +-- observatory/                             (13,945 lines, 23 files)  *** RAPID GROWTH
|       +-- candidates.py, comparator.py, recommender.py, etc.
|       +-- dashboard/                           (9,004 lines, 11 files)
|           +-- tab_command_center.py            (1,464 lines)
|           +-- data_manager.py                  (1,312 lines)
|           +-- tab_horizon_bias.py              (1,110 lines)
|           +-- 8 more dashboard modules
|
+-- config/                                      (~300 lines, 2 files)   Stable
+-- core/                                        (~1,750 lines, 5 files) Stable
+-- data/                                        (~12,600 lines, 18 files)
|   +-- popest_shared.py                         (misplaced -- only used by scripts/)
|   +-- fetch/                                   vital_stats.py still a stub
|   +-- load/                                    base_population_loader.py still 1,471 lines
|   +-- process/                                 example_usage.py still dead code
|
+-- geographic/                                  (~1,700 lines, 3 files) Stable
+-- output/                                      (~2,500 lines, 4 files) Stable
+-- utils/                                       (~1,200 lines, 5 files) Stable
+-- version.py                                   (24 lines) Fixed since Feb
```

### Size Distribution by Subpackage

| Subpackage | Lines | Files | % of Package | Change Since Feb |
|---|---:|---:|---:|---|
| `analysis/observatory/dashboard/` | 9,004 | 11 | 20.5% | New |
| `analysis/observatory/` (non-dashboard) | 4,941 | 12 | 11.2% | New |
| `analysis/evaluation/` | 5,199 | 12 | 11.8% | Existed |
| `data/process/` | ~9,400 | 14 | 21.4% | Grew (new place modules) |
| `data/load/` | ~2,000 | 3 | 4.5% | Stable |
| `core/` | ~1,750 | 5 | 4.0% | Stable |
| Everything else | ~11,700 | 19 | 26.6% | -- |

**Key observation:** The `analysis/` subpackage has grown from 0 lines (not yet created in the Feb audit) to 21,294 lines -- now 48% of the entire package. The observatory alone (13,945 lines across 23 files) is larger than the original package was in February.

### `__init__.py` Assessment

All 14 subpackages have `__init__.py` files. Quality is generally good:

- **Clean:** `config/`, `core/`, `data/fetch/`, `data/load/`, `geographic/`, `output/`, `analysis/observatory/`, `analysis/observatory/dashboard/`
- **Bloated:** `data/process/__init__.py` (262 lines, ~60 re-exported symbols, inconsistently excludes `convergence_interpolation` and `residual_migration`)
- **Complex but functional:** `utils/__init__.py` (150 lines, conditional project_utils fallback)

---

## 2. Dead and Orphaned Code

### Confirmed Dead Code

| File | Lines | Evidence | Severity |
|---|---:|---|---|
| `data/process/example_usage.py` | 286 | Only referenced by a lazy-load helper in `data/process/__init__.py` (`load_example_usage_module()`). No production caller. Never run. | WARNING |
| `data/fetch/vital_stats.py` | 133 | Zero imports across the entire codebase. Not exported by `data/fetch/__init__.py`. Contains `VitalStatsFetcher` with a `fetch_mortality_rates()` that raises `NotImplementedError`. | WARNING |

### Stale TODO Comments

| File | Line | Content | Severity |
|---|---:|---|---|
| `scripts/pipeline/02_run_projections.py` | 1522 | `# TODO: Implement actual base population loading per geography` | INFO |
| `scripts/intelligence/link_documentation.py` | 11-12 | Two `(TODO)` markers for unimplemented heuristics 2 and 3 | INFO |

### Misplaced Files

| File | Issue | Severity |
|---|---|---|
| `scripts/data/test_ves_extraction.py` | A test/validation script living in `scripts/data/` rather than `tests/` or `scripts/validation/`. Contains PDF extraction validation logic. | INFO |
| `scripts/run_integration_test.py` | An integration test script (621 lines) at the scripts root, separate from the formal `tests/` directory. | INFO |

### Commented-Out Code

No significant commented-out code blocks found in either the package or scripts. One benign comment in `scripts/analysis/uncertainty_analysis.py:1128` serves as a section header.

---

## 3. Import Hygiene

### Star Imports

**None found.** Zero star imports across the entire `cohort_projections/` package.

### Unused Imports

AST analysis of 10 sampled modules found only one unused import:

| Module | Import | Note |
|---|---|---|
| `analysis/benchmarking.py` | `from __future__ import annotations` | Technically unused (no forward references needed), but harmless and conventional |

### Circular Import Risk

**No circular imports detected.** The dependency graph remains clean and acyclic:

```
config  <-----+
  |            |
  v            |
utils  <-------+---+---+---+---+
  |            |   |   |   |   |
  v            |   |   |   |   |
core  ---------+   |   |   |   |
  |                |   |   |   |
  v                |   |   |   |
data/load  --------+   |   |   |
  |                    |   |   |
  v                    |   |   |
data/process  ---------+   |   |
  |                        |   |
  v                        |   |
geographic  ---------------+   |
  |                            |
  v                            |
output  -----------------------+
  |
analysis  (depends on: utils, observatory internals, benchmarking)
```

Critical: `core/` does not import from `data/`, `geographic/`, `output/`, or `analysis/`. The no-upward-import rule is maintained.

### Import Style Consistency

The mixed style persists. Relative imports in `core/` and `output/`; absolute imports everywhere else (including all new `analysis/` code):

| Style | Modules Using It |
|---|---|
| **Absolute** (`from cohort_projections.X import Y`) | `data/process/*`, `data/load/*`, `geographic/*`, `analysis/*` (all 34 files) |
| **Relative** (`from ..X import Y`) | `core/*` (4 files), `output/*` (3 files), `data/fetch/vital_stats.py` |
| **Mixed within observatory** | `__init__.py` files use relative; non-init files use absolute |

No module mixes both styles internally. The observatory subpackage is internally consistent (absolute for cross-module, relative for `__init__` re-exports).

---

## 4. File Size Concerns

### Package Modules Over 500 Lines

| Module | Lines | Concern | Priority |
|---|---:|---|---|
| `data/process/migration_rates.py` | **1,963** | Unchanged since Feb. 17 functions spanning IRS loading, distribution, BEBR scenarios, dampening, validation. | HIGH |
| `data/process/place_projection_orchestrator.py` | **1,786** | New since Feb. Complex orchestration -- reasonable for an orchestrator but could extract helpers. | MEDIUM |
| `data/load/base_population_loader.py` | **1,471** | Unchanged since Feb. Mixes race distribution loading, GQ management, and base pop assembly. | HIGH |
| `observatory/dashboard/tab_command_center.py` | **1,464** | New. UI tab with search controls, queue management, execution monitoring. | MEDIUM |
| `observatory/report.py` | **1,385** | New. HTML report generation. | LOW |
| `analysis/evaluation/html_report.py` | **1,360** | HTML report generator for evaluation framework. | LOW |
| `data/process/residual_migration.py` | **1,316** | Slightly grew. Main pipeline function still large. | MEDIUM |
| `observatory/dashboard/data_manager.py` | **1,312** | New. Dashboard state management (two classes). | MEDIUM |
| `observatory/dashboard/tab_horizon_bias.py` | **1,110** | New. Horizon bias visualization tab. | LOW |
| `observatory/search_controller.py` | **1,085** | New. Autonomous search loop orchestration. | MEDIUM |
| `data/process/survival_rates.py` | **1,018** | Stable. | LOW |

**15 more modules** are in the 500-1,000 line range.

### Scripts Over 1,500 Lines

| Script | Lines | Purpose |
|---|---:|---|
| `exports/build_walk_forward_report.py` | 2,998 | Interactive HTML report builder |
| `analysis/walk_forward_validation.py` | 2,992 | Walk-forward validation pipeline |
| `analysis/observatory.py` | 2,794 | Observatory CLI entry point |
| `exports/build_qc_report.py` | 2,563 | QC analysis HTML report |
| `exports/build_sdc_replication_comparison.py` | 2,117 | SDC comparison report |
| `pipeline/02_run_projections.py` | 2,103 | Main projection pipeline (contains reusable transforms) |
| `exports/build_sensitivity_report.py` | 2,072 | Sensitivity analysis report |
| `analysis/uncertainty_analysis.py` | 1,964 | Uncertainty quantification |
| `reviews/build_pp3_human_review_package.py` | 1,905 | Human review report builder |
| `exports/build_sdc_comparison_report.py` | 1,847 | SDC comparison report |
| `exports/build_historical_trends_report.py` | 1,656 | Historical trends report |
| `analysis/sensitivity_analysis.py` | 1,614 | Sensitivity analysis |
| `analysis/qc_diagnostics.py` | 1,582 | QC diagnostics |
| `analysis/build_experiment_dashboard.py` | 1,549 | Experiment dashboard builder |

---

## 5. Naming Consistency

### Module Names

All Python modules use `snake_case`. Consistent.

### Class Names

All 35 classes use `PascalCase`. Consistent. Notable pattern: dataclasses use `PascalCase` as well (`RunIdentity`, `ComparisonResult`, `ReconciliationResult`, etc.).

### Function Names

All functions use `snake_case`. Private functions consistently prefixed with `_`. No naming violations found.

### One Naming Anomaly

The `_SafeFormatDict` class in `observatory/recipe_registry.py` inherits from `dict[str, Any]` -- the underscore prefix is correct for a private helper, and the name accurately describes its purpose.

---

## 6. Scripts Organization

### Current Structure

```
scripts/                                  (8 root-level scripts, should be grouped)
+-- analysis/                             (16 files, 24,000+ lines)
+-- backtesting/                          (2 files)
+-- data/                                 (17 files)
+-- data_processing/                      (4 files) -- overlaps with data/
+-- db/                                   (4 files, 2 SQL)
+-- exports/                              (14 files)
+-- hooks/
+-- intelligence/                         (2 files + README)
+-- maintenance/                          (2 files)
+-- migrations/                           (1 SQL file)
+-- pipeline/                             (8 files)
+-- projections/                          (3 files)
+-- reviews/                              (2 files)
+-- setup/                                (3 files)
+-- testing/
+-- validation/                           (1 file)
+-- windows/
```

Total: **62,743 lines** across Python scripts (up from 20,531 in Feb, primarily from observatory CLI and analysis growth).

### Issues

**Root-level scripts (should be grouped):**

| Script | Lines | Suggested Home |
|---|---:|---|
| `check_test_coverage.py` | 161 | `scripts/testing/` |
| `extract_sdc_fertility_rates.py` | 388 | `scripts/data/` |
| `fetch_data.py` | 672 | `scripts/data/` |
| `generate_article_pdf.py` | 330 | `scripts/exports/` |
| `generate_visualizations_and_reports.py` | 304 | `scripts/exports/` |
| `process_nd_migration.py` | 342 | `scripts/data/` or `scripts/data_processing/` |
| `run_integration_test.py` | 621 | `scripts/testing/` or `tests/` |
| `validate_data.py` | 893 | `scripts/validation/` |

**Overlapping directories:**

`scripts/data/` (17 files) and `scripts/data_processing/` (4 files) serve similar purposes. The `data_processing/` scripts are ADR-035 Phase implementations that process PEP data. These could be consolidated under `scripts/data/` with a clear naming convention (e.g., `pep_` prefix).

**Reusable logic trapped in scripts:**

`scripts/pipeline/02_run_projections.py` (2,103 lines) contains ~500 lines of data transformation functions (`_transform_fertility_rates`, `_transform_survival_rates`, `_transform_migration_rates`, `expand_5yr_migration_to_engine_format`, `_build_convergence_rate_dicts`, `_build_survival_rates_by_year`, `aggregate_county_results_to_state`) that bridge `data/process` output to `core` engine input. These are reusable and should live in a `cohort_projections/data/transform/` module.

---

## 7. Duplicate and Similar Code

### `_normalize_fips()` -- 4 Implementations

Four nearly identical FIPS normalization functions exist across the place-projection modules:

| Module | Signature | Handles None? |
|---|---|---|
| `data/process/place_backtest.py:24` | `(value: object, width: int) -> str` | No |
| `data/process/multicounty_allocation.py:73` | `(value: object, width: int) -> str | None` | Yes |
| `data/process/place_shares.py:21` | `(value: str | int | float | None, width: int) -> str | None` | Yes |
| `data/process/place_projection_orchestrator.py:90` | `(value: object, width: int) -> str` | No |

All four share identical core logic: `str(value).strip().removesuffix(".0")` then digit extraction and zero-padding. The only variation is whether they accept/return `None`. This should be a single function in `utils/demographic_utils.py` or a new `data/process/_fips_utils.py`.

**Severity:** WARNING

### Validation Function Duplication

Validation functions exist in two layers with different purposes but identical names:

| Function | `core/` version | `data/process/` version |
|---|---|---|
| `validate_fertility_rates()` | Engine-facing: checks DataFrame shape for projection input (returns `tuple[bool, list]`) | Data-facing: comprehensive plausibility check on processed rates (returns `dict[str, Any]`) |
| `validate_survival_rates()` | Engine-facing | Data-facing |
| `validate_migration_data()` | Engine-facing | Data-facing |

The names collide in the `data/process/__init__.py` namespace (both `core/` and `data/process/` export them). However, since consumers import from specific modules, this is not a runtime problem -- just a naming collision risk.

**Severity:** INFO (different scope, but confusing for new developers)

### `generate_html_report()` -- 2 Implementations

| Module | Purpose |
|---|---|
| `output/reports.py:370` | General projection HTML report (population summaries, scenarios) |
| `analysis/evaluation/html_report.py:52` | Evaluation framework HTML report (scorecards, benchmarks) |

These serve fundamentally different purposes. The name collision is benign since they live in different packages, but could be clarified with more specific names.

**Severity:** INFO

### Race Mapping Aliases

The pattern of re-exporting race mapping constants via aliases persists:

- `SEER_MORTALITY_RACE_MAP = SEER_RACE_MAP` in `survival_rates.py`
- `SEER_RACE_ETHNICITY_MAP` in `fertility_rates.py` (alias for `SEER_RACE_MAP`)
- `MIGRATION_RACE_MAP` re-exported through `migration_rates.py.__all__`
- `RACE_ETHNICITY_MAP` in `base_population.py` (standalone mapping, not using `config/`)

**Severity:** INFO (backward compatibility, low risk)

---

## 8. Changes Since February 2026 Audit

### Fixed

1. **`version.py` modernized** -- Now reads version from `pyproject.toml` or installed package metadata. No longer orphaned.
2. **`output/__init__.py` stale `__version__` removed** -- Single source of truth restored.
3. **`demographic_utils.py` deduplication** -- `calculate_median_age()` and `calculate_dependency_ratio()` are now imported from the canonical location in `core/cohort_component.py` and `output/reports.py` (3 production callers total). The private-copy pattern identified in the Feb audit is resolved.

### Not Fixed

1. **`data/process/example_usage.py`** -- Still present, still dead code.
2. **`data/fetch/vital_stats.py`** -- Still a stub, still zero imports.
3. **`migration_rates.py` sizing** -- Still 1,963 lines, no split.
4. **`base_population_loader.py` sizing** -- Still 1,471 lines, no GQ extraction.
5. **Mixed import styles** -- Relative in core/output, absolute elsewhere.
6. **Root-level scripts** -- Still ungrouped.
7. **`data/popest_shared.py` placement** -- Still at `data/` top level, only used by `scripts/data/` CLI scripts.
8. **`data/process/__init__.py` inconsistency** -- Still selectively exports some modules but not others (now worse at 262 lines / ~60 symbols).

### New Since February

1. **`analysis/` subpackage created** -- 21,294 lines across 34 files. This is now the dominant subpackage.
2. **Observatory dashboard** -- 9,004 lines across 11 dashboard tab/widget modules. Several over 1,000 lines.
3. **`_normalize_fips()` quadruplication** -- Four copies appeared with the place-projection modules (PP-003/PP-005).
4. **Place projection modules added** -- `place_backtest.py`, `place_housing_unit_projection.py`, `place_projection_orchestrator.py` (1,786 lines), `place_share_trending.py`, `place_shares.py`, `multicounty_allocation.py`, `rolling_origin_backtest.py`.
5. **New analysis modules** -- `benchmark_contract.py`, `evaluation_policy.py`, `experiment_log.py`.

---

## 9. Recommendations

### Priority 1: Remove Dead Code (LOW EFFORT, HIGH SIGNAL)

1. **Delete `data/process/example_usage.py`** (286 lines) and remove `load_example_usage_module()` from `data/process/__init__.py`. If needed for documentation, move to an `examples/` directory.
2. **Delete `data/fetch/vital_stats.py`** (133 lines) or mark it explicitly as a scaffold with a module-level comment and a skip in any future dead-code scans.

### Priority 2: Extract `_normalize_fips()` (LOW EFFORT, MEDIUM VALUE)

Create a single `_normalize_fips()` in a shared location (e.g., `data/process/_fips_utils.py` or `utils/demographic_utils.py`). Provide two variants: one that returns `str` (never None), one that returns `str | None`. Update the 4 callsites.

### Priority 3: Split Oversized Modules (MEDIUM EFFORT, HIGH VALUE)

Priority targets unchanged from Feb audit:

1. **`data/process/migration_rates.py`** (1,963 lines) -- Extract `migration_loading.py`, `migration_distribution.py`, `migration_scenarios.py`.
2. **`data/load/base_population_loader.py`** (1,471 lines) -- Extract `gq_population.py` (~200 lines).
3. **`data/process/residual_migration.py`** (1,316 lines) -- Extract adjustment helpers from the 305-line `run_residual_migration_pipeline()`.
4. **`observatory/dashboard/tab_command_center.py`** (1,464 lines) -- Consider extracting search UI and queue management into separate modules.

### Priority 4: Clean Up `data/process/__init__.py` (MEDIUM EFFORT, MEDIUM VALUE)

At 262 lines and ~60 re-exported symbols, this `__init__` is doing too much. Options:
- (a) Export everything consistently (add `convergence_interpolation` and `residual_migration` exports), or
- (b) Export nothing and rely on direct module imports (consumers already do this for convergence/residual).

Option (b) is preferable for a package of this size.

### Priority 5: Group Root-Level Scripts (LOW EFFORT, LOW VALUE)

Move the 8 root-level scripts in `scripts/` to their appropriate subdirectories per the table in Section 6. Consolidate `scripts/data_processing/` into `scripts/data/`.

### Priority 6: Create `data/transform/` Layer (MEDIUM EFFORT, HIGH VALUE)

Extract the ~500 lines of rate transformation functions from `scripts/pipeline/02_run_projections.py` into a new `cohort_projections/data/transform/` subpackage. This removes reusable logic from a script and makes it testable.

### Priority 7: Monitor Observatory Growth

The observatory went from 0 to 13,945 lines in ~2 weeks. It is well-structured internally (clean module boundaries, consistent imports, good test coverage with 11 test files). However:
- 5 of 23 modules exceed 1,000 lines
- `data_manager.py` holds two large classes (767-line class boundary)
- Consider whether `dashboard/` should be a standalone installable (it requires Panel, which is a heavy dependency)

---

## 10. Overall Metrics

| Metric | Feb 2026 | Mar 2026 | Assessment |
|---|---|---|---|
| Package line count | 17,759 | 43,994 | +148% (observatory growth) |
| Package Python files | 35 | 76 | +117% |
| Scripts line count | 20,531 | 62,743 | +206% |
| Test line count | 24,976 | 44,943 | +80% |
| Circular imports | 0 | 0 | Excellent |
| Star imports | 0 | 0 | Excellent |
| Dead/orphaned modules | 3 | 2 | Improved (version.py fixed) |
| Modules >1,000 lines | 3 | 11 | Worsened (observatory + place modules) |
| Modules >500 lines | 13 | 28+ | Worsened |
| Duplicate implementations | 3 | 5 | Worsened (_normalize_fips x4 new) |
| TODO/FIXME in package | 0 | 0 | Clean |
| TODO/FIXME in scripts | 3 | 3 | Stable |
| Import style consistency | Mixed | Mixed | No change |

**Overall assessment:** The package has maintained its clean architectural layering and zero-circular-dependency record while growing 2.5x in size. The new `analysis/` subpackage is well-organized internally with proper test coverage. The main debt items from February (oversized migration modules, dead code, ungrouped scripts) remain unaddressed, and a new debt (_normalize_fips quadruplication) has appeared. The highest-leverage cleanups are dead code removal (Priority 1) and FIPS utility extraction (Priority 2), both requiring minimal effort.
