# Test Suite Reference

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-02-28 15:42 CST |
| **Total Tests** | 1,263 |
| **Full Suite Runtime** | ~5 minutes |
| **Pytest Config** | `pyproject.toml [tool.pytest.ini_options]` |
| **Pre-commit Hook** | `.pre-commit-config.yaml` (pytest-check) |

---

## Quick Reference: Suites by Runtime

| Suite | Tests | Time | Domain |
|------:|------:|-----:|--------|
| `tests/test_data/` | 416 | **2m 44s** | Data loading, rate computation, pipeline anchoring |
| `tests/test_output/` | 103 | **57s** | Reports, Excel/CSV writers, visualizations |
| `tests/test_integration/` | 137 | **55s** | End-to-end pipeline, cross-module contracts |
| `tests/test_core/` | 173 | **21s** | Projection engine: cohort-component, migration, mortality, fertility |
| `tests/test_statistical/` | 171 | **14s** | Regime models, covariate anchoring, Bayesian VAR, placebo tests |
| `tests/unit/` | 24 | **9s** | Journal article QA, SDC loader, archive fingerprints |
| `tests/test_geographic/` | 74 | **7s** | Geography loader, multi-level aggregation |
| `tests/test_config/` | 27 | **5s** | Race category mappings |
| `tests/test_tools/` | 10 | **5s** | Citation audit tooling |
| `tests/test_utils/` | 123 | **3s** | Config loader, demographic math, BigQuery, reproducibility |

---

## When Each Suite Runs

### Pre-commit hook (on `git commit`)

Defined in `.pre-commit-config.yaml` line 65:

```
pytest tests/ -x -q --ignore=tests/test_integration/ -m "not slow"
```

- **Excludes** `test_integration/` entirely
- **Excludes** anything marked `@pytest.mark.slow`
- **Only triggers** when files in `cohort_projections/**/*.py` change
- Still runs ~1,100 tests including the expensive `test_data/` suite
- **Typical time: 3+ minutes**

### Full test run (`pytest`)

Default addopts in `pyproject.toml`:

```
-v --tb=short --cov=cohort_projections --cov-report=term-missing --timeout=300
```

Runs all 1,263 tests with coverage. **Typical time: ~5 minutes.**

### Faster alternatives for development

| Command | Tests | Time | Use Case |
|---------|------:|-----:|----------|
| `pytest tests/unit/ tests/test_utils/ tests/test_config/` | ~174 | **~17s** | Quick sanity check |
| `pytest -x -q --ignore=tests/test_data/ --ignore=tests/test_integration/` | ~575 | **~1 min** | Skip the two slowest suites |
| `pytest -k "not test_residual_computation_single_period"` | ~1,262 | **~5 min** | Full run minus the known timeout offender |

---

## Suite-by-Suite Detail

### `tests/test_data/` -- Data Pipeline Integrity (416 tests, 2m 44s)

The largest and slowest suite. Guards the entire data ingestion and rate computation pipeline -- the inputs that feed every projection. If these tests fail, the engine may still run but will produce silently wrong results.

#### `test_residual_migration.py`

**What it validates:** Residual migration rate computation -- Rogers-Castro age distribution, period dampening, college-age adjustments, PEP recalibration, and GQ correction (ADR-055 Phase 2).

**Why it matters:** Migration rates must peak at ages 20-30 (the well-established Rogers-Castro pattern). Without this test, rates could flatten across ages, eliminating the young-adult migration peak that drives Cass and Grand Forks County growth. PEP recalibration anchors synthetic rates to actual Census components -- if that anchoring fails, rates diverge from observed reality with no warning.

**Known issue:** `test_residual_computation_single_period` has a pre-existing timeout (>60s) reading a large Excel file. Exclude with `-k "not test_residual_computation_single_period"`.

#### `test_base_population.py`

**What it validates:** Population harmonization, cohort matrix creation, race standardization, validation of negative/zero/implausible populations.

**Why it matters:** The base population is the starting point for every projection. If race codes are silently dropped (e.g., unmapped "UNKNOWN_RACE"), 5-10% of the population vanishes. If sex values aren't standardized ("male" vs "Male"), DataFrame joins fail silently. These are the kinds of bugs that produce a plausible-looking but wrong result.

#### `test_convergence_interpolation.py`

**What it validates:** The 5-10-5 convergence schedule that interpolates migration rates across Phase 1 recent/medium/long windows. Also tests rate capping (ADR-043) and high-scenario lifting (ADR-046).

**Why it matters:** The convergence schedule controls how projections transition from recent trends to long-run averages. If year-1 weights aren't 80% recent + 20% medium, the first five years of every projection are wrong. Rate caps prevent extreme values from dominating; high-scenario lifting ensures "high" is actually above baseline.

#### `test_bebr_averaging.py`

**What it validates:** BEBR multi-period averaging (ADR-036) -- period calculation, trimmed baseline (drop min/max), low/baseline/high scenario generation, county dampening.

**Why it matters:** This is the scenario generation engine. Without these tests, the three scenarios could collapse to identical outputs, oil county rates might not be dampened, or the trimmed mean could accidentally include extremes. Scenario ordering (low < baseline < high) is a core invariant.

#### `test_migration_rates.py`

**What it validates:** Net migration calculation, age/sex/race distribution via Rogers-Castro, IRS flow loading, international migration, rate table creation.

**Why it matters:** Migration is the dominant driver of ND population change. If the age distribution flattens, young adults don't get their disproportionate share. If race distribution doesn't match population proportions, minority migration signals disappear. Extreme values (50,000 per county) must be flagged.

#### `test_fertility_rates.py`

**What it validates:** SEER fertility data loading, race harmonization, ASFR averaging, reproductive age enforcement (15-49 only), TFR plausibility.

**Why it matters:** Fertility rates must be zero outside ages 15-49. If unmapped SEER codes are silently dropped, birth counts fall. TFR outside 0.5-10.0 indicates a data or calculation error that would compound over 20 years of projection.

#### `test_mortality_improvement.py`

**What it validates:** Census Bureau NP2023 mortality improvement pipeline -- survival ratios within (0, 1], ND baseline expansion from 18 age groups to 101 single-year ages, time-varying improvements (2025 survival < 2045 survival).

**Why it matters:** Survival rates above 1.0 or below 0 are physically impossible. If the 18-to-101 age expansion fails, ages 85+ all get the same grouped rate, distorting elderly population projections. The sex mortality differential (males higher than females) is a fundamental demographic fact -- inverting it produces nonsense.

#### `test_survival_rates.py`

**What it validates:** Life table processing, survival rate calculation from lx/qx, mortality improvement application, life expectancy bounds.

**Why it matters:** Life expectancy below 50 or above 100 flags a calculation error. Division by zero in the lx method (when lx[x+1]=0) would crash the engine. Age 90+ requires special open-interval handling.

#### `test_pep_migration_rates.py`

**What it validates:** PEP county-level net migration distributed to age/sex/race-specific rates using BEBR scenarios and Rogers-Castro. Validates output dimensions (1,092 = 91 ages x 2 sexes x 6 races per county).

**Why it matters:** Wrong output dimensions mean the projection engine either crashes or silently drops cohorts. If the Rogers-Castro age pattern isn't applied, age-specific rates are flat, eliminating the young-adult migration peak.

#### `test_pep_regime_analysis.py`

**What it validates:** PEP regime classification (oil boom 2011-2015, recovery 2022+), regime-specific averages, county type classification (oil/metro/rural).

**Why it matters:** Oil county identification drives ADR-051 dampening. If regime boundaries are wrong (2011-2015 vs 2010-2016), the wrong years get dampened. Weighted averages must sum to 1.0.

#### `test_census_api.py`

**What it validates:** Census API client initialization, retry logic, caching (parquet + metadata), PEP/ACS data fetching.

**Why it matters:** Without retry logic, transient network errors cause hard failures. Without caching, repeated API calls waste quota and slow development. Vintage metadata preserves data lineage.

#### `test_county_race_distributions.py`

**What it validates:** County-specific age-sex-race distributions (ADR-047), proportion sums, reservation county distinctness, fallback to state-level distributions.

**Why it matters:** Reservation counties have fundamentally different race distributions than urban counties. If the fallback doesn't trigger on missing data, some counties get no race detail at all.

---

### `tests/test_core/` -- Projection Engine (173 tests, 21s)

Guards the mathematical core: the cohort-component method that advances population forward in time.

#### `test_cohort_component.py`

**What it validates:** CohortComponentProjection class -- initialization, single/multi-year projection, scenario handling (baseline vs. high_growth), summary statistics (median age, dependency ratio), CSV/Parquet export.

**Why it matters:** This is the engine. If population isn't properly aged, births aren't calculated, or deaths/migration aren't applied in the correct sequence, every projection year compounds the error. Scenario logic must actually produce different results (high_growth > baseline). Export must preserve all columns and years.

#### `test_migration.py`

**What it validates:** `apply_migration` and scenario types -- constant, percentage adjustments, and critically, the additive_reduction method (ADR-050) that guarantees restricted <= baseline.

**Why it matters:** ADR-050's additive reduction was specifically designed to fix a multiplicative bug where negative-migration counties would *increase* population under the "restricted" scenario (multiplying a negative rate by a factor < 1 makes it less negative = more growth). This test is the primary guard against that regression. Scenario ordering (zero < restricted < baseline < high_growth) is a core invariant verified here.

#### `test_mortality.py`

**What it validates:** Survival application, cohort aging (population at age N moves to age N+1), open-age group handling (90+ accumulation), mortality improvement compounding, infant survival plausibility.

**Why it matters:** The aging mechanism is the backbone of cohort-component projection. If age 0 doesn't advance to age 1, the entire age structure collapses. The 90+ open interval must accumulate rather than discard elderly population. Double-applying mortality improvement (once in preprocessing, once in the engine) would kill the population too fast.

#### `test_fertility.py`

**What it validates:** Birth calculation (female population x fertility rates), sex ratio at birth (51% male default), scenario multipliers, age restriction to 15-49.

**Why it matters:** A wrong sex ratio (50/50 instead of 51/49) compounds over 20 years, distorting the sex structure. Fertility applied outside ages 15-49 produces phantom births. Scenario multipliers must compound correctly over long horizons.

#### `test_time_varying_engine.py`

**What it validates:** Backward compatibility with constant-rate projections, year-specific rate lookup (year_offset for migration, calendar year for survival), fallback on missing years, format bridge from 5-year age groups to single-year engine format.

**Why it matters:** The time-varying engine was added after the constant-rate engine. If backward compatibility breaks, all existing projection configurations produce different results. Off-by-one errors in year lookup (2025 using 2026 rates) are silent and devastating. The format bridge must expand 5-year groups to all 91 single-year ages x 6 races = 1,092 rows.

---

### `tests/test_integration/` -- Cross-Module Contracts (137 tests, 55s)

Validates that independently-correct modules work together. Unit tests can all pass while the integrated system fails.

#### `test_end_to_end.py`

**What it validates:** Real projection engine execution using actual processed data files (Cass County, FIPS 38017). Tests base population construction, rate preparation, 5-year projection completeness, and demographic plausibility.

**Why it matters:** This is the only test that runs the full pipeline from data files to projection output. Every other test uses synthetic data. If data file formats change (column rename, schema drift), only this test catches it.

#### `test_pep_pipeline.py`

**What it validates:** PEP_components migration method with per-county parquet splits, dictionary vs. DataFrame handling for multi-county scenarios, geographic projection runner.

**Why it matters:** The PEP pipeline stores rates as a dictionary of DataFrames keyed by FIPS code. If type mismatches corrupt scenario adjustments (dict vs. DataFrame), some counties silently get zero migration. The format bridge for backward compatibility with the legacy IRS path is validated here.

#### `test_census_method_validation.py`

**What it validates:** Phase 2 convergence schedule correctness, Phase 3 mortality improvement bounds, Phase 1 no-negatives constraint over 20-year projections, format bridge expansion, NaN leakage detection.

**Why it matters:** These are cross-phase contracts. A convergence schedule that works in isolation could still produce NaN when combined with survival rates from a different phase. The no-negatives constraint over 20 years catches slow accumulation errors that don't appear in 5-year tests.

#### `test_adr021_modules.py`

**What it validates:** ADR-021 regime framework -- regime boundaries, policy event dataclasses, regime-aware indicators, status durability, two-component estimand decomposition.

**Why it matters:** The regime framework provides context for interpreting migration data across methodological vintages (pre-2010, 2010-2020, post-2020). If regime boundaries overlap or have gaps, rates from one vintage contaminate another. Frozen dataclass immutability prevents accidental mutation of policy events.

---

### `tests/test_output/` -- Deliverable Quality (103 tests, 57s)

Guards the reports, exports, and visualizations that stakeholders actually see.

#### `test_reports.py`

**What it validates:** Summary statistics (by-year, age structure, growth analysis, diversity metrics), scenario comparison tables, HTML/text/Markdown report generation.

**Why it matters:** These reports are the primary deliverables. Missing growth rates or dependency ratios in a by-year summary make the report incomplete. If diversity metrics include spurious race categories, the report is misleading. HTML reports without proper structure won't render in browsers.

#### `test_writers.py`

**What it validates:** CSV export (long/wide format, compression), Excel export (6 required sheets: Summary, By Age, By Sex, By Race, Detail, Metadata), Parquet/JSON export, age/sex/race filtering, metadata file generation.

**Why it matters:** Stakeholders consume projections in Excel and CSV. If wide-format CSV doesn't pivot correctly, columns are mislabeled. If Excel sheets are missing or misnamed, the workbook is unusable. Metadata files provide the audit trail that makes results reproducible.

#### `test_visualizations.py`

**What it validates:** Population pyramids (male/female inversion), trend charts (by sex, age, race), growth rate charts, scenario comparison charts, batch generation with output format control (PNG, SVG).

**Why it matters:** Visualizations are the primary communication tool for decision-makers. A population pyramid with male/female on the same side is unreadable. Growth rates with wrong period denominators produce misleading trends.

---

### `tests/test_statistical/` -- Methodology Validation (171 tests, 14s)

Guards the statistical methods used in research analysis alongside the projections. Five tests are skipped (dependency-gated, likely PyMC/Bayesian packages).

#### `test_regime_aware.py`

**What it validates:** Vintage dummy variable creation (2000s reference, 2010s, 2020s dummies), piecewise trend estimation, COVID intervention modeling, robust standard errors.

**Why it matters:** Migration data spans three methodological vintages. If vintage boundaries are misaligned, regression coefficients capture vintage artifacts rather than real trends. The reference category (2000s omitted from regression) must be correctly set or all coefficients are biased.

#### `test_covariate_anchor.py`

**What it validates:** Lag construction for exogenous variables (refugee arrivals, LPR flows), Last-Observation-Carried-Forward extension, local level regression, confidence intervals.

**Why it matters:** Lag indices must align correctly -- year 2020 must use 2019 covariate values, not 2020. Off-by-one lag errors bias near-term forecasts. LOCF extension provides the forecast-period covariate values; if it doesn't repeat correctly, forecasts use stale or missing data.

#### `test_bayesian_var.py`

**What it validates:** Minnesota prior construction, Bayesian VAR estimation (conjugate + PyMC fallback), panel VAR for multi-state analysis, model comparison metrics.

**Why it matters:** With only 25 years of ND migration data, Bayesian methods prevent overfitting. The Minnesota prior encodes the random-walk assumption -- if prior dimensions don't match data structure, estimation fails or produces unbounded coefficients.

#### `test_multistate_placebo.py`

**What it validates:** State boom category classification (Bakken, Permian, Other Shale, Mature Oil, Non-Oil), shift calculations, oil vs. non-oil hypothesis tests, ND percentile ranking.

**Why it matters:** Placebo tests validate that ND's migration patterns are genuinely unusual, not just noise. If state classification is wrong (oil state counted as non-oil), the comparison group is contaminated. ND's FIPS code must be correctly matched or it's excluded from its own ranking.

---

### `tests/unit/` -- Research Artifact QA (24 tests, 9s)

Guards the integrity of journal article outputs, data archives, and research module logic.

#### `test_sdc_data_loader.py`

**What it validates:** SDC data loader correctly switches between file-based and database-based sources based on environment variables.

**Why it matters:** Silent source switching could cause analyses to use stale file data when the database has been updated, or vice versa.

#### `test_journal_article_versioning.py`

**What it validates:** Versioning system for journal article artifacts -- build names, file hashing, version index updates.

**Why it matters:** Published research must have a reproducible audit trail. If version metadata is corrupted, there's no way to match a published PDF to its source code and data.

#### `test_popest_archive_and_fingerprints.py`

**What it validates:** Data integrity for archived Census PEP files via schema fingerprinting and MD5 checksums.

**Why it matters:** Census data is the ground truth. If archived files are modified, corrupted, or replaced without detection, all downstream analyses built on that data are invalidated.

#### `test_duration_figure_table_consistency.py`

**What it validates:** Publication figures (Kaplan-Meier curves, log-rank tests) match the LaTeX tables in the journal article.

**Why it matters:** If figures and tables fall out of sync, the published article contains internally contradictory results -- a serious credibility problem for peer review.

#### `test_module_7_causal_inference.py`

**What it validates:** Travel-ban causal inference logic -- filtering pseudo-nationalities, event-study year filtering.

**Why it matters:** Pseudo-nationality codes ("Total", "Fy Refugee Admissions") contaminating the analysis would bias causal estimates.

#### `test_journal_article_derived_stats.py`

**What it validates:** PEP vintage aggregation for LaTeX macro generation -- shares, natural increase, domestic migration statistics.

**Why it matters:** Context statistics cited in the published article (e.g., "ND's share of US migration") must be computed from the correct vintage and aggregation level.

#### `test_build_dhs_lpr_panel_variants.py`

**What it validates:** DHS LPR panel construction -- state filtering (exclude PR, US aggregates), panel balancing, ND share calculations.

**Why it matters:** Invalid states leaking into the panel bias all estimates. Unbalanced panels produce biased regression results.

#### `test_module_8_duration_analysis.py`

**What it validates:** Duration analysis preprocessing -- dropping states with incomplete post-2020 data, identifying immigration waves by threshold and gap tolerance.

**Why it matters:** Incomplete states bias survival estimates. Incorrect wave definitions (non-consecutive years) invalidate the duration analysis entirely.

---

### `tests/test_utils/` -- Infrastructure (123 tests, 3s)

Guards the utility functions that everything else depends on. Fast and foundational.

#### `test_config_loader.py`

**What it validates:** YAML parsing, file-not-found errors, caching, nested key access, default values.

**Why it matters:** Every module reads configuration. If the loader silently returns wrong values, caches stale data, or fails without a clear error on missing files, the failure propagates everywhere.

#### `test_reproducibility.py`

**What it validates:** Execution logging -- database connection, git commit tracking, parameter/input/output manifest logging, success/failure status.

**Why it matters:** Without execution logs, there's no way to audit which code version ran, what parameters were used, or whether a run succeeded. Reproducibility is a core requirement for demographic projections used in policy.

#### `test_bigquery_client.py`

**What it validates:** BigQuery client initialization (credentials, project ID), query execution, dataset management, DataFrame uploads.

**Why it matters:** Queries running against the wrong project, or credentials failing silently, would produce wrong data or data loss. These tests use mocks to validate the interface without requiring actual GCP credentials.

#### `test_demographic_utils.py`

**What it validates:** Core demographic math -- age grouping, sex ratio, dependency ratio, median age, Sprague graduation (5-year to single-year interpolation), growth rates, cohort validation.

**Why it matters:** These are the building blocks. An inverted sex ratio, incorrect dependency calculation, or Sprague interpolation producing step artifacts instead of smooth curves would corrupt every module that uses them. This is the fastest suite and the highest leverage.

---

### `tests/test_config/` -- Data Consistency (27 tests, 5s)

#### `test_race_mappings.py`

**What it validates:** Centralized race/ethnicity mapping consistency -- canonical categories exist, all source mappings (Census, SEER, Migration) resolve to canonical categories, all processing modules import from centralized config.

**Why it matters:** Race categories are used across every data source. If one module uses "White NH" while another uses "WA_NH", joins fail silently, aggregations lose data, and stratified analyses produce wrong results. This test enforces a single source of truth.

---

### `tests/test_geographic/` -- Spatial Hierarchy (74 tests, 7s)

#### `test_geography_loader.py`

**What it validates:** Geographic reference data loading -- county and place lists, FIPS code formatting (zero-padded), place-to-county mappings.

**Why it matters:** Unpadded FIPS codes ("38017" vs "038017") cause join failures across datasets. Incorrect place-to-county mappings aggregate city populations to the wrong county.

#### `test_multi_geography.py`

**What it validates:** Multi-level projection orchestration -- single-geography runs, place-to-county-to-state aggregation, validation that aggregates equal component sums.

**Why it matters:** ADR-054 requires state totals to equal county sums by construction. If aggregation logic has rounding errors or missing geographies, the hierarchy violates this invariant.

---

### `tests/test_tools/` -- Publication Quality (10 tests, 5s)

#### `test_citation_audit.py`

**What it validates:** BibTeX citation audit -- nocite handling, string macro expansion, multiline citation parsing, LaTeX `\input` traversal, bibliography fix suggestions.

**Why it matters:** Missing or uncited references in a published article are a peer-review defect. String macros that don't expand produce broken journal names in the bibliography.

---

## The Bottleneck: Why `test_data/` Takes So Long

The `test_data/` suite accounts for **55% of total runtime** (2m 44s of ~5 minutes). This is because:

1. **Real data loading:** Several tests read actual parquet and Excel files from `data/processed/` and `data/raw/`, including one large Excel file that causes a known timeout.
2. **Computational complexity:** Convergence interpolation, BEBR averaging, and residual migration tests run full rate computation pipelines with realistic county counts.
3. **High test count:** 416 tests covering 12 data processing modules, each with multiple parameterized scenarios.

This suite is intentionally thorough because data errors are the most common and most dangerous failure mode -- a structurally valid but numerically wrong input produces plausible-looking but incorrect projections.

---

## Recommendations for Agent Workflows

1. **After editing `cohort_projections/core/`:** Run `pytest tests/test_core/` (~21s). This validates the engine without waiting for data loading.

2. **After editing `cohort_projections/data/`:** Run `pytest tests/test_data/` (~2m 44s). There's no shortcut -- data processing correctness requires these tests.

3. **Quick smoke test:** Run `pytest tests/test_utils/ tests/test_config/ tests/unit/` (~17s). Validates infrastructure and research artifacts.

4. **Before committing:** Run `pytest -x -q --ignore=tests/test_integration/ -k "not test_residual_computation_single_period"` to approximate what pre-commit does, minus the known timeout.

5. **Full validation:** Run `pytest` (~5 min). Required before any release or major merge.
