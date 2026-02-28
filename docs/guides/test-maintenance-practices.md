# Test Maintenance Practices

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-02-28 16:15 CST |
| **Based On** | Analysis of codebase and coverage reports as of 2026-02-28 |
| **Complements** | `test-suite-reference.md` (what each test does), `testing-workflow.md` (how to run tests) |

---

## Current State Assessment

### Coverage by Module

Coverage as measured by `pytest --cov` with the `htmlcov/` line-by-line report. Modules are listed from lowest to highest coverage.

| Coverage | Module | Notes |
|---------:|--------|-------|
| 0.0% | `data/fetch/vital_stats.py` | No test file exists |
| 0.0% | `data/process/example_usage.py` | Demo script, not production code |
| 18.5% | `utils/bigquery_client.py` | Heavy mocking bypasses real code paths |
| 25.0% | `utils/reproducibility.py` | Heavy mocking |
| 36.4% | `data/load/base_population_loader.py` | GQ functions from ADR-055 completely untested |
| 38.2% | `config/race_mappings.py` | |
| 39.1% | `data/load/census_age_sex_population.py` | No standalone test file |
| 44.3% | `data/process/convergence_interpolation.py` | Pipeline orchestrator untested |
| 52.5% | `data/process/base_population.py` | |
| 54.0% | `data/process/residual_migration.py` | `subtract_gq_from_populations` untested |
| 55.3% | `utils/demographic_utils.py` | |
| 59.3% | `geographic/multi_geography.py` | |
| 74.5%-77.8% | survival_rates, fertility_rates, migration_rates, geography_loader | Solid coverage |
| 87.6%-98.9% | Core engine modules | cohort_component, visualizations, writers, pep_regime_analysis, migration, mortality_improvement, reports, mortality, fertility |

### Structural Findings

**Silent test skips.** Approximately 195 tests are gated on `IMPORTS_AVAILABLE` or data file existence checks and can silently skip. When these tests skip, the modules they guard have no effective coverage. Five additional tests are permanently skipped due to a PyMC dependency, and four tests in `test_bayesian_var.py` are skipped for upstream bugs.

**Pre-commit hook blind spot.** The pre-commit hook only triggers on changes to `cohort_projections/**/*.py`. If only test files change, the hook does not run. This means broken tests can be committed without detection.

**No CI/CD pipeline.** There is no automated test run on push or pull request. All test enforcement depends on local pre-commit hooks and developer discipline.

**ADR-011 staleness.** ADR-011 (testing strategy) is materially stale and no longer reflects current test infrastructure or coverage expectations. ADR-056 supersedes it.

### Untested Critical Functions

These functions have zero test coverage and directly affect projection accuracy:

**GQ separation (ADR-055):**
- `subtract_gq_from_populations` -- core Phase 2 function that removes GQ population from migration rate denominators
- `_separate_gq_from_base_population` -- splits base population into household and group quarters components
- `get_county_gq_population` -- loads GQ population data for a county
- `_expand_gq_to_single_year_ages` -- converts broad Census age groups to single-year ages
- `_distribute_gq_across_races` -- distributes GQ population across race categories

**Pipeline orchestrators:**
- `run_residual_migration_pipeline` -- wires together all residual migration components
- `run_convergence_pipeline` -- wires together convergence interpolation components

**Base population loaders:**
- `load_base_population_for_county` -- entry point for county-level projections
- `load_base_population_for_all_counties` -- entry point for multi-county runs
- `load_base_population_for_state` -- entry point for state-level projections

---

## Coverage Monitoring Practices

### Reading the `--cov` output

The default pytest configuration in `pyproject.toml` includes `--cov=cohort_projections --cov-report=term-missing`. Every test run prints a coverage table to the terminal with three key columns:

- **Stmts**: Total executable statements in the module
- **Miss**: Statements not executed by any test
- **Cover**: Percentage of statements executed

The `Missing` column lists specific line ranges that were not executed. These are the lines where bugs can hide undetected.

### Using `htmlcov/index.html`

After running `pytest`, open `htmlcov/index.html` in a browser for line-by-line analysis. Green lines were executed during tests. Red lines were not. This is the most effective way to identify exactly which code paths lack coverage -- particularly useful when a module shows 50% coverage and you need to determine whether the untested half is error handling (lower risk) or core logic (higher risk).

### What to watch for

- **Modules dropping below previous coverage levels.** If `residual_migration.py` was at 54% and drops to 48% after a change, new code was added without tests.
- **New modules appearing at 0%.** Any new file in `cohort_projections/` should have corresponding tests before it is committed.
- **Coverage increases that come only from mocking.** A module jumping from 20% to 80% through heavy mocking may not reflect real testing improvement.

### Recommended approach

Check coverage after adding new production code. Before committing, verify that the new code has tests and that overall coverage on affected modules has not decreased. This does not require a formal coverage threshold -- just the discipline of not shipping untested code.

---

## When to Write New Tests

Prioritized by risk to projection accuracy:

### Must test

Any new function in `cohort_projections/core/` or `cohort_projections/data/process/`. These modules directly affect projection numbers. An undetected bug in cohort aging, migration rate computation, or convergence interpolation produces plausible-looking but wrong results that may not be caught until a publication is challenged.

### Should test

New data loaders, new export formats, and new configuration options. These are one step removed from the projection math but still affect whether the right data enters the engine and whether the right results reach stakeholders.

### Can defer

Scripts in `scripts/`, one-time analysis code, and visualization formatting details. These are either run interactively (where errors are visible immediately) or affect presentation rather than substance.

### Special case: ADR implementations

When implementing an ADR, the ADR's new functions should have tests before the ADR is marked Accepted. The ADR represents an architectural commitment -- if its implementation is untested, the commitment is to code that may not work as specified. The current gap in ADR-055 GQ functions is an example of this rule being violated.

---

## When to Update Existing Tests

### Data vintage change

When the project moves to a new data vintage (e.g., PEP Vintage 2024 to Vintage 2025), real-data integration tests will fail because expected values no longer match. The correct response is:

1. Verify the new values are correct by independent inspection of the source data
2. Update the expected values in the tests
3. Document the vintage change in the commit message

Do not simply delete failing assertions. The test was correct for the old vintage -- it should be correct for the new one.

### ADR implementation

Add tests for new functions introduced by the ADR. Update existing tests that reference behavior the ADR changes. For example, ADR-055 Phase 2 changed how migration rate denominators are computed -- any test that hardcodes expected migration rates for a county with significant GQ population needs to be updated.

### Schema change

When a column is renamed or a new required column is added, update test fixtures and assertions to match. Search for the old column name across all test files to ensure no references are missed.

### Bug fix

Before fixing a bug, write a regression test that reproduces it. The test should fail before the fix and pass after. This ensures the bug stays fixed and documents the failure mode for future developers.

---

## Test Hygiene Practices

### Avoid over-mocking

If a test mocks so much that less than 30% of the production code actually runs, the test is not testing the real code. It is testing the mock. The BigQuery client tests (18.5% coverage) and reproducibility tests (25.0% coverage) illustrate this problem -- the mocks are so extensive that entire code paths are never exercised.

Prefer integration-style tests with real (small) data over heavily-mocked unit tests. A test that creates a 3-county, 5-age-group DataFrame and runs it through the actual function is more valuable than a test that mocks every dependency and checks that the function called the right methods in the right order.

### Prefer invariant assertions

Tests that check relationships are more durable than tests that check specific numbers:

- `restricted <= baseline <= high_growth` (scenario ordering)
- `0 < survival_rate <= 1` (physical constraint)
- Fertility rates zero outside ages 15-49 (demographic definition)
- County sums equal state total (ADR-054 invariant)
- Population is non-negative (physical constraint)

These assertions survive data vintage changes, parameter tuning, and methodology updates. Tests that assert `population == 882146` break every time the inputs change.

### Handle IMPORTS_AVAILABLE gates

When a test is gated on `IMPORTS_AVAILABLE`, ensure the import actually works in the development environment. If 195 tests are silently skipping, those modules have no effective coverage. Periodically run the full suite and check the "skipped" count in pytest output. If the count increases, investigate why.

### Pin random seeds

All synthetic test fixtures should use `np.random.seed()` or `rng = np.random.default_rng(42)` for reproducibility. A test that passes 99% of the time and fails 1% due to random data generation is worse than a test that always fails -- it erodes trust in the test suite and trains developers to re-run rather than investigate.

### One assertion per concept

Multiple assertions in one test function are fine, but they should all relate to one concept. Do not test migration rate computation AND fertility rate computation in the same test function. When a multi-concept test fails, the failure message does not clearly indicate which concept broke, and fixing one may mask a failure in the other.

---

## Periodic Review Cadence

Aligned with the existing PP-002 non-regression cadence.

### At each publication milestone

Run full `pytest` and review the coverage report. Note any modules that dropped below their previous coverage level. This is the minimum enforcement mechanism -- if coverage is declining at publication time, untested code is accumulating.

### Quarterly

Review the skip list. Are tests still skipping for valid reasons? Have dependencies been installed that would un-skip tests? The PyMC dependency is a known long-term skip, but `IMPORTS_AVAILABLE` gates may resolve themselves when the environment is updated. Check whether the skip count has grown.

### After each ADR implementation

Verify the ADR's new functions appear in the coverage report. If they show 0%, add tests before marking the ADR complete. This is a direct application of the "special case" rule from the decision framework above.

### Lightweight checklist

Five items to verify at each review:

1. `pytest` passes with no new failures
2. Coverage has not dropped on any module that was previously tested
3. No new `IMPORTS_AVAILABLE` gates were added without verification that the import works
4. Real-data integration tests still pass (data files exist and match expected schemas)
5. Pre-existing skip/timeout issues have not grown (currently: 5 PyMC skips, 4 upstream bug skips, 1 known timeout)

---

## Priority Coverage Gaps

Ranked by how much damage an undetected bug would cause to projection results.

### 1. GQ Separation Functions (ADR-055)

**Risk: Critical.** These are recent additions that change migration rate computation for every county. `subtract_gq_from_populations` removes group quarters population from the denominators used to compute residual migration rates. If this function silently fails (returns the original population unchanged), migration rates for counties with large GQ populations (Cass County with NDSU dorms, Grand Forks County with UND) are computed on inflated denominators, producing understated rates. The Phase 2 delta from Phase 1 was -1.5pp at the state level -- an error of that magnitude would be significant. Currently at zero test coverage.

### 2. Pipeline Orchestrators

**Risk: High.** `run_residual_migration_pipeline` and `run_convergence_pipeline` wire together individually-tested components. If the wiring is wrong -- arguments passed in the wrong order, intermediate results not propagated, error handling swallowing exceptions -- the individually-correct components produce wrong results when composed. This is the classic integration gap: unit tests pass, but the system does not work.

### 3. Base Population Loaders

**Risk: High.** `load_base_population_for_county`, `load_base_population_for_all_counties`, and `load_base_population_for_state` are the entry points for the projection pipeline. If loading is wrong -- wrong file path, wrong column selection, wrong filtering -- everything downstream is wrong. These functions also contain the GQ separation logic from ADR-055 Phase 1, which is itself untested.

### 4. `census_age_sex_population.py`

**Risk: Medium.** At 39.1% coverage with no standalone test file, this module is only exercised by data-dependent tests that skip when data files are not present. The module handles Census age/sex population data loading, which feeds into base population construction. An error here would affect starting populations but would likely be caught by downstream plausibility checks (negative populations, implausible age distributions).

### 5. `vital_stats.py`

**Risk: Lower.** At 0% coverage, this module has no tests at all. However, it is a data fetcher -- its errors would manifest as missing data or load failures rather than silent numerical errors. A fetch that returns no data causes a visible crash; a fetch that returns wrong data is harder to detect but less likely than a computation error in a processing module.

---

## Relationship to Other Governance

This guide is part of a set of interconnected documents:

- **`test-suite-reference.md`** documents what each test does and why it matters. This guide documents how to keep those tests healthy over time.
- **`testing-workflow.md`** documents how to run tests. This guide documents when to write, update, and review them.
- **ADR-056** supersedes ADR-011 and establishes the strategic direction for testing. This guide provides the tactical practices that implement that strategy.
- **AGENTS.md** requires "new functionality must have tests." This guide defines what that requirement means in practice -- which modules are must-test, which are can-defer, and what constitutes adequate coverage.
- **PP-002 non-regression cadence** is the enforcement mechanism. The periodic review checklist in this guide is designed to run alongside PP-002 milestones.
