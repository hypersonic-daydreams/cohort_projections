# ADR-054: State-County Aggregation Reconciliation

## Status
Accepted

## Date
2026-02-23

## Scope
Resolve the systematic divergence between the sum of 53 county-level projections and the independent state-level projection, which reaches +86,037 persons (10.6%) by 2055 in the baseline scenario.

## Context

### Problem: County Sum Exceeds State Projection by 10.6% at 2055

When the projection pipeline runs with `--all` (state + counties), it produces an independent state-level projection and 53 independent county-level projections. The config specifies `validate_aggregation: true` with a 1% tolerance (`aggregation_tolerance: 0.01`), but this validation is not enforced -- the function at `cohort_projections/geographic/multi_geography.py` lines 618-745 (`validate_aggregation()`) exists but is never called with actual data. The validation stub at `scripts/pipeline/02_run_projections.py` lines 1444-1456 logs "Hierarchical validation not yet implemented - skipping."

The discrepancy at the final projection year (2055) across all three active scenarios:

| Scenario | State Projection | County Sum | Difference | % Difference |
|----------|:----------------:|:----------:|:----------:|:------------:|
| Baseline | 814,934 | 900,971 | +86,037 | +10.6% |
| High Growth | 974,581 | 1,067,814 | +93,233 | +9.6% |
| Restricted Growth | 766,576 | 842,885 | +76,309 | +10.0% |

The discrepancy starts at zero in 2025 (identical base populations of 799,358) and grows monotonically:

| Year | State (Baseline) | County Sum | Diff | % |
|------|:----------------:|:----------:|:----:|:-:|
| 2025 | 799,358 | 799,358 | 0 | 0.0% |
| 2030 | 802,408 | 814,315 | +11,907 | +1.5% |
| 2035 | 815,428 | 836,864 | +21,436 | +2.6% |
| 2040 | 828,081 | 860,485 | +32,404 | +3.9% |
| 2045 | 833,707 | 880,325 | +46,618 | +5.6% |
| 2050 | 825,582 | 891,331 | +65,749 | +8.0% |
| 2055 | 814,934 | 900,971 | +86,037 | +10.6% |

### How the State-Level Migration Rate Is Constructed

When the pipeline encounters a state-level FIPS code (2-digit), it constructs a state migration rate by computing the **population-weighted average** of county convergence rates, using base-year (2025) county populations as weights. This logic is in `scripts/pipeline/02_run_projections.py`, lines 1265-1304 (for constant rates) and lines 1316-1365 (for year-by-year convergence rates).

The formula for each age-group x sex cell is:

```
state_rate(age, sex) = SUM(county_rate_i(age, sex) * county_pop_i) / SUM(county_pop_i)
```

where `county_pop_i` is the **base-year** population of county *i*.

## Root Cause Analysis

The investigation identified **two independent sources** of divergence, both originating from the fundamental architectural choice of running state and county projections independently rather than hierarchically.

### Root Cause 1: Jensen's Inequality / Compound Growth Nonlinearity (Primary, ~85% of divergence)

The state-level projection applies a single population-weighted average migration rate to one evolving state population. County projections each apply county-specific rates to independently evolving county populations. Over time, these diverge due to a mathematical nonlinearity analogous to Jensen's inequality:

1. **Growing counties amplify growth**: Counties with positive migration rates (Cass +29.8%, Williams +57.1%, McKenzie +81.8%) gain population each year. In subsequent years, the same positive rate is applied to a **larger** base, producing more net in-migration. This compounds.

2. **Declining counties dampen decline**: Counties with negative migration rates (Ward -14.6%, Grand Forks -8.7%, Walsh -24.8%) lose population each year. In subsequent years, the same negative rate is applied to a **smaller** base, producing less net out-migration. This also compounds.

3. **State projection uses static weights**: The population-weighted average rate is computed from 2025 populations. By 2055, Cass County grows from 25.2% to ~29.1% of the state total, while Ward drops from 8.5% to ~6.5%. The state rate does not reflect this population redistribution.

The net effect is that the county sum's effective state-level migration rate becomes **less negative** over time than the state projection's fixed-weight rate. This is a purely mathematical consequence of applying heterogeneous rates to independently evolving populations.

**Verification**: At year 1 (2026), the state migration total and county sum migration total are identical (-7,237 persons), confirming the initial rate application is correct. The divergence emerges entirely from the compound effect in subsequent years: by year 2, the growing counties have more people receiving positive rates, and the shrinking counties have fewer people receiving negative rates, so the county sum's year-over-year delta exceeds the state's by +2,812 persons. This gap persists and grows (+2,000 to +4,000/year) throughout the 30-year horizon.

### Root Cause 2: Base Population Age Distribution Mismatch (Secondary, ~15% of divergence)

The base populations for state and counties have identical **totals** (799,358) but different **age-sex-race distributions**:

- **State base population**: Constructed by applying the statewide age-sex-race distribution (from the Vintage 2025 single-year distribution file) to the total state population. See `scripts/pipeline/02_run_projections.py`, lines 607-628, function `load_base_population()`.

- **County base populations**: Each county uses county-specific age-sex-race distributions (from `data/processed/county_age_sex_race_distributions.parquet`, with blending for small counties). See `cohort_projections/data/load/base_population_loader.py`, function `load_base_population_for_county()`.

The sum of 53 county distributions is **not identical** to the statewide distribution applied to the total. Key differences at 2025:

| Age Group | State | County Sum | Difference |
|-----------|:-----:|:----------:|:----------:|
| Reproductive-age females (15-49) | 179,848 | 181,703 | +1,856 (+1.0%) |
| Working age (20-64) | 448,940 | 451,486 | +2,546 (+0.6%) |
| Elderly (65+) | 138,247 | 134,930 | -3,317 (-2.4%) |

The county sum has slightly more reproductive-age females and fewer elderly than the state distribution. This means the county sum produces slightly more births and slightly fewer deaths from year 1 onward, adding a secondary upward bias of roughly 400-600 persons/year.

Total absolute cell-level difference across all age-sex-race combinations: 28,032 persons (3.5% of total), though these net to zero in aggregate.

### Why This Was Not Caught Earlier

1. **Validation not implemented**: `02_run_projections.py` lines 1444-1456 contain a stub: `"Hierarchical validation not yet implemented - skipping"`.
2. **Separate runs**: State and county projections can be run independently (`--state`, `--counties`), so the comparison requires deliberate cross-check.
3. **Identity at base year**: Both sum to 799,358 at 2025, masking the distributional difference.

### What Does Not Contribute to the Divergence

- **Fertility rates**: Shared statewide (`data/processed/fertility_rates.parquet`), identical for all geographies.
- **Survival rates**: Shared statewide (`data/processed/survival_rates.parquet`), identical for all geographies.
- **Time-varying mortality improvement**: Shared statewide, applied identically.
- **Convergence rate computation**: The convergence pipeline (`cohort_projections/data/process/convergence_interpolation.py`) produces only county-level rates. The state pipeline derives its rates from these by population-weighting. No independent state-level convergence computation exists.

## Decision Options

### Option A: Bottom-Up State (Derive State from County Sum) -- RECOMMENDED

**Eliminate the independent state-level projection entirely.** The state projection becomes the sum of 53 county projections.

**Implementation**:
1. When running `--all` or `--state`, instead of running the cohort-component engine for FIPS 38, aggregate the county results.
2. Use `multi_geography.py`'s existing `aggregate_to_state()` function (lines 563-615).
3. Remove the state-level base population construction logic (the statewide distribution approach).
4. The state-level migration rate construction (lines 1265-1365 of `02_run_projections.py`) becomes unnecessary.

**Rationale**:
- County projections are the primary product (users want county-level data).
- The county-level migration rates are the empirically grounded data (from PEP residual components per county).
- The state-level rate is a synthetic derivative (pop-weighted average) with no independent data source.
- Bottom-up aggregation is the standard practice at the Census Bureau and most state demography offices.
- The compound growth behavior is **correct** for the county sum -- it reflects the actual demographic dynamics of population redistribution.
- Eliminates the aggregation discrepancy definitionally.

**Consequences**:
- (+) Zero discrepancy by construction
- (+) State totals reflect actual county dynamics (population redistribution captured)
- (+) Simpler pipeline (no state-level rate construction needed)
- (+) Consistent with Census Bureau practice
- (-) Loses the independent state-level projection as a cross-check
- (-) State projection can only be produced when all 53 counties are projected (slower)
- (-) Any county projection error propagates to the state total

### Option B: Top-Down Constraint (Scale County Results to Match State)

**Run both state and county projections independently, then scale county populations to match the state total at each year.**

**Implementation**:
1. Run state projection as now.
2. Run all county projections as now.
3. At each year *t*, compute `scale_factor(t) = state_pop(t) / county_sum(t)`.
4. Multiply every county's population at year *t* by `scale_factor(t)`.

**Consequences**:
- (+) State total is the "anchor" -- useful if state-level projections have external validation (e.g., Census Bureau state projections)
- (+) Both state and county projections run independently, cross-checking each other
- (-) County projections are no longer internally consistent (a county's projection depends on all other counties)
- (-) The scaling distorts county-level demographic structure (growth counties are scaled down, decline counties are scaled up)
- (-) The state-level rate is itself synthetic with no independent data source, making it a poor anchor
- (-) Introduces a new source of methodological complexity

### Option C: Reconciliation Layer with Dynamic Reweighting

**Keep both independent runs but fix the state-level rate to use dynamic population weights.**

**Implementation**:
1. Instead of computing state migration rates from base-year population weights, recompute the weighted average at each projection year using the state population's own evolving age distribution.
2. Also align the state base population to match the county sum's age distribution.

**Consequences**:
- (+) Reduces the divergence substantially (addresses Root Cause 1)
- (+) State projection remains an independent computation
- (-) Requires passing county population trajectories into the state projection (circular dependency)
- (-) Does not fully eliminate divergence (compound growth within the state projection still differs from 53 independent compound growths)
- (-) Significant implementation complexity

### Option D: Hybrid (Bottom-Up Default + Independent State Cross-Check)

**Use bottom-up aggregation as the published state total, but retain the independent state projection as a diagnostic/validation tool.**

**Implementation**:
1. Default state projection = sum of counties (Option A).
2. Also run an independent state projection as a "shadow" calculation.
3. Report the divergence in metadata/logs as a diagnostic metric.
4. Flag if divergence exceeds a threshold (e.g., 5%) as a warning.

**Consequences**:
- (+) Best of both worlds: accurate published numbers + diagnostic cross-check
- (+) The divergence metric itself is informative (measures degree of population redistribution)
- (-) Slightly more complex pipeline
- (-) Users may be confused by two state numbers in metadata

## Recommended Decision

**Option A: Bottom-Up State (Derive State from County Sum)**, with elements of Option D (retain independent state as a diagnostic).

The primary rationale:

1. **The county projections are the authoritative product.** Users of this system need county-level data. The state total should be derived from those, not computed independently with a synthetic rate.

2. **The state-level migration rate has no independent empirical basis.** It is a population-weighted average of county rates using static weights -- a methodological convenience, not a data source. There is no "true" state-level migration rate distinct from the county dynamics.

3. **The compound growth behavior is correct at the county level.** Growing counties _should_ attract more migrants as they grow (larger labor markets, more housing, network effects). Declining counties _should_ lose fewer migrants as they shrink (fewer people left to leave). The county sum captures this correctly; the state projection does not.

4. **Standard practice.** The Census Bureau, BEBR (Florida), and most state demography offices use bottom-up aggregation as the primary method for deriving state totals from county projections.

5. **Retaining the independent state projection as a diagnostic** (Option D element) provides a useful metric: the divergence between independent-state and county-sum quantifies the degree of intra-state population redistribution implied by the projection. A large divergence suggests significant spatial reallocation, which is informative for planners.

## Consequences

### Positive

1. **Eliminates the 10.6% aggregation discrepancy** by construction -- the state total is the county sum.
2. **More accurate state totals** reflecting actual county-level population redistribution dynamics (compound growth captured).
3. **Simpler pipeline**: No need for state-level migration rate construction (lines 1265-1365 of `02_run_projections.py`).
4. **Consistent with standard practice** at Census Bureau and peer state demography offices.
5. **Base population consistency**: State age-sex-race distribution matches the actual county aggregate rather than a separately computed statewide distribution.
6. **Diagnostic value**: The divergence metric (county-sum vs independent-state) quantifies projected population redistribution.

### Negative

1. **Loses independent cross-check**: If a county projection has an error, it flows directly into the state total with no independent constraint.
2. **Slower state-only runs**: Cannot produce a quick state projection without running all 53 counties first (though the pipeline already runs all counties by default).
3. **Export adjustments**: Export workbooks and reports that currently use the state-level projection output will need to read the aggregated county data instead.
4. **Sensitivity to county coverage**: If any county is excluded (e.g., missing data), the state total will be incomplete. The pipeline must enforce that all 53 counties succeed before publishing a state total.

## Implementation Plan

### Phase 1: Bottom-Up Aggregation (Immediate)

| Step | File | Change |
|------|------|--------|
| 1 | `scripts/pipeline/02_run_projections.py` | After county projections complete, call `aggregate_to_state()` and save the result as the state-level projection file (replacing the independent run). |
| 2 | `cohort_projections/geographic/multi_geography.py` | Ensure `aggregate_to_state()` (lines 563-615) preserves all required columns and metadata. |
| 3 | `scripts/pipeline/02_run_projections.py` | In `run_geographic_projections()`, change the execution order: run counties first, then derive state from aggregation. Remove the state-level migration rate construction (lines 1265-1365). |
| 4 | `scripts/pipeline/02_run_projections.py` | In `load_base_population()`, for state FIPS, sum the actual county base populations rather than applying a statewide distribution. This eliminates Root Cause 2. |

### Phase 2: Validation Infrastructure

| Step | File | Change |
|------|------|--------|
| 5 | `scripts/pipeline/02_run_projections.py` | Implement the `validate_projection_results()` stub (lines 1414-1464). Call `validate_aggregation()` from `multi_geography.py` with actual data. |
| 6 | `cohort_projections/geographic/multi_geography.py` | In `validate_aggregation()`, add a check comparing county-sum with independent state projection. Log the divergence as a diagnostic metric. |
| 7 | Config | Consider adjusting `aggregation_tolerance` from 0.01 (1%) based on the expected diagnostic divergence range. |

### Phase 3: Diagnostic State Projection (Optional)

| Step | File | Change |
|------|------|--------|
| 8 | `scripts/pipeline/02_run_projections.py` | Optionally retain the independent state projection as a "shadow" run (not published). Log the divergence between shadow and county-sum. |
| 9 | Export scripts | Update `scripts/exports/build_detail_workbooks.py` and `scripts/exports/build_provisional_workbook.py` to read state totals from the county-aggregated file. |

### Testing Strategy

1. **Unit test**: Verify `aggregate_to_state()` produces correct totals matching the sum of county DataFrames.
2. **Integration test**: Run full pipeline with `--all`, verify state output equals county sum at every year.
3. **Regression test**: Verify county-level projections are unchanged (only the state output changes).
4. **Diagnostic test**: Run independent state projection, verify divergence is in expected range (8-12% at 2055 for baseline).
5. **Edge case**: Verify pipeline fails gracefully if any county projection fails and state total cannot be computed.

### Pipeline Rerun Required

1. **Step 02**: Projections (new aggregation logic)
2. **Step 03**: Exports (state totals change)

## Key File References

| File | Relevance |
|------|-----------|
| `scripts/pipeline/02_run_projections.py` | State migration rate construction (lines 1265-1365), base population loading (lines 592-631), validation stub (lines 1414-1464) |
| `cohort_projections/geographic/multi_geography.py` | `aggregate_to_state()` (lines 563-615), `validate_aggregation()` (lines 618-745) |
| `cohort_projections/data/process/convergence_interpolation.py` | County-only convergence rates, no state-level computation |
| `cohort_projections/core/cohort_component.py` | Engine applies rates identically regardless of geography level |
| `config/projection_config.yaml` | `validate_aggregation: true`, `aggregation_tolerance: 0.01` (lines 38-39) |

## References

1. **Census Bureau Population Projections Methodology**: State totals derived bottom-up from county/region projections
2. **BEBR (University of Florida)**: Uses county-level projections as primary product, state is the sum
3. **ADR-035**: Census PEP Components -- source data for county migration rates
4. **ADR-043**: Migration Rate Cap -- county-level rate caps in convergence interpolation
5. **ADR-046**: High Growth BEBR Convergence -- county-level migration rate redesign
6. **ADR-050**: Restricted Growth Additive Migration -- per-capita adjustment preserving county independence
7. **Jensen's Inequality**: Mathematical basis for why `E[f(X)] != f(E[X])` when f is nonlinear (here, compound population growth is the nonlinear function)

## Revision History

- **2026-02-23**: Initial version (Proposed) -- root cause analysis and recommendation
- **2026-02-23**: Accepted — Option A (Bottom-Up State) implemented; state projection derived from county aggregation

## Implementation Results

### Changes Made

| File | Change | Lines |
|------|--------|-------|
| `scripts/pipeline/02_run_projections.py` | Added `aggregate_county_results_to_state()` function for bottom-up state derivation; modified `run_geographic_projections()` to skip independent state engine run when counties are also requested; implemented `validate_projection_results()` with diagnostic logging | ~350 lines added |
| `scripts/projections/run_pep_projections.py` | Added `save_bottom_up_state_projection()` function; integrated into post-processing loop to save state parquet/metadata/summary after county runs complete | ~130 lines added |
| `cohort_projections/geographic/multi_geography.py` | Updated docstrings for `aggregate_to_state()` and `validate_aggregation()` to reflect ADR-054 | Docstring only |

### Decision Logic

When both "state" and "county" levels are requested (the `--all` flow):
1. The state level is **skipped** in the main processing loop
2. All 53 county projections run normally through the cohort-component engine
3. After counties complete, county parquet files are read, concatenated, and grouped by `(year, age, sex, race)` with population summed
4. The result is saved as the state-level projection file

When only `--state` is used without `--counties`, the old independent state projection behavior is preserved as a fallback.

### Verification

State projection totals now match county sums by construction (zero discrepancy):

| Scenario | 2025 | 2035 | 2045 | 2055 | 30yr Change |
|----------|------|------|------|------|-------------|
| Baseline | 799,358 | 836,864 | 880,325 | 900,971 | +12.7% |
| High Growth | 799,358 | 891,815 | 986,406 | 1,067,814 | +33.6% |
| Restricted | 799,358 | 804,264 | 835,611 | 842,885 | +5.4% |

Previously, the independent state projection diverged from county sums by 10.6% at 2055 (baseline). This divergence is now eliminated.

### Metadata

State projection files now include:
- `method: "bottom_up_county_aggregation"` — clearly identifies derivation method
- `adr: "ADR-054"` — links to this decision record
- `n_counties: 53` — confirms all counties were aggregated

### Tests

All 1,257 existing tests pass with zero regressions. No new test failures introduced.
