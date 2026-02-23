# ADR-041: Census+PUMS Hybrid Base Population Age-Sex-Race Distribution

## Status
Superseded by ADR-044

## Date
2026-02-17

## Last Reviewed
2026-02-18

## Scope
Base population age-sex-race distribution methodology for county projections

## Context

### Problem: PUMS-Only Distribution Produces Skewed Sex Ratios

The projection system needs an age-sex-race distribution to allocate county total populations into detailed cohorts (single-year age x sex x 6 race/ethnicity categories). The base population loader (`cohort_projections/data/load/base_population_loader.py`) applies this distribution to each county's total population to create the cohort matrix required by the projection engine.

The original distribution file (`data/raw/population/nd_age_sex_race_distribution.csv`) was derived entirely from ACS PUMS (American Community Survey Public Use Microdata Sample) for North Dakota. PUMS is a ~1% sample -- only 12,277 records for ND. This produced a badly skewed sex ratio of 119.1 males per 100 females (actual Census ratio: 105.5).

Some age-sex cells had absurd ratios (e.g., 80-84 age group: 2.63 males per female). The skewed distribution propagated through all county projections, inflating male populations and distorting age structure (65+ share was 20.1% instead of realistic 17.3%).

### Why PUMS Alone Is Insufficient

PUMS sampling noise is acceptable for large states where the sample provides tens of thousands of records. For North Dakota (population ~800,000), the ~1% sample yields only ~12,000 records spread across 18 age groups x 2 sexes x 6-8 race categories. Many cells have fewer than 50 observations, producing highly unreliable proportions. The sex ratio distortion is a direct consequence of this small-sample problem.

### Requirements

- The age-sex proportions must match Census Bureau official estimates (which are based on full-count or near-full-count data)
- Race allocation must still account for within-cell variation across race/ethnicity categories
- The solution must produce a CSV file compatible with the existing loader pipeline
- The distribution must normalize to 1.0 across all cohorts

## Decision

### Decision: Use Census for Age-Sex, PUMS for Race Allocation

**Decision**: Use Census Bureau County Characteristics Estimates (cc-est2024-agesex-all) for accurate age-sex proportions, and use PUMS only for race allocation within each age-sex cell (where sample noise matters less since race shares are relative proportions within cells).

**Rebuild process**:

1. Load Census cc-est2024 parquet, filter to ND (STATE='38'), YEAR='6' (2024 estimate)
2. Sum all 53 counties to get state age-sex totals by 5-year age group
3. Compute `age_sex_proportion = census_count / total_population`
4. Load original PUMS file, compute race shares within each `age_group x sex` cell
5. Multiply: `final_proportion = age_sex_proportion * race_share_within_cell`
6. Normalize to sum to 1.0

**Census source file**: `~/workspace/shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet`

**Output file**: `data/raw/population/nd_age_sex_race_distribution.csv` (115 rows, 18 age groups x ~6-8 race categories per sex)

**Rationale**:

- **Census age-sex proportions are authoritative**: The cc-est2024 file is derived from the Census Bureau's population estimates program, which uses administrative records, demographic analysis, and postcensal methodology. These are not sample-based -- they represent the Bureau's best estimate of the full population by age and sex.
- **PUMS race shares are adequate within cells**: Once the age-sex structure is fixed by Census data, PUMS only needs to allocate the race distribution *within* each age-sex cell. These within-cell race shares are relative proportions (e.g., "of males age 25-29, what fraction is Hispanic?"). Sample noise in these proportions is smaller in magnitude and has less impact on projections than the gross sex ratio error it replaces.
- **Minimal pipeline disruption**: The output file has the same schema (age_group, sex, race_ethnicity, proportion) as the original PUMS-only file. The loader code requires no changes.

**Alternatives Considered**:

- **Full Census race data (cc-est2024 by race)**: The Census Bureau does publish population estimates by race, but the available vintage covers broad race categories without the Hispanic/non-Hispanic cross-tabulation needed for the 6-category scheme. Using PUMS for within-cell race allocation avoids this limitation.
- **ACS 5-year detailed tables (B01001 series)**: These provide age-sex-race distributions from the 5-year ACS, which has a larger sample than PUMS 1-year. However, the 5-year ACS estimates for small populations still carry sampling error, and the cc-est2024 is strictly superior for age-sex proportions.
- **Iterative Proportional Fitting (IPF)**: IPF could combine Census age-sex marginals with PUMS race conditionals in a more statistically rigorous framework. Deferred as unnecessary given the simple multiplicative approach produces accurate results.

## Consequences

### Positive

1. **Sex ratio corrected**: From 119.1 to 105.5 males per 100 females, matching the Census Bureau estimate
2. **Male/female split corrected**: 51.3% / 48.7% (matching Census), was 54.4% / 45.6%
3. **Age structure corrected**: 65+ share is 17.3% (was 20.1%), consistent with Census estimates for ND
4. **State population 2025 unchanged**: 799,358 (the total comes from county population data, not the distribution)
5. **Projection endpoint corrected**: State population 2055 changed from 1,020,775 to 1,005,281 due to corrected age-sex structure feeding into the cohort-component engine
6. **No code changes required**: The rebuilt distribution file has the same schema as the original; the loader reads it identically

### Negative

1. **PUMS race allocations are still approximate**: Within-cell race shares from a ~1% sample are noisy, particularly for small race categories (AIAN, multiracial) in older age groups. The error is much smaller than the age-sex error it replaced, but it is not zero.
2. **Hybrid methodology is less elegant**: The approach mixes two data sources at different levels of the distribution hierarchy. This requires documentation (this ADR) to ensure future maintainers understand the provenance.
3. **Manual rebuild process**: The distribution file was rebuilt using an ad-hoc script rather than an automated pipeline step. Future data updates (when cc-est2025 becomes available) will require repeating the rebuild.

### Pipeline Rerun Required

The corrected distribution file affects only the base population allocation step. The following pipeline steps require rerunning:

- **Step 02**: Population projections (base population cohorts change)
- **Step 03**: Export workbooks (to reflect updated results)

Steps 00, 01, and 01b are unaffected because they do not depend on the age-sex-race distribution.

## Implementation Notes

### Key Files
- `data/raw/population/nd_age_sex_race_distribution.csv` -- the rebuilt distribution file (hybrid Census+PUMS)
- `cohort_projections/data/load/base_population_loader.py` -- loads the distribution and applies it to county populations

### Configuration Integration
No configuration changes are required. The distribution file path is resolved by the loader using the project root convention (`data/raw/population/nd_age_sex_race_distribution.csv`). The file schema is unchanged.

### Testing Strategy
1. **Distribution validation**: Verify proportions sum to 1.0; verify sex ratio is approximately 105.5; verify 65+ share is approximately 17.3%
2. **Loader regression**: Run `load_state_age_sex_race_distribution()` and verify output shape and column names are unchanged
3. **Projection comparison**: Run baseline projections for a representative county (e.g., Cass) and verify population totals match county input data; compare age structure to Census benchmarks

## References

1. **Census Bureau County Characteristics Estimates (cc-est2024-agesex-all)**: Source for age-sex proportions. Published as part of the Vintage 2024 Population Estimates program.
2. **ACS PUMS 1-Year (North Dakota)**: Source for within-cell race/ethnicity shares. ~12,277 person records.
3. **Census PEP Vintage 2025**: Source for county total populations used as the base population input.

## Revision History

- **2026-02-17**: Initial version (ADR-041) -- Document Census+PUMS hybrid approach for base population distribution

## Related ADRs

- **ADR-007: Race/Ethnicity Categorization** -- Defines the 6-category race/ethnicity scheme used in the distribution
- **ADR-004: Core Projection Engine** -- Defines the cohort-component methodology that consumes the base population distribution
- **ADR-035: Census PEP Components of Change for Migration Inputs** -- Related Census data source usage
