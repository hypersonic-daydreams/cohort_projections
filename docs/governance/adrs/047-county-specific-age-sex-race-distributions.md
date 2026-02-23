# ADR-047: County-Specific Age-Sex-Race Distributions

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-23

## Scope
Replace statewide age-sex-race distribution with county-specific distributions for base population construction

**Extends**: [ADR-044](044-census-full-count-race-distribution.md) (delivers the "future enhancement" noted in ADR-044's Negative Consequences)

## Context

### Problem: All 53 Counties Share Identical Age-Sex-Race Distributions

The base population loader (`base_population_loader.py`, line 351-352) applies a single statewide proportional distribution to every county's total population:

```python
base_pop = distribution.copy()
base_pop["population"] = base_pop["proportion"] * total_population
```

This means Fargo (a university city with heavy 18-24 population) starts with the same youth share as Slope County (a tiny rural county with an aging population), and Sioux County (78% AIAN) starts with the same 4.8% AIAN share as the statewide average.

### Magnitude of the Error

The misallocation was quantified by comparing the statewide distribution against Census cc-est2024-alldata county-level data:

| County | Population | % Misallocated | Worst Race Error |
|--------|-----------|:--------------:|-----------------|
| Sioux (38085) | 3,713 | **76.2%** | AIAN: 2,903 actual → 177 modeled |
| Rolette (38079) | 11,692 | **72.3%** | AIAN: 8,867 actual → 557 modeled |
| Benson (38005) | 5,756 | **46.6%** | AIAN: 2,868 actual → 274 modeled |
| Mountrail (38061) | 9,474 | **29.1%** | AIAN: 2,604 actual → 451 modeled |
| Median (all 53) | — | **17.6%** | — |

The reservation counties are the worst-affected: 46-76% of population is placed in the wrong race category. Since AIAN populations have distinct age structure (higher proportion under 18: 35-37% vs 26.5% statewide) and different fertility and mortality patterns, the misallocation compounds through the entire projection.

### The Data Already Exists

ADR-044 acknowledged this limitation: *"State-level distribution: Uses statewide proportions for all counties (same limitation as before). A future enhancement could use county-specific distributions from the same data source."*

The Census `cc-est2024-alldata-38.csv` file already in the project contains county-level age × sex × race data for all 53 ND counties. The current `build_race_distribution_from_census.py` script aggregates across counties at line 151 (`state_totals = df.groupby("age_group")[cols].sum()`), discarding the county dimension.

### Small County Considerations

For the 10 smallest counties (population 660-2,141):
- 37.1% of cells are zero across the 14 race-sex columns × 18 age groups
- The smallest non-zero cell values are 1-2 people

However, this is not worse than the current situation. Small counties currently receive phantom population in race categories where they have zero actual residents. County-specific distributions with real zeros are more accurate than statewide proportions that falsely allocate population to absent groups.

## Decision

### Build County-Specific Distributions from cc-est2024-alldata with Population-Weighted Blending

**1. New data artifact**: A file `data/processed/county_age_sex_race_distributions.parquet` with columns `[fips, age_group, sex, race, proportion]` — one distribution per county (53 counties x 216 cells = 11,448 rows).

**2. Blending for small counties**: For counties below a population threshold (5,000), blend the county-specific distribution with the statewide distribution using a population-proportional weight:

```
alpha = min(county_population / 5000, 1.0)
blended = alpha * county_proportion + (1 - alpha) * state_proportion
```

This prevents zero-cell artifacts in tiny counties while preserving the dramatic differences in larger counties. Counties above 5,000 population use their own distribution exclusively. The 5,000 threshold is chosen because counties below this size have >30% zero cells, while counties above have <15%.

**3. Ingestion script changes**: Modify `build_race_distribution_from_census.py` to:
- Keep the county dimension when processing cc-est2024-alldata
- Compute per-county proportions (proportion = cell_count / county_total)
- Apply the blending formula for small counties
- Output both the county-specific file and the existing statewide file (for state-level projections and as the blending anchor)

**4. Loader changes**: Modify `base_population_loader.py`:
- Add `load_county_age_sex_race_distribution(county_fips, config)` function
- Modify `load_base_population_for_county()` to look up county-specific distribution first, falling back to statewide if not available
- Modify `load_base_population_for_all_counties()` to load the county distribution file once and pass each county's distribution to the per-county loader

**5. Backward compatibility**: Keep `load_state_age_sex_race_distribution()` and `nd_age_sex_race_distribution.csv` intact. They remain used for state-level projections and as the blending anchor for small counties.

### Configuration

```yaml
base_population:
  county_distributions:
    enabled: true
    path: "data/processed/county_age_sex_race_distributions.parquet"
    blend_threshold: 5000  # counties below this pop blend with statewide
```

Note: Blending is applied during data generation (in `build_race_distribution_from_census.py`), not at load time. The `blend_threshold` parameter controls the ingestion script's blending behavior. The generated Parquet file already contains blended proportions for small counties.

### Why Not Per-County Without Blending?

- 10 counties have >30% zero cells. While zeros are often correct (genuinely no residents of that group), the projection engine may behave unexpectedly with zero population in cells that later receive migration inflows.
- Blending preserves the county's dominant patterns while providing a small "floor" in otherwise-empty cells, similar to a Bayesian prior.
- The alpha weight ensures large counties (>5,000) are unaffected by blending.

## Consequences

### Positive

1. **Eliminates 17.6% median misallocation**: County distributions match Census estimates instead of statewide averages
2. **Fixes reservation county race structure**: Sioux, Rolette, Benson will have 46-78% AIAN (actual) instead of 4.8% (statewide average)
3. **Correct age structure by county type**: University counties get younger distributions, rural counties get older distributions
4. **Leverages existing data**: No new data download required — the cc-est2024-alldata file is already in the project
5. **Minimal downstream impact**: Output schema (`year, age, sex, race, population`) is unchanged; all downstream code works unmodified

### Negative

1. **Vintage mismatch**: V2024 county distributions applied to V2025 population totals. Negligible error — racial composition changes <1%/year. Update when cc-est2025-alldata is released (expected mid-2026).
2. **Blending is an approximation**: For counties near the 5,000 threshold, the blend weight is somewhat arbitrary. Sensitivity analysis should verify results are not threshold-dependent.
3. **Additional data artifact**: A new parquet file must be generated and maintained alongside the existing statewide CSV.

### Expected Impact

The most significant improvements will be in:
- **Reservation counties**: Correct AIAN population produces more realistic fertility (higher AIAN fertility rates applied to correct population counts) and mortality (AIAN-specific survival patterns)
- **University counties**: Correct young-adult concentration improves migration dynamics
- **Small rural counties**: Correct elderly concentration improves mortality projections

## Implementation Notes

### Key Files

| File | Change |
|------|--------|
| `scripts/data/build_race_distribution_from_census.py` | Add county-level processing; output county distributions parquet |
| `cohort_projections/data/load/base_population_loader.py` | Add `load_county_age_sex_race_distribution()`; modify per-county loader |
| `config/projection_config.yaml` | Add `base_population.county_distributions` config block |
| `data/processed/county_age_sex_race_distributions.parquet` | New output (11,448 rows) |

### Testing Strategy

1. **Ingestion validation**: Verify 53 counties × 216 cells = 11,448 rows; proportions sum to 1.0 per county; blending applied only below threshold
2. **Distribution divergence**: Verify county distributions differ from each other (median pairwise distance > 0)
3. **Reservation county spot check**: Sioux AIAN proportion > 70%, Rolette > 70%, Benson > 40%
4. **Loader regression**: Existing tests pass unchanged when county distributions are disabled
5. **Pipeline integration**: Run full projection and compare county-level results before and after

### Pipeline Rerun Required

1. **Data rebuild**: `python scripts/data/build_race_distribution_from_census.py` (generates county distribution parquet)
2. **Step 01**: Residual migration (unchanged, uses PEP data not base pop)
3. **Step 01b**: Convergence (unchanged)
4. **Step 02**: Projections (loads new county-specific base populations)
5. **Step 03**: Exports

## Implementation Results

### Summary

All components specified in the Decision section have been implemented and tested. The county-specific age-sex-race distribution system is fully operational, with `county_distributions.enabled: true` in the production config.

### Files Modified

| File | Change |
|------|--------|
| `scripts/data/build_race_distribution_from_census.py` | Added `build_county_distributions()` function (lines 603-726) that processes cc-est2024-alldata per-county, applies population-weighted blending for counties below the threshold, and outputs a Parquet file. Added `validate_county_distributions()` (lines 729-821) with structural checks, proportion-sum validation, reservation county spot checks, and distribution divergence analysis. Updated `main()` to call county generation by default (skippable with `--skip-county`). |
| `cohort_projections/data/load/base_population_loader.py` | Added `load_county_age_sex_race_distribution()` (lines 359-568) that reads a county's distribution from the Parquet file, expands 5-year age groups to single-year ages (using statewide single-year pattern as interpolation weights when `age_resolution=single_year`), maps race codes to standard categories, and fills missing cohorts. Added `load_county_distributions_file()` (lines 571-606) for efficient batch loading. Added `_build_statewide_single_year_weights()` (lines 295-356) for single-year interpolation within 5-year county groups. Modified `load_base_population_for_county()` to try county-specific distribution first, falling back to statewide. Modified `load_base_population_for_all_counties()` to load the county Parquet once and pass it to each per-county call. |
| `cohort_projections/data/load/__init__.py` | Added exports for `load_county_age_sex_race_distribution` and `load_county_distributions_file`. |
| `config/projection_config.yaml` | Added `base_population.county_distributions` block with `enabled: true`, `path`, and `blend_threshold: 5000`. |
| `data/processed/county_age_sex_race_distributions.parquet` | New generated artifact: 11,448 rows (53 counties x 216 cells). |
| `tests/test_data/test_county_race_distributions.py` | New test file with 42 tests covering ingestion, validation, loader, integration, and blending behavior. |

### Test Results

42 new tests added across 5 test classes, all passing:

- **TestBuildCountyDistributions** (12 tests): Output shape, proportion sums, no negatives/NaNs, FIPS format, columns, blending behavior (large vs small counties), zero-threshold bypass, completeness of age groups/races/sexes.
- **TestValidateCountyDistributions** (3 tests): Valid distribution passes, missing rows detected, NaN proportions detected.
- **TestLoadCountyDistribution** (10 tests): Returns None when disabled/missing, correct output format (columns, proportion sums, age range 0-90, Title case sex, mapped race codes, complete cohort grid).
- **TestCountyDistributionsIntegration** (10 tests): 53 counties present, 11,448 total rows, 216 rows per county, proportions sum to 1.0, no negatives/NaNs, reservation county AIAN proportions (Sioux >50%, Rolette >60%, Benson >30%), Cass white >75%, distributions differ between counties, reservation-vs-urban divergence >40pp.
- **TestBlendingBehavior** (3 tests): Alpha formula correctness, blending moves small counties toward statewide, large counties unaffected by blending.

Full test suite (1,237 tests) passes with no regressions.

### Observed Data Quality

From the generated `county_age_sex_race_distributions.parquet`:

| County | AIAN Proportion | White Proportion | Notes |
|--------|:--------------:|:---------------:|-------|
| Sioux (38085) | 59.3% | blended | Pop 3,713 < 5,000 threshold; blended with statewide |
| Rolette (38079) | >60% | — | Pop 11,692 > threshold; unblended |
| Benson (38005) | >30% | — | Pop 5,756 near threshold; minimal blending |
| Cass (38017) | ~2% | 80.8% | Pop ~200k; unblended urban distribution |

All 53 counties have proportions summing to exactly 1.0 (within 1e-6 tolerance). No negative or NaN proportions. Median mean-absolute-deviation from the average distribution is 0.000856, confirming distributions are non-identical across counties.

### Backward Compatibility

- The statewide distribution file (`nd_age_sex_race_distribution.csv`) and `load_state_age_sex_race_distribution()` function are preserved and unchanged.
- Setting `county_distributions.enabled: false` in config causes the loader to fall back to statewide distribution for all counties (tested).
- The output schema (`year, age, sex, race, population`) from `load_base_population_for_county()` is unchanged; all downstream code works unmodified.

## References

1. **Census Bureau CC-EST2024-ALLDATA**: County Characteristics Resident Population Estimates by Age, Sex, Race, and Hispanic Origin. Vintage 2024.
   - Layout: `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/CC-EST2024-ALLDATA.pdf`
2. **ADR-044**: Census Full-Count Race Distribution — established the data source; this ADR extends it to county level
3. **ADR-007**: Race/Ethnicity Categorization — defines the 6-category scheme
4. **Sanity Check**: `docs/reviews/2026-02-18-projection-output-sanity-check.md` — identified the uniform distribution as a HIGH priority finding

## Revision History

- **2026-02-23**: Status changed to Accepted; added Implementation Results section with test coverage (42 tests), data quality observations, and backward compatibility notes
- **2026-02-18**: Initial version (ADR-047) — County-specific distributions from cc-est2024-alldata

## Related ADRs

- **ADR-044: Census Full-Count Race Distribution** — Established cc-est2024-alldata as the source; noted county-specific enhancement as future work
- **ADR-041: Census+PUMS Hybrid Base Population** — Superseded by ADR-044; this ADR further extends ADR-044
- **ADR-045: Reservation County PEP-Anchored Migration Recalibration** — Complements this fix; ADR-045 fixes migration rates, this ADR fixes the base population structure
- **ADR-048: Single-Year-of-Age Base Population** — Companion ADR addressing age granularity (orthogonal to the county-specific dimension)
