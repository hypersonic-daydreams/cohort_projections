# ADR-048: Single-Year-of-Age Base Population from Census SC-EST Data

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-23

## Scope
Replace uniform 5-year age group splitting with Census single-year-of-age estimates to eliminate step-function artifacts in projection outputs

**Related**: [ADR-044](044-census-full-count-race-distribution.md) (current distribution source), [ADR-047](047-county-specific-age-sex-race-distributions.md) (county-specific enhancement)

## Context

### Problem: Step-Function Artifacts from Uniform 5-Year Group Splitting

The base population loader (`base_population_loader.py`, lines 132-135) converts 5-year age group data to single-year-of-age by dividing each group's proportion equally across 5 years:

```python
proportion_per_year = row["proportion"] / len(ages)
```

This creates a staircase pattern: flat plateaus within each 5-year group with abrupt jumps at boundaries. For example, the proportion jumps ~4.4% between ages 29 and 30, then remains flat from 30 to 34, then jumps again at age 35.

### Consequences of the Artifacts

1. **Persistent through 2055**: The cohort-component engine advances cohorts one year at a time but never re-smooths. The initial step discontinuities propagate forward as permanent features, creating visible bumps and dips in every projection year's age distribution.

2. **Growth rate discontinuity at 2046-2047**: The largest step functions (at ages 20, 25, 30 in the 2025 base) create an abrupt -38% drop in annual growth rate as the artificially shaped cohorts from the 2025 base year age through key demographic thresholds.

3. **False precision**: The engine operates at single-year-of-age resolution (ages 0-90), but the input data only has 5-year resolution. The uniform split provides the appearance of single-year precision without the substance.

### How the SDC Avoids This

The North Dakota SDC 2024 projections operate entirely in 5-year age groups. They never disaggregate to single years, so they never produce step functions. Their approach avoids the artifact entirely but at the cost of coarser age resolution.

Our engine's single-year-of-age design is methodologically superior for tracking cohort dynamics, but requires genuine single-year input data to realize its advantage.

### Available Data: Census Publishes Single-Year-of-Age Estimates

The Census Bureau publishes two files with single-year-of-age data:

| File | Level | Age | Sex | Race | Status |
|------|-------|-----|-----|------|--------|
| `SC-EST2024-ALLDATA6` | **State** | Single year (0-85+) | Yes | Yes (6 groups) | **Available, not yet downloaded** |
| `CC-EST2024-SYASEX` | **County** | Single year (0-85+) | Yes | **No** | Available, not yet downloaded |

**SC-EST2024-ALLDATA6** is the primary solution: it provides state-level single-year × sex × race directly from Census full-count estimates. This eliminates the need for any interpolation at the state level.

For county-level single-year distributions (in conjunction with ADR-047), the county-level `CC-EST2024-SYASEX` provides single-year × sex without race. Race proportions can be applied from the 5-year group county data (cc-est2024-alldata) using Sprague interpolation to smooth within each group.

### Standard Demographic Practice

Four methods are used to convert 5-year age groups to single years, ordered from crudest to most sophisticated:

| Method | Smoothness | Group-total preservation | Negative values possible | Standard usage |
|--------|-----------|------------------------|-------------------------|----------------|
| **Uniform** (current) | Poor — step functions | Yes | No | Quick approximation only |
| **Karup-King-Newton** | Good | Yes | Rare | Some national offices |
| **Beers Modified** | Good | Yes | Rare | Less common |
| **Sprague** | Best | Yes | Possible (clamp to zero) | **UN Population Division standard** |

The Census Bureau and UN Population Division both use Sprague osculatory interpolation when converting 5-year estimates to single years. The current uniform method is not standard practice for production-quality population projections.

## Decision

### Use SC-EST2024-ALLDATA6 for State-Level; Sprague Interpolation for County-Level Race

**Part 1: State-level distribution (primary fix)**

1. **Download** `sc-est2024-alldata6.csv` from `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata6.csv`
2. **Filter** to North Dakota (STATE=38), most recent year estimate (YEAR=7 for July 1, 2024)
3. **Map** Census SEX/ORIGIN/RACE codes to the project's 6 race categories per ADR-007
4. **Build distribution** with 1,092 rows (91 ages × 2 sexes × 6 races) instead of the current 216 rows (18 groups × 2 sexes × 6 races)
5. **Output** to `data/raw/population/nd_age_sex_race_distribution.csv` (same filename, expanded schema)

**Part 2: Simplify the loader**

Remove the `AGE_GROUP_RANGES` expansion logic and the `proportion_per_year = row["proportion"] / len(ages)` uniform splitting entirely. The distribution file already contains single-year ages.

**Part 3: County-level (in conjunction with ADR-047)**

When ADR-047 introduces county-specific distributions, the county-level race data (from cc-est2024-alldata, which is 5-year groups) should be graduated to single years using Sprague interpolation. The county-level age × sex totals from CC-EST2024-SYASEX provide the single-year target for total population; Sprague is applied only to the race proportions within each 5-year group.

### Why SC-EST Over Sprague for the State Level

SC-EST2024-ALLDATA6 provides actual Census single-year estimates — not interpolated values. Using real data is always preferred over statistical interpolation:
- No risk of negative values (a known Sprague edge case)
- No methodological debate about interpolation method
- The project already has infrastructure for fetching `sc-est` files (`census_api.py`, line 427)
- Simplifies the code (removes expansion logic rather than adding interpolation logic)

### Sprague Interpolation for County Race (When Needed)

For county-level race distributions (ADR-047), where only 5-year group data is available, implement Sprague osculatory interpolation:

```python
# Standard Sprague multiplier matrix (5 coefficients × 5 single years per group)
# Applied to each sex × race combination's 18-group vector
# Clamp any negative values to zero
# Renormalize so the 5 interpolated values sum to the original group total
```

The Sprague multipliers are well-documented and widely available (UN Population Division, R DemoTools package). The implementation is approximately 30 lines of code.

### Configuration

```yaml
base_population:
  age_resolution: "single_year"  # "single_year" or "five_year_uniform" (legacy)
  state_distribution_source: "sc-est2024-alldata6"
  county_race_interpolation: "sprague"  # Only used when county distributions have 5-year race groups
```

## Consequences

### Positive

1. **Eliminates step-function artifacts**: Age distribution is smooth and realistic at single-year resolution
2. **Eliminates the 2046-2047 growth discontinuity**: No more artificial cohort-boundary effects
3. **Uses authoritative Census data**: SC-EST provides the Census Bureau's own single-year estimates, not a statistical approximation
4. **Simplifies code**: Removes the `AGE_GROUP_RANGES` expansion logic from the loader
5. **Consistent with engine design**: The engine already operates at single-year-of-age resolution; the input now matches

### Negative

1. **Vintage mismatch**: SC-EST2024 is July 2024; base year is 2025. The one-year gap introduces negligible error in age proportions. Update when SC-EST2025 is released.
2. **Schema change**: The distribution CSV changes from 216 rows (age groups) to 1,092 rows (single years). Any code that reads the CSV expecting the old schema may need updates. However, the loader already expands to single years internally, so downstream code is unaffected.
3. **Sprague edge cases**: For very small county race groups, Sprague may produce negative values at extreme ages (85+). These must be clamped to zero and the group total renormalized.
4. **Additional data dependency**: A new Census file must be downloaded and maintained.

### What This Changes in the Output

| Metric | Before (uniform) | After (SC-EST) |
|--------|-------------------|----------------|
| Age distribution shape | Staircase with 5-year steps | Smooth, reflecting actual single-year variation |
| Max age-to-age jump (non-infant) | ~20% at group boundaries | ~2-3% (natural demographic variation) |
| 2046-2047 growth discontinuity | -38% drop in annual growth rate | Smooth transition |
| Population at age 30 vs 29 | Abrupt 4.4% jump | Gradual change |

## Implementation Notes

### Key Files

| File | Change |
|------|--------|
| `scripts/data/build_race_distribution_from_census.py` | Add SC-EST ingestion path; generate 1,092-row distribution |
| `cohort_projections/data/load/base_population_loader.py` | Remove `AGE_GROUP_RANGES` expansion; read single-year distribution directly |
| `data/raw/population/nd_age_sex_race_distribution.csv` | Expanded from 216 to 1,092 rows |

### New Data File

| File | Source |
|------|--------|
| `data/raw/population/sc-est2024-alldata6.csv` | Census Bureau FTP (state-level single-year × sex × race) |

### Ingestion Script Changes

The `build_race_distribution_from_census.py` script will be extended with a second data path:
1. **State-level (SC-EST)**: Read SC-EST2024-ALLDATA6, filter to ND, map race codes, output single-year distribution
2. **County-level (cc-est2024-alldata + Sprague)**: When generating county distributions (ADR-047), apply Sprague graduation to the 5-year race groups

### Testing Strategy

1. **Distribution smoothness**: Verify no age-to-age proportion jump exceeds 5% (excluding infant mortality drop at age 0-1)
2. **Proportion sum**: Single-year proportions sum to 1.0
3. **Group consistency**: Sum of single-year proportions within each 5-year group matches the original 5-year group proportion (within rounding tolerance)
4. **Projection smoothness**: Re-run state baseline projection; verify year-over-year growth rate has no discontinuity exceeding 0.1 percentage points
5. **Regression**: Existing loader tests updated for new schema

### Pipeline Rerun Required

1. **Data download**: Fetch SC-EST2024-ALLDATA6 from Census FTP
2. **Data rebuild**: `python scripts/data/build_race_distribution_from_census.py` (generates single-year distribution)
3. **Step 02**: Projections (loads new single-year base populations)
4. **Step 03**: Exports

## References

1. **Census Bureau SC-EST2024-ALLDATA6**: Annual State Resident Population Estimates for 6 Race Groups by Age, Sex, and Hispanic Origin.
   - Data: `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata6.csv`
   - Layout: `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/SC-EST2024-ALLDATA6.pdf`
2. **Census Bureau CC-EST2024-SYASEX**: County Population by Single Year of Age and Sex.
   - Layout: `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/CC-EST2024-SYASEX.pdf`
3. **Sprague Multipliers**: Sprague, T. B. (1880). "Explanation of a new formula for interpolation." *Journal of the Institute of Actuaries*, 22, 270-285.
4. **DemoTools R Package**: `graduate_sprague()` function. https://timriffe.github.io/DemoTools/reference/graduate_sprague.html
5. **UN Population Division**: Uses Sprague interpolation in World Population Prospects methodology.

## Implementation Results (2026-02-23)

### Changes Made

| File | Change |
|------|--------|
| `cohort_projections/utils/demographic_utils.py` | Added `SPRAGUE_MULTIPLIERS` constant (5x5 standard coefficient matrix), `_pad_groups()` helper (linear extrapolation of 2 virtual groups on each boundary), and `sprague_graduate()` function implementing Sprague osculatory interpolation with optional negative clamping and renormalization. ~100 lines including docstrings. |
| `cohort_projections/data/load/base_population_loader.py` | Added `_ORDERED_AGE_GROUPS` constant (18 standard 5-year groups), `_expand_county_with_sprague()` function that applies Sprague graduation per sex-race combination, and updated `load_county_age_sex_race_distribution()` to support three interpolation modes (`"sprague"`, `"statewide_weights"`, `"five_year_uniform"`) controlled by config parameter `base_population.county_race_interpolation`. Terminal age group (85+) uses exponential decay with survival factor 0.7 for the 5 single years 85-89. |
| `config/projection_config.yaml` | Added `county_race_interpolation: "sprague"` parameter under `base_population`. This selects Sprague as the default method for county-level 5-year-to-single-year race distribution expansion. |
| `tests/test_utils/test_demographic_utils.py` | Added `TestSpragueGraduate` class with 13 unit tests: output length, group-total preservation, total-population preservation, smoothness (no step functions), no boundary steps, non-negative with clamping, clamping preserves totals, minimum groups validation, 5-group minimum, all-zeros, single-nonzero-group, monotonic decline at old ages, and Sprague-vs-uniform smoothness comparison. |
| `tests/test_data/test_county_race_distributions.py` | Added `TestSpragueCountyInterpolation` class with 7 integration tests: output shape (91 ages x 2 sexes x 6 races), proportions sum to 1, no negative proportions, Sprague smoother than uniform, no step at group boundaries, age range 0-90, and statewide-weights fallback. Added `_build_varying_county_df()` helper with realistic age-gradient data. |

### Verification

- **Full test suite**: 1,257 tests passed (5 skipped), 0 failures, 0 regressions. Run time 162.5s.
- **Group-total preservation**: Each 5-year group's sum of interpolated single-year values matches the original total to within floating-point precision (< 1e-10 relative error).
- **Smoothness**: Maximum age-to-age proportion change (ages 2+) is bounded. With realistic population data, Sprague produces 40-60% lower roughness (sum of squared second differences) compared to uniform splitting.
- **No boundary steps**: The proportion difference at each 5-year group boundary (ages 5, 10, 15, ...) does not exceed 5% of the local mean, eliminating the step-function artifacts.
- **Non-negativity**: All interpolated values are non-negative after clamping and renormalization.
- **Backward compatibility**: The `"statewide_weights"` and `"five_year_uniform"` modes remain available for comparison or fallback.

### Key Design Notes

1. **Padding approach over full coefficient table**: The implementation uses 2 linearly-extrapolated virtual groups on each boundary rather than distinct coefficient sets for boundary groups. This guarantees exact group-total preservation because the standard center-group multipliers have column sums = [0, 0, 1, 0, 0]. The alternative (Siegel & Swanson full table) has known transcription errors in published sources.
2. **Terminal age group (85+)**: The 85+ open-ended group cannot be interpolated by Sprague (it represents an unknown age span). Instead, it is distributed using exponential decay with a survival factor of 0.7 per year, producing a realistic decline across ages 85-89. The 90+ remainder is implicitly handled by the engine's open-ended final age group.
3. **Three interpolation modes**: The config parameter `county_race_interpolation` supports `"sprague"` (default, recommended), `"statewide_weights"` (original ADR-047 method), and `"five_year_uniform"` (legacy uniform split). This allows A/B comparison and graceful fallback.
4. **State-level already resolved**: Part 1 of this ADR (SC-EST2024-ALLDATA6 for state-level distributions) was implemented previously via `scripts/data/build_race_distribution_from_census.py`, which generates the 1,092-row single-year distribution file. This implementation session focused on Part 3 (county-level Sprague interpolation).

## Revision History

- **2026-02-23**: Accepted and implemented — added Sprague interpolation in `demographic_utils.py`, integrated into `base_population_loader.py` with three configurable modes, added 20 new tests (13 unit + 7 integration), updated config
- **2026-02-18**: Initial version (ADR-048) — Single-year-of-age from Census SC-EST data

## Related ADRs

- **ADR-044: Census Full-Count Race Distribution** — Current 5-year group source; this ADR upgrades the age resolution
- **ADR-047: County-Specific Age-Sex-Race Distributions** — Companion ADR for county-level enhancement; this ADR addresses the orthogonal age-resolution dimension
- **ADR-004: Core Projection Engine** — Engine operates at single-year-of-age; this ADR aligns input data with engine resolution
