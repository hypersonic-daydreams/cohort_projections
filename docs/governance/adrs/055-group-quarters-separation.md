# ADR-055: Group Quarters Population Separation

## Status
Accepted — Amended by [ADR-061](061-college-fix-model-revision.md) (fractional GQ correction parameter)

## Date
2026-02-23

## Last Reviewed
2026-03-04

## Scope
Separate group quarters (GQ) populations from household populations in the projection pipeline to prevent institutional population rotations (military PCS cycles, university enrollment turnover) from distorting migration rates and eroding institutional populations that are structurally stable.

**Related**: [ADR-049](049-college-age-smoothing-convergence-pipeline.md) (college-age smoothing), [ADR-052](052-ward-county-high-growth-floor.md) (high-growth migration floor), [ADR-045](045-reservation-county-pep-recalibration.md) (reservation county recalibration)

## Context

### Problem: Institutional Population Rotations Distort Migration Rates

The residual migration method computes county-level age-sex migration rates using **total resident population** (Respop), which includes both household population and group quarters (GQ) population. Group quarters include military barracks, college dormitories, nursing facilities, and correctional institutions.

For counties with significant institutional populations, this creates systematic distortions:

1. **University dorm rotation appears as migration.** When a UND freshman replaces a graduating senior in the same dorm bed, the residual method counts an in-migrant at age 18 and an out-migrant at age 25. Both are "real" in Census accounting, but neither represents a change in the county's settled population. The beds remain filled; the institution continues operating.

2. **Military PCS (Permanent Change of Station) transfers appear as migration.** When a Minot AFB airman rotates out after a 3-4 year assignment and is replaced by another, the residual method sees out-migration and in-migration. But the base population is determined by force structure, not county-level demographic trends.

3. **The asymmetric college cycle.** ADR-049 smooths the student *arrival* signal (ages 15-24) with a 50/50 statewide blend, but the student *departure* signal (ages 25-29) is unsmoothed. This creates an asymmetry where arrivals are dampened but departures are counted at full value, making the net college cycle strongly negative in the model. For Grand Forks County, the net college cycle effect (ages 15-34) is **-0.069 annual rate** — UND enrollment is a net population *drain* in the projection.

4. **GQ populations erode under negative migration.** When the projection engine applies negative migration rates to the total population (including GQ), it gradually reduces the institutional population over 30 years. Ward County's baseline projects military-age (18-40) population dropping from 24,840 to 16,857 — a loss of ~8,000 people, substantially more than the entire MAFB active-duty population. This implies either MAFB closes or the civilian economy collapses, neither of which the model is designed to predict.

### Affected Counties

| County | GQ Pop (2025 PEP) | GQ % | Major GQ Institutions |
|--------|------------------:|:----:|----------------------|
| Grand Forks (38035) | 4,512 | 6.1% | UND dorms (~2,800), GFAFB barracks (~135), nursing (~750) |
| Cass (38017) | 8,684 | 4.3% | NDSU dorms (~3,900), nursing (~1,150) |
| Burleigh (38015) | 5,274 | 5.1% | U of Mary/BSC dorms (~1,550), correctional (~1,050), nursing (~1,180) |
| Ward (38101) | 2,161 | 3.2% | MAFB barracks (~590), Minot State dorms (~440), nursing (~480) |

GQ type breakdowns are from the 2020 Census DHC; total GQ from PEP Vintage 2025 stcoreview.

### Root Cause Analysis

The fundamental issue is that the residual migration formula uses total population in both numerator and denominator:

```
expected = pop_start[age] × survival_rate[age]
migration = pop_end[next_age] - expected
rate = migration / expected
```

When `pop_start` and `pop_end` include GQ residents, institutional turnover inflates the apparent migration flows. A university with 3,000 dorm residents that completely turns over every 4 years generates ~750 "in-migrants" and ~750 "out-migrants" per year that are artifacts of institutional operation, not settlement patterns.

The current mitigations (ADR-049 college-age smoothing, ADR-043 rate cap, ADR-052 high-growth floor) treat symptoms of this root cause:
- **ADR-049** dampens the arrival signal but not the departure signal
- **ADR-043** caps extreme rates but the cap (-0.080) still allows substantial GQ-driven out-migration
- **ADR-052** fixes the high-growth scenario but baseline and restricted still decline

### Why Smoothing the 25-29 Rate Is Not the Right Fix

Extending college-age smoothing to the 25-29 age group (as initially recommended in the Ward/Grand Forks review) would treat the symptom rather than the cause:

1. **Masks real brain drain.** Grand Forks genuinely doesn't retain graduates like Fargo does. Cass County's 25-29 rate (-0.044) is still negative — blending Grand Forks's rate with the statewide average would mask the real retention gap.

2. **Doesn't address military.** Military PCS rotations affect ages 18-35 and are not covered by college-age smoothing.

3. **Rate-based fixes don't prevent GQ erosion.** Even with smoothed rates, the projection engine still applies migration rates to the GQ population, gradually eroding institutional populations that are structurally stable.

4. **Treats one-leg of a two-legged problem.** The college cycle has both an arrival leg (ages 15-24) and a departure leg (ages 25-29). Smoothing one or both legs is an ad-hoc patch. Separating GQ addresses the root cause — the beds are not part of the migratory population.

### What Other States Do

Standard practice in state demography offices is to separate GQ from household population:

- **North Carolina OSBM**: Projects household population only, adds GQ held constant at most recent estimate level.
- **New York City DCP**: Holds GQ constant; changes in institutional populations are policy decisions, not demographic processes.
- **New Hampshire OPD**: Removes GQ from base population before computing age-specific rates. GQ added back as constant after projecting household population.
- **California DOF** and **Washington OFM**: Project certain GQ categories separately using facility-specific knowledge (enrollment projections, force structure plans).

The consensus: GQ populations distort age-specific demographic rates and should be separated from the rate-computation step, even if the ultimate treatment is simply to hold them constant.

### Data Availability

| Data Source | Years | GQ Detail | Age-Sex Detail | County Coverage | Status |
|-------------|-------|-----------|----------------|-----------------|--------|
| PEP stcoreview V2025 | 2020-2025 | Total only | 3 broad groups (0-17, 18-64, 65+) | All 53 | Already ingested |
| 2020 Census DHC (PCO1) | 2020 | Total | 5-year age × sex | All 53 | Available via API |
| 2020 Census DHC (PCO8) | 2020 | College housing | 5-year age × sex | All 53 | Available via API |
| 2020 Census DHC (PCO9) | 2020 | Military quarters | 5-year age × sex | All 53 | Available via API |
| 2010 Census SF1 (PCT20) | 2010 | By type | Some age-sex | All 53 | Available via API |

**Key data gap:** GQ by age × sex × race is not available at the county level. Race-specific GQ data is only available at the state level (ACS B26101). This means we cannot separate GQ from household population in the race dimension without assuming GQ has the same race distribution as the county.

## Decision

### Separate GQ from Household Population in Phased Implementation

**Phase 1 (Implemented): Base Population Separation + GQ Hold-Constant**

1. **Fetch 2020 Census DHC PCO1 data** (GQ by 5-year age × sex) for all 53 ND counties via Census API
2. **Estimate 2025 GQ by age × sex** by scaling the 2020 DHC age-sex proportions to 2025 stcoreview GQ totals (preserving the 2020 age-sex shape but matching the current GQ total)
3. **Distribute GQ to single-year ages** using the same Sprague interpolation method (ADR-048) already used for the total population
4. **Distribute GQ across races** proportionally to each county's overall race distribution (since race-specific GQ data is unavailable at the county level)
5. **Subtract GQ from base population** before feeding into the cohort-component engine. The engine projects only the household population.
6. **After projection, add GQ back as a constant** (at 2025 levels) to produce the final output. This means the GQ population is frozen at the base year — dorm beds stay filled, barracks stay staffed, nursing facilities maintain capacity.

**Configuration:**

```yaml
base_population:
  group_quarters:
    enabled: true
    method: "hold_constant"  # GQ held at base-year levels through projection horizon
    data_sources:
      census_2020_dhc: "data/processed/gq_county_age_sex_2020.parquet"
      pep_stcoreview: "data/raw/population/stcoreview_v2025_nd_parsed.parquet"
    scaling_method: "proportional"  # Scale 2020 DHC age-sex profile to 2025 stcoreview totals
    race_distribution: "county_proportional"  # Distribute GQ across races using county totals
```

**Phase 2 (Implemented): GQ-Corrected Migration Rates**

1. Estimate historical GQ at all 6 time points (2000, 2005, 2010, 2015, 2020, 2024):
   - **2020 and 2024**: Use stcoreview GQpop for those specific years
   - **2000, 2005, 2010, 2015**: Use 2020 GQ levels as a backward constant (institutional capacity changes slowly)
2. Subtract GQ from population snapshots before computing residual migration rates
3. Recompute residual migration rates on household-only population
4. The convergence pipeline, projection engine, and export pipeline all read the updated rates automatically

The backward-constant assumption for pre-2020 years is defensible because institutional capacity (barracks, dorms, nursing beds) changes slowly over 5-10 year windows. The primary goal is removing institutional rotation from rates, not precisely tracking historical GQ changes. Can be refined with decennial Census GQ data (SF1/DHC) if greater precision is needed.

### Why This Approach

1. **Addresses root cause.** Instead of smoothing, capping, or flooring rates that are distorted by GQ rotation, we remove GQ from the population that migration rates are applied to.

2. **Simple hold-constant assumption is defensible.** Military base populations are determined by force structure decisions (MAFB's nuclear deterrence mission spans decades). University dormitory capacity is determined by enrollment and construction decisions. Nursing facility capacity is determined by licensing and construction. None of these are driven by county-level demographic trends.

3. **Conservative.** Holding GQ constant means we project no growth in institutional capacity. In reality, NDSU has been expanding dorm capacity (Cass GQ grew from 5,757 in 2020 to 8,684 in 2025). The hold-constant assumption under-projects for growing institutions and over-projects for shrinking ones. For a projection model, this cautious approach is appropriate.

4. **Race approximation is reasonable.** GQ populations do have different racial compositions than the general population (e.g., military bases are more diverse than rural ND counties), but at the county level, the population is small enough that the proportional distribution is an acceptable approximation.

### Alternatives Considered

| Option | Description | Verdict |
|--------|-------------|---------|
| A: Extend college-age smoothing to 25-29 | Smooth departure signal like arrival signal | Rejected — masks real brain drain, doesn't fix military, doesn't prevent GQ erosion |
| B: Military population floor | Hold military-age male population constant for base counties | Rejected — too specific, doesn't address college/nursing GQ |
| **C: Full GQ separation (chosen)** | Separate GQ from household pop, project HH only, add GQ back | **Selected** — addresses root cause, standard practice, data available |
| D: GQ sub-model with facility-level projections | Project each GQ category with institutional data | Rejected — too complex, requires data we don't have, unpredictable policy inputs |
| E: Adjust migration rates with GQ fraction | Scale rates by HH/total ratio | Rejected — doesn't prevent GQ erosion in projection, ad-hoc |

## Consequences

### Positive

1. **Eliminates GQ erosion.** Institutional populations (military, university, nursing) are no longer projected to decline under negative migration rates. Ward County's MAFB population stays constant instead of eroding by ~8,000 people.
2. **Addresses the root cause** of the Grand Forks college cycle asymmetry. UND dorm residents are no longer subject to migration rates — they are held constant. The household population is projected with rates that (in Phase 1) still include some GQ signal, but at least the GQ population itself isn't being reduced.
3. **Standard practice.** Aligns with NC OSBM, NH OPD, NYC DCP, and other state demography offices.
4. **Replaces ad-hoc fixes with principled approach.** College-age smoothing (ADR-049) and the rate cap (ADR-043) remain in the pipeline as additional safeguards, but the primary GQ distortion is addressed structurally.

### Negative

1. **Hold-constant assumption is imperfect.** Some GQ populations change over time (Cass GQ grew 50% from 2020-2025). Holding constant may understate growth for expanding institutions.
2. **Race distribution approximation.** Using county-wide race proportions for GQ introduces error for counties where GQ racial composition differs significantly from the general population.
3. **Migration rates still computed on total population (Phase 1).** The residual migration rates are still derived from total Respop, meaning some GQ rotation signal remains embedded in the rates applied to household population. Phase 2 would address this.
4. **Pipeline rerun required.** All scenarios must be re-run after implementation.

### Actual Impact (Full Pipeline: Phase 1 + Phase 2)

| County | Pre-ADR-055 | Phase 1 Only | Phase 1 + Phase 2 | Net Change |
|--------|:-----------:|:------------:|:-----------------:|:----------:|
| Grand Forks | -8.7% | -9.0% | -11.9% | -3.2pp |
| Ward | -14.6% | -14.8% | -15.5% | -0.9pp |
| Cass | +29.8% | +27.1% | +25.2% | -4.6pp |
| Burleigh | +20.0% | +17.5% | +18.1% | -1.9pp |
| **State** | **+12.7%** | **+11.9%** | **+10.4%** | **-2.3pp** |

Note: Phase 2 produces more conservative projections than initially estimated because the dominant effect is removing recent GQ growth (especially NDSU's +2,929 dorm beds) from the 2020-2024 migration signal, not improving declining counties' rates.

### Interaction with Existing ADRs

- **ADR-049 (college-age smoothing):** Remains active. Still smooths 15-24 rates for college counties. The smoothing addresses rate distortion; GQ separation addresses population distortion. Both are needed in Phase 1 (before Phase 2 corrects the rates).
- **ADR-052 (high-growth migration floor):** Remains active. Still lifts high-growth convergence rates to zero minimum. With GQ separation, the floor may affect fewer counties since some baseline rates may improve.
- **ADR-043 (rate cap):** Remains active. Still clips extreme rates. With GQ separation, fewer rates may hit the cap since GQ rotation is removed from the base population.
- **ADR-054 (bottom-up state):** Compatible. State projection is the sum of county projections (including their re-added GQ).

## Implementation Notes

### Key Files to Modify

| File | Change |
|------|--------|
| `scripts/data/fetch_census_gq_data.py` | **New** — Fetch 2020 Census DHC PCO1 data via API, process to parquet |
| `cohort_projections/data/load/base_population_loader.py` | Modify `load_base_population_for_county()` to subtract GQ from base pop |
| `cohort_projections/core/cohort_component.py` | Add GQ re-addition step to output (after projection completes) |
| `config/projection_config.yaml` | Add `group_quarters` config section under `base_population` |
| `data/processed/gq_county_age_sex_2020.parquet` | **New** — Processed GQ age-sex data for all 53 counties |

### Data Pipeline

```
1. Fetch 2020 DHC PCO1 data → data/raw/population/census_2020_dhc_gq_county.csv
2. Process: Scale 2020 age-sex profile to 2025 stcoreview GQ totals
3. Sprague interpolation: 5-year age groups → single years
4. Race distribution: Apply county race proportions
5. Output: data/processed/gq_county_age_sex_2025.parquet
   (columns: county_fips, age, sex, race, gq_population)
```

### Projection Pipeline

```
1. Load total base population (existing)
2. Load GQ estimate (new)
3. Subtract: household_pop = total_pop - gq_pop
4. Project household_pop through cohort-component engine (existing)
5. Add GQ back: final_pop = projected_hh_pop + gq_2025 (new)
6. Output final_pop (existing format, now includes constant GQ)
```

### Testing Strategy

1. **GQ data fetch**: Verify PCO1 data loads correctly for all 53 counties
2. **GQ scaling**: Verify scaled 2025 GQ totals match stcoreview within rounding
3. **Base population**: Verify `total = household + gq` identity at every county × age × sex × race cell
4. **Projection output**: Verify GQ is added back correctly (check that 2025 output matches original 2025 base)
5. **GQ constancy**: Verify that the GQ component is identical at all projection years (2025-2055)
6. **Scenario ordering**: Verify restricted < baseline < high still holds at all cells
7. **Regression**: All existing 1,257 tests pass

### Pipeline Rerun Required

1. **Step 00 (new)**: Fetch and process Census DHC GQ data
2. **Step 02**: Projections (with GQ separation)
3. **Step 03**: Exports

## References

1. **Ward & Grand Forks Institutional Population Review** (2026-02-23): [Review document](../../reviews/2026-02-23-ward-grand-forks-institutional-population-review.md) — identifies the root cause and institutional population dynamics
2. **Census 2020 DHC PCO1**: Group quarters population by sex by age, county level
3. **Census 2020 DHC PCO8**: College/university student housing by sex by age, county level
4. **Census 2020 DHC PCO9**: Military quarters by sex by age, county level
5. **PEP stcoreview V2025**: `data/raw/population/stcoreview_v2025_nd_parsed.parquet` — annual HHpop and GQpop totals
6. **North Carolina OSBM Projections**: [Technical Documentation](https://www.osbm.nc.gov/demog/projections-technical-doc/download) — hold-constant GQ approach
7. **New Hampshire Population Projections 2020-2050**: [Methodology Report](https://www.nheconomy.com/) — remove-and-correct GQ approach

## Implementation Results (2026-02-23)

### Files Modified/Created

| File | Change |
|------|--------|
| `scripts/data/fetch_census_gq_data.py` | **New** -- Builds GQ by county x 5-year age group x sex from PEP stcoreview |
| `data/processed/gq_county_age_sex_2025.parquet` | **New** -- 1,908 rows (53 counties x 18 age groups x 2 sexes) |
| `config/projection_config.yaml` | Added `base_population.group_quarters` config section |
| `cohort_projections/data/load/base_population_loader.py` | Added GQ separation functions, modified `load_base_population_for_county()` |
| `cohort_projections/data/load/__init__.py` | Exported GQ access functions |
| `cohort_projections/geographic/multi_geography.py` | Added GQ re-addition after projection in `run_single_geography_projection()` |

### Verification Results

Identity check (total = household + GQ) at 4 key counties:

| County | Total Pop | Household | GQ | Identity |
|--------|----------:|----------:|---:|:--------:|
| Ward (38101) | 68,233 | 66,097 | 2,136 | OK |
| Grand Forks (38035) | 74,501 | 70,035 | 4,466 | OK |
| Cass (38017) | 201,794 | 193,187 | 8,607 | OK |
| Burleigh (38015) | 103,251 | 98,040 | 5,211 | OK |

**Note:** GQ actuals are slightly less than stcoreview totals (e.g., Ward: 2,136 vs 2,161) because GQ is capped at total population in each age-sex-race cell to prevent negative household populations. The race distribution approximation causes some cells where GQ exceeds total pop.

### Test Results

All 1,257 existing tests pass. No regressions introduced.

### Projection Results (Post-Implementation)

Full pipeline re-run with GQ separation enabled:

| Scenario | 2025 | 2035 | 2045 | 2055 | 30yr Change |
|----------|------|------|------|------|-------------|
| Baseline | 799,358 | 833,780 | 874,756 | 894,362 | +11.9% |
| High Growth | 799,358 | 885,375 | 975,893 | 1,054,062 | +31.9% |
| Restricted | 799,358 | 800,867 | 831,418 | 839,002 | +5.0% |

**Comparison with pre-ADR-055 results (state-level baseline):**

| Metric | Pre-ADR-055 | Post-ADR-055 | Change |
|--------|:-----------:|:------------:|:------:|
| Baseline 2055 | 900,971 (+12.7%) | 894,362 (+11.9%) | -0.8pp |
| High Growth 2055 | 1,067,814 (+33.6%) | 1,054,062 (+31.9%) | -1.7pp |
| Restricted 2055 | 842,885 (+5.4%) | 839,002 (+5.0%) | -0.4pp |

**Key county impacts:**

| County | Before | After | Change | Explanation |
|--------|:------:|:-----:|:------:|-------------|
| Cass | +29.8% | +27.1% | -2.7pp | 8,607 GQ held constant instead of growing with positive migration |
| Burleigh | +20.0% | +17.5% | -2.5pp | 5,211 GQ held constant |
| Ward | -14.6% | -14.8% | -0.2pp | GQ held constant but HH base is smaller; near-neutral net effect |
| Grand Forks | -8.7% | -9.0% | -0.3pp | Same mechanism as Ward |

**Analysis of Phase 1 impact:**

Phase 1 (hold-constant GQ) achieves the structural goal: institutional populations no longer erode under negative migration. However, the bottom-line impact on total projections is modest because:

1. **Growing counties see reduced growth** (correctly): GQ doesn't compound with positive migration, producing 2-3pp less growth for Cass and Burleigh.
2. **Declining counties see near-neutral effect**: GQ is held constant (good), but the household base is smaller, so fewer births are produced by the smaller female population. The two effects approximately cancel.
3. **The migration rates themselves are unchanged** in Phase 1 — they still embed GQ rotation from the residual computation. Phase 2 addresses this by recomputing rates on household-only population.

The primary value of Phase 1 is structural correctness: dorm beds stay filled, barracks stay staffed, and nursing facility capacity is maintained in the projections.

## Implementation Results — Phase 2 (2026-02-26)

### Files Modified/Created

| File | Change |
|------|--------|
| `scripts/data/fetch_census_gq_data.py` | Added `build_historical_gq()` function, multi-year GQ generation |
| `data/processed/gq_county_age_sex_historical.parquet` | **New** — 11,448 rows (53 counties × 18 age groups × 2 sexes × 6 years) |
| `cohort_projections/data/process/residual_migration.py` | Added `subtract_gq_from_populations()`, wired into Step 1b |
| `config/projection_config.yaml` | Added `residual.gq_correction` section |

### GQ Data Used

| Year | Source | State GQ Total |
|------|--------|:--------------:|
| 2000 | Backward constant from 2020 | 26,223 |
| 2005 | Backward constant from 2020 | 26,223 |
| 2010 | Backward constant from 2020 | 26,223 |
| 2015 | Backward constant from 2020 | 26,223 |
| 2020 | Stcoreview period="2020" | 26,223 |
| 2024 | Stcoreview period="2024" | 30,884 |

Key GQ changes 2020→2024 by county:

| County | GQ 2020 | GQ 2024 | Change | Driver |
|--------|--------:|--------:|-------:|--------|
| Cass | 5,756 | 8,685 | +2,929 (+50.9%) | NDSU dorm construction |
| Burleigh | 4,191 | 5,272 | +1,081 (+25.8%) | U of Mary/BSC + correctional |
| Grand Forks | 4,227 | 4,516 | +289 (+6.8%) | Modest UND growth |
| Ward | 1,957 | 2,163 | +206 (+10.5%) | Minot State + MAFB |

### Test Results

All 1,257 existing tests pass. No regressions.

### Projection Results (Post-Phase 2)

Full pipeline re-run with GQ-corrected migration rates:

| Scenario | 2025 | 2035 | 2045 | 2055 | 30yr Change |
|----------|------|------|------|------|-------------|
| Baseline | 799,358 | 828,536 | 865,137 | 882,146 | +10.4% |
| High Growth | 799,358 | 893,159 | 990,561 | 1,078,346 | +34.9% |
| Restricted | 799,358 | 797,694 | 823,275 | 828,470 | +3.6% |

**Comparison across all phases (state-level baseline):**

| Phase | 2055 Pop | 30yr Growth | vs Phase 1 |
|-------|:--------:|:-----------:|:----------:|
| Pre-ADR-055 | 900,971 | +12.7% | — |
| Phase 1 (hold-constant GQ) | 894,362 | +11.9% | — |
| **Phase 2 (GQ-corrected rates)** | **882,146** | **+10.4%** | **-1.5pp** |

**Key county impacts (baseline, Phase 1 → Phase 2):**

| County | Phase 1 | Phase 2 | Change | Explanation |
|--------|:-------:|:-------:|:------:|-------------|
| Cass | +27.1% | +25.2% | -1.9pp | NDSU GQ growth (+2,929) excluded from recent HH migration |
| Burleigh | +17.5% | +18.1% | +0.6pp | Modest rebalancing of convergence rates |
| Ward | -14.8% | -15.5% | -0.7pp | Smaller HH denominator slightly amplifies negative rates |
| Grand Forks | -9.0% | -11.9% | -2.9pp | GQ growth excluded from recent migration + smoothing interaction |

### Analysis of Phase 2 Impact

Phase 2 produces **more conservative projections** across all scenarios. The dominant mechanism:

1. **2020-2024 GQ growth drives the main effect.** State GQ grew from 26,223 to 30,884 (+4,661, +17.8%) between 2020 and 2024, primarily at Cass (+2,929 from NDSU dorm construction). When computing household-only migration for 2020-2024, this GQ growth is excluded from in-migration, making the "recent" period's household migration look weaker. Since the convergence pipeline uses the 2020-2024 period as the "recent" window (starting point of convergence), this more negative starting point cascades through the projection.

2. **Grand Forks 25-29 rates improved modestly** (Male: -0.110 → -0.105; Female: -0.128 → -0.122), confirming that some GQ rotation was embedded in the old rates. However, most of the 25-29 out-migration is real brain drain, not institutional rotation. The overall projection worsened because the statewide averages used in college-age smoothing (ADR-049) became less favorable when Cass's GQ-inflated in-migration was excluded.

3. **Pre-2020 periods with backward-constant GQ**: For periods 2000-2005 through 2015-2020, the same GQ is subtracted from both start and end populations. The migration numerator changes by approximately `-GQ × (1 - survival_rate)` (GQ mortality removed from apparent migration), and the denominator shrinks. The net effect is a modest amplification of existing rate magnitudes.

4. **Scenario ordering is preserved**: restricted ≤ baseline ≤ high at all 53 counties for all projection years (2035, 2045, 2055).

### Key Structural Improvements

Despite producing more conservative bottom-line numbers, Phase 2 achieves important structural corrections:

- **Migration rates are computed on the population that actually migrates.** Household migration rates no longer include institutional rotation signals.
- **Recent-period GQ growth (especially NDSU dorm construction) is correctly identified as institutional expansion**, not demographic in-migration. This prevents dorm construction from inflating future household migration projections.
- **The pipeline is now fully GQ-aware end-to-end**: GQ separated from base population (Phase 1), migration rates computed on HH-only population (Phase 2), GQ re-added as constant in output.

## Revision History

- **2026-03-04**: Amended by ADR-061 — added `fraction` parameter (0.0-1.0) to `subtract_gq_from_populations()` allowing calibration of Phase 2 GQ correction intensity
- **2026-02-26**: Phase 2 implemented — GQ-corrected migration rates (subtract GQ from population snapshots before residual computation)
- **2026-02-23**: Accepted — Phase 1 implementation complete (GQ separation + hold-constant re-addition)
- **2026-02-23**: Initial version (Proposed) — root cause analysis, phased implementation plan

## Related ADRs

- **ADR-049**: College-Age Smoothing — treats the arrival-leg symptom of GQ distortion; GQ separation addresses the root cause
- **ADR-052**: Ward County High-Growth Floor — treats the scenario-level symptom; GQ separation prevents baseline GQ erosion
- **ADR-043**: Migration Rate Cap — safety net for extreme rates, including GQ-inflated rates
- **ADR-045**: Reservation County PEP Recalibration — analogous county-specific adjustment for a different data quality issue
- **ADR-047**: County-Specific Distributions — the base population distributions that GQ will be subtracted from
- **ADR-048**: Single-Year-of-Age Base Population — Sprague interpolation used for GQ age distribution
- **ADR-054**: State-County Aggregation — bottom-up state compatible with per-county GQ separation

## Related Reviews

- **[College Fix Research Implications](../../reviews/2026-03-04-college-fix-research-implications.md)** — Census Bureau "College Fix" methodology suggests enrollment-based partition may be more appropriate than GQ-based partition; challenges Phase 2's 100% GQ subtraction from historical denominators
- **[Projection Accuracy Analysis](../../reviews/2026-03-04-projection-accuracy-analysis.md)** — identifies Phase 2 GQ correction as the 3rd-largest sensitivity factor (37,084-person impact on 2050 projection)
