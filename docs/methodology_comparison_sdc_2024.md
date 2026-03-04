# Methodology Comparison: ND State Data Center 2024 vs. Our Project

*Updated March 2026. Previous version (December 2025) is obsolete; the model has undergone
extensive methodological development since that time.*

## Summary

Both the North Dakota State Data Center (SDC) 2024 projections and our project use the
**cohort-component method** -- the demographic gold standard. Since December 2025, our model
has been substantially rebuilt: migration was switched from IRS county-to-county flows to
**residual migration from Census PEP** (the same conceptual method the SDC uses), the scenario
structure was refined from four to three variants, race/ethnicity detail was added, group
quarters were separated, and dozens of targeted adjustments were implemented. As a result, the
two sets of projections now tell a broadly **similar story of continued state growth**, though
they differ in magnitude and in county-level trajectories.

| Dimension | SDC 2024 | Our Project (March 2026) |
|-----------|----------|--------------------------|
| State 2025 | 796,989 | 799,358 |
| State 2045 | 925,101 | 865,137 |
| State 2050 | 957,194 | 874,473 |
| Direction | Growth (+20.1% by 2050) | Growth (+9.4% by 2050) |
| Scenarios | Single projection | 3 (Baseline, High Growth, Restricted) |

The gap has narrowed dramatically compared to the December 2025 comparison, where the two
models diverged by ~170,000 people by 2045. The current baseline gap at 2045 is **~60,000**
-- still meaningful, but far more aligned than before.

---

## 1. Where the Methodologies Align

### 1.1 Core Method: Cohort-Component Projection

Both use the same fundamental demographic framework:

- Age cohorts forward through time
- Apply survival rates (mortality)
- Add births (fertility applied to reproductive-age females)
- Add/subtract net migration

This is the industry-standard method used by the Census Bureau, UN Population Division, and
state demographic offices worldwide.

### 1.2 Base Population Source

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Base year | Census 2020 | Census PEP Vintage 2025 (base year 2025) |
| Geography | State + 53 counties | State + 53 counties + 406+ places |

Both anchor to Census counts, though we use the most recent PEP estimates as our launch point
while the SDC starts from the 2020 decennial.

### 1.3 Survival Rates from CDC Life Tables

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Source | CDC life tables for ND, 2020 | ND-adjusted CDC life tables (ADR-053) |
| Application | By age and sex | By age, sex, and race |
| ND e0 (Male) | 74.2 | 75.37 |
| ND e0 (Female) | 80.0 | 80.76 |

Both use official CDC/NCHS life tables as the mortality foundation. Our values are slightly
higher due to use of a different vintage and ND-specific adjustment methodology.

### 1.4 Fertility Data Sources

Both incorporate:

- ND DHHS Vital Statistics (state-specific rates)
- CDC NVSS (national rates for blending/comparison)

The SDC uses 2018-2022 blended county rates; we use ND-adjusted CDC WONDER rates
(ADR-053) with a TFR of 1.863 (vs. national 1.621, ratio 1.15x).

### 1.5 Migration Estimation: Residual Method

Both now calculate migration as a residual:

```
Net Migration = Actual Population Change - Natural Increase (Births - Deaths)
```

This is the most important alignment change since the December 2025 comparison, where
our model used IRS county-to-county flows. We now use Census PEP component data for
2000-2024 across five historical periods (ADR-036).

### 1.6 Multi-Year Averaging for Stability

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Fertility | 2018-2022 (blended) | Constant (baseline), +/-5% (scenarios) |
| Migration | 4 five-year periods (2000-2020) | 5 five-year periods (2000-2024), BEBR averaging (ADR-036) |

Both recognize that single-period rates are too volatile for projections.

### 1.7 Bakken / Oil-County Migration Dampening

Both models explicitly dampen oil-boom-era migration for Bakken counties:

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Counties affected | "Bakken region counties" | Williams, McKenzie, Mountrail, Dunn, Stark |
| Dampening level | ~60% of 2000-2020 historical | 40-50% period-specific factors (ADR-040/051) |
| Rationale | Boom "unlikely to occur again" | Oil-boom migration was structurally atypical |

Both models agree that the 2005-2015 Bakken oil boom produced migration patterns that
should not be projected forward at face value.

---

## 2. Where the Methodologies Diverge

### 2.1 Race/Ethnicity Detail

**This remains the most significant structural divergence.**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Race categories | **None** -- total population only | **6 categories** (White NH, Black NH, AIAN NH, Asian/PI NH, Two+, Hispanic) (ADR-047) |
| Cohorts projected | ~182 (91 ages x 2 sexes) | **1,092** (91 ages x 2 sexes x 6 races) |
| Fertility by race | No | Yes -- 6 categories (ADR-053) |
| Mortality by race | No | Yes -- 6 categories (ADR-053) |

**Implications:**

- Our project can project how North Dakota's demographic composition will change over time
- Our project can show differential growth by race/ethnicity (e.g., growing Hispanic
  and Asian/PI populations, AIAN trends on reservations)
- SDC cannot address these questions with their current framework
- Our approach requires substantially more data but provides richer policy-relevant outputs

### 2.2 Group Quarters Separation

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| GQ treatment | Combined with household population | **Separated** (ADR-055) |
| GQ projection | Implicit in totals | Held constant at 2025 levels (30,884 persons) |
| Rate computation | On total population | On household-only population (Phase 2) |

Our model explicitly separates group quarters (military barracks, college dorms, correctional
facilities) from the household population before computing migration residuals. This prevents
large GQ changes (e.g., NDSU dorm expansions in Cass County, +2,929 persons 2020-2024) from
distorting migration signals for the general population.

### 2.3 Migration Adjustments

Both models apply adjustments to raw migration rates, but our model has a more extensive
and documented set:

| Adjustment | SDC 2024 | Our Project |
|------------|----------|-------------|
| Bakken dampening | ~60% of historical | 40-50% period-specific (ADR-040/051) |
| Male migration dampening | "Further reduced" (undocumented factor) | 0.80 factor for 2005-2015 periods |
| College-age smoothing | "Additional adjustments" | 50/50 blend with statewide for ages 15-24 in college counties (ADR-049) |
| Reservation recalibration | Not documented | PEP-anchored for Benson, Sioux, Rolette (ADR-045) |
| Age-aware rate caps | Not documented | +/-15% ages 15-24, +/-8% all others (ADR-043) |
| Convergence schedule | Not documented | 5-10-5 schedule: 5yr recent-to-medium, 10yr hold, 5yr to long-term |
| Ward County floor | Not applicable | High-growth scenario prevents decline (ADR-052) |
| BEBR high-growth increment | Not applicable | Additive migration boost for high scenario (ADR-046) |
| Manual adjustments | Yes -- per "Adjustments" sheets | **None** -- fully algorithmic |

The SDC applies expert-judgment manual adjustments via spreadsheet worksheets. Our model
uses exclusively algorithmic adjustments, each documented in an Architecture Decision Record.

### 2.4 Mortality Improvement

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Improvement | None visible (constant rates) | Lee-Carter style: 0.5%/yr (all scenarios) |
| Over 30 years | No improvement | ~14% cumulative reduction in age-specific mortality |

The SDC holds survival rates constant at 2020 levels. Our model incorporates gradual
mortality improvement, consistent with long-term historical trends and the approach used
by the Social Security Administration and Census Bureau.

### 2.5 Scenario Structure

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Scenarios | Single projection | **3 scenarios** |
| Fertility variation | Fixed | Constant (baseline), +5% (high), -5% (restricted) |
| Migration variation | Fixed (dampened) | Convergence (baseline), BEBR-optimistic+floor (high), CBO immigration reduction (restricted) |
| Mortality variation | Fixed | 0.5%/yr improvement (all) |

Our three scenarios:

- **Baseline**: Trend continuation, constant fertility, convergence migration, 0.5%/yr mortality improvement
- **High Growth**: +5% fertility, BEBR-optimistic migration with Ward County floor, 0.5%/yr mortality improvement
- **Restricted Growth**: -5% fertility, CBO immigration policy reduction (additive), 0.5%/yr mortality improvement

### 2.6 Base Population Processing

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Age detail | 5-year groups (18 groups) | Single-year of age via Sprague interpolation (91 cohorts, 0-90+) (ADR-048) |
| Base year | 2020 | 2025 (PEP Vintage 2025) |
| Projection step | 5-year intervals | Annual |

Our model projects in single-year steps and single-year-of-age cohorts, avoiding the
aggregation artifacts that can arise from 5-year groupings.

### 2.7 Geographic Granularity

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Levels | State, 8 regions, 53 counties | State, 53 counties, **406+ places** |
| State derivation | Independent state projection | State = sum of counties (ADR-054) |
| Place projections | No | Yes (threshold: 500+ population) |

Our state total is derived bottom-up as the sum of 53 county projections, ensuring
internal consistency. The SDC appears to project state and counties independently.

### 2.8 Validation and Quality Assurance

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Automated tests | Not documented | **1,570 tests** (unit, integration, validation) |
| Backtesting | Not documented | Rolling-origin backtests, 34 folds (ADR-057) |
| Cross-validation | Not documented | Housing-unit method cross-validation (ADR-060) |
| Documentation | PDF report | ADRs, SOPs, methodology docs with full provenance |

---

## 3. Results Comparison

### 3.1 State-Level Projections

#### SDC 2024

| Year | Population | Change from 2020 |
|------|-----------|-------------------|
| 2020 | 779,094 | -- |
| 2025 | 796,989 | +2.3% |
| 2030 | 831,543 | +6.7% |
| 2035 | 865,397 | +11.1% |
| 2040 | 890,424 | +14.3% |
| 2045 | 925,101 | +18.7% |
| 2050 | 957,194 | +22.9% |

#### Our Project (Three Scenarios, from 2025 base)

| Year | Baseline | High Growth | Restricted |
|------|----------|-------------|------------|
| 2025 | 799,358 | 799,358 | 799,358 |
| 2030 | 809,857 | 847,464 | 784,237 |
| 2035 | 828,536 | 893,159 | 797,694 |
| 2040 | 848,252 | 941,303 | 811,960 |
| 2045 | 865,137 | 990,561 | 823,275 |
| 2050 | 874,473 | 1,034,655 | 826,944 |
| 2055 | 882,146 | 1,078,346 | 828,470 |

#### Direct Comparison (Baseline vs. SDC, Overlapping Years)

| Year | SDC 2024 | Our Baseline | Difference | Diff % |
|------|----------|--------------|------------|--------|
| 2025 | 796,989 | 799,358 | +2,369 | +0.3% |
| 2030 | 831,543 | 809,857 | -21,686 | -2.6% |
| 2035 | 865,397 | 828,536 | -36,861 | -4.3% |
| 2040 | 890,424 | 848,252 | -42,172 | -4.7% |
| 2045 | 925,101 | 865,137 | -59,964 | -6.5% |
| 2050 | 957,194 | 874,473 | -82,721 | -8.6% |

The SDC projects more vigorous growth, but both models agree on the direction: **North Dakota
will grow**. The gap at 2045 is ~60,000 -- a significant improvement from the ~170,000
divergence in the December 2025 comparison, when our model used IRS data and projected decline.

Notably, our **High Growth** scenario (990,561 in 2045) exceeds the SDC projection
(925,101 in 2045), while our **Restricted** scenario (823,275) falls below it. The SDC's
single projection falls between our Baseline and High Growth scenarios, closer to our Baseline.

### 3.2 Key County Comparisons (2025-2050)

#### Cass County (Fargo)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 194,767 | 201,794 | +7,027 |
| 2030 | 211,322 | 212,222 | +900 |
| 2035 | 227,406 | 223,450 | -3,956 |
| 2040 | 239,681 | 234,065 | -5,616 |
| 2045 | 255,799 | 242,389 | -13,410 |
| 2050 | 272,878 | 247,708 | -25,170 |

Both project strong Cass County growth, but the SDC projects faster acceleration. Our model
starts higher (201,794 vs. 194,767 in 2025, reflecting PEP Vintage 2025 data) but grows
more conservatively due to GQ-corrected migration (ADR-055 Phase 2 removed NDSU dorm
growth from the migration signal).

#### Burleigh County (Bismarck)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 100,657 | 103,251 | +2,594 |
| 2030 | 108,057 | 106,137 | -1,920 |
| 2035 | 114,646 | 109,783 | -4,863 |
| 2040 | 117,739 | 113,460 | -4,279 |
| 2045 | 123,366 | 117,041 | -6,325 |
| 2050 | 128,663 | 119,663 | -9,000 |

Both project continued Burleigh County growth. The SDC projects steeper growth after 2030.

#### Grand Forks County

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 74,966 | 74,501 | -465 |
| 2030 | 77,443 | 72,953 | -4,490 |
| 2035 | 79,159 | 71,752 | -7,407 |
| 2040 | 80,561 | 71,064 | -9,497 |
| 2045 | 81,238 | 69,599 | -11,639 |
| 2050 | 81,582 | 67,501 | -14,081 |

**Grand Forks is the largest county-level divergence.** The SDC projects modest growth while
our model projects decline. This is driven by our college-age smoothing (ADR-049) and
GQ correction (ADR-055), which reduce the apparent migration signal in a county dominated
by UND enrollment patterns. Our model also reflects more recent PEP data (through 2024)
showing ongoing population loss.

#### Ward County (Minot)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 72,066 | 68,233 | -3,833 |
| 2030 | 74,545 | 65,988 | -8,557 |
| 2035 | 77,503 | 64,758 | -12,745 |
| 2040 | 79,852 | 63,441 | -16,411 |
| 2045 | 82,831 | 61,959 | -20,872 |
| 2050 | 85,975 | 59,420 | -26,555 |

Ward County shows the second-largest divergence. The SDC projects growth while our baseline
projects decline. This reflects different assessments of Minot's migration trajectory,
particularly the impact of Minot Air Force Base and post-2020 population trends. Our High
Growth scenario includes a Ward County floor (ADR-052) that prevents decline, bringing
it closer to the SDC's projection in that variant.

#### Williams County (Williston)

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2025 | 43,807 | 41,767 | -2,040 |
| 2030 | 46,170 | 43,129 | -3,041 |
| 2035 | 48,635 | 46,780 | -1,855 |
| 2040 | 50,953 | 51,011 | +58 |
| 2045 | 53,957 | 55,851 | +1,894 |
| 2050 | 56,047 | 60,732 | +4,685 |

Williams County is interesting: both models project growth, but our model eventually
projects **faster** growth than the SDC. This likely reflects our use of more recent
2020-2024 PEP data capturing continued Bakken-area in-migration, combined with our
convergence schedule that transitions from recent rates to medium-term rates over 5 years
before holding for 10 years and then converging to long-term rates.

---

## 4. Root Cause Analysis: Why the Projections Diverge

### 4.1 The Gap is Much Smaller Than Before

In the December 2025 comparison, the projections diverged by ~170,000 people by 2045 and
pointed in opposite directions (SDC: growth, ours: decline). The current gap is ~60,000
by 2045, with both models pointing in the same direction: growth.

**What changed:**

1. **Migration method**: We switched from IRS county-to-county flows (which showed net
   out-migration for 2019-2022) to Census PEP residual migration (which shows positive
   net migration across the full 2000-2024 period after averaging). This single change
   accounts for the majority of the convergence between the two models.

2. **Base year update**: Moving from 2020 to 2025 as the base year, using PEP Vintage 2025,
   gives us a higher starting population and incorporates recent growth.

3. **Multi-period averaging**: Using five periods (2000-2024) with BEBR averaging (ADR-036)
   incorporates the positive migration signal from the early-to-mid 2000s and 2020s.

### 4.2 The Remaining ~60,000 Gap

The remaining divergence is driven by several factors:

#### Migration Rate Magnitude

Even though both models use residual migration, they produce different net migration rates:

- **SDC**: Projects net in-migration of ~4,000-6,000/year after dampening
- **Our model**: Projects more moderate net in-migration that declines over time due to
  convergence interpolation

Our convergence schedule (5-10-5) gradually moves rates from recent observed levels toward
long-term equilibrium, producing a decelerating growth trajectory. The SDC appears to hold
dampened rates constant throughout the projection horizon.

#### Mortality Improvement vs. Constant Mortality

Our 0.5%/year mortality improvement slightly increases our projected population relative
to a no-improvement scenario. However, the SDC's constant mortality (no improvement) is
offset by their higher migration assumptions, so this factor partially narrows the gap
rather than widening it.

#### GQ Correction Effect

Our GQ separation (ADR-055 Phase 2) removes institutional population changes from the
migration signal. This makes projections more conservative for counties with recent GQ
growth (especially Cass, Grand Forks) and slightly changes the statewide total.

The Phase 2 GQ correction reduced the state baseline by approximately 1.5 percentage points
compared to a model without GQ separation.

#### Base Year Difference

The SDC starts from 2020 (779,094) while we start from 2025 (799,358). The SDC must
project through 2020-2025 to reach their 2025 estimate of 796,989 -- already 2,369 below
our PEP-anchored starting point. This initial difference persists and can compound.

### 4.3 County-Level Divergence Patterns

The divergence is not uniform across counties:

- **Urban growth counties** (Cass, Burleigh): Both models agree on growth; SDC is more
  optimistic in the later years
- **College counties** (Grand Forks, Ward): Largest divergence; our college-age smoothing
  and GQ correction produce more conservative trajectories
- **Oil counties** (Williams, McKenzie): Both dampen; our model eventually projects faster
  growth due to convergence dynamics and recent PEP data
- **Rural decline counties**: Generally similar -- both project continued decline in most
  rural counties

---

## 5. Which Projection is More Realistic?

### Arguments for the SDC's More Optimistic View

- **Track record**: The SDC has decades of experience projecting ND population
- **Expert judgment**: Their manual adjustments incorporate local knowledge difficult to
  encode algorithmically
- **Economic momentum**: North Dakota's energy sector and economic development initiatives
  could sustain stronger in-migration than trend data alone suggests
- **Historical precedent**: The 2020-2024 period showed population growth after brief
  COVID-related disruption
- **Urban attraction**: Fargo, Bismarck, and other cities continue to attract workers from
  the broader region

### Arguments for Our More Moderate View

- **Newer data**: We incorporate PEP data through 2024, two years beyond the SDC's 2022
  cutoff for fertility and four years beyond their 2020 migration endpoint
- **Algorithmic consistency**: No manual adjustments means the methodology is fully
  reproducible and auditable
- **GQ correction**: Separating institutional populations prevents misleading migration
  signals (e.g., NDSU dorm construction appearing as Cass County in-migration)
- **Convergence**: The 5-10-5 schedule reflects the demographic principle that extreme
  rates tend to regress toward means over long horizons
- **Mortality improvement**: Including gradual mortality improvement is consistent with
  long-run historical trends and actuarial practice
- **Validation**: 1,570 automated tests, rolling-origin backtests, and housing-unit
  cross-validation provide quantitative evidence for model performance

### The Likely Reality

**The two models now bracket a plausible range.** With both projecting growth, the key
question is no longer "will North Dakota grow or decline?" but rather "how fast?"

- **More optimistic bound**: SDC 2024 (~925K by 2045)
- **Central estimate**: Our Baseline (~865K by 2045)
- **Conservative bound**: Our Restricted (~823K by 2045)

For most planning purposes, the 865K-925K range for 2045 represents a reasonable planning
envelope. Our High Growth scenario (991K) suggests an upper bound if energy and economic
conditions prove exceptionally favorable.

### What Would Need to Happen for Each Scenario

**For SDC's trajectory to materialize:**

- Sustained net in-migration of 4,000-6,000/year for 20+ years
- Continued strong energy sector employment
- Successful diversification of the economy beyond energy
- Remote work and quality-of-life attracting new residents

**For our Baseline trajectory to materialize:**

- Moderate positive net in-migration that gradually declines toward equilibrium
- Natural increase remaining positive but narrowing as the population ages
- No dramatic energy sector expansion or contraction
- Gradual mortality improvement consistent with national trends

---

## 6. SDC Source File Analysis

This section documents detailed findings from analysis of the SDC's actual source files
and working spreadsheets, providing insight into their specific methodological choices and
calculations.

### Data Sources Used

#### Base Population

- **Census 2020**: Used as the authoritative base population
- **Census 2010**: Used for historical comparison and migration rate calculation
- **Population Estimates Program (PEP)**: Used cc-est2019-agesex-38 for interim estimates

#### Fertility Data

- **Source**: North Dakota Vital Statistics, Department of Health
- **Time Period**: 2016-2022 (with emphasis on 2018-2022 for rate calculation)
- **Data File**: "2018-2022 ND Res Birth for Kevin Iverson.xlsx" -- prepared September 15, 2023 by Vital Records
- **Categories**: Births by county of residence, age group of mother (Under 20, 20-25, 25-29, 30-34, 35-39, 40-44, 45+)
- **Female Population**: "Average Female Count 2018 to 2022.xlsx" -- average of 2018 and 2022 female populations by age group by county
- **National Reference**: NVSS Report (nvsr72-01.pdf) -- National Vital Statistics Reports Volume 71, Number 1, dated January 31, 2023

**Fertility Rate Calculation Method:**

1. Average female population by age group (10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49) for each county from 2018-2022
2. Sum births by mother's age group for 2018-2022 period
3. Calculate 5-year fertility rate = (Total births in age group) / (Average female population in age group)
4. Rates "smoothed to reduce anomalies" -- blended with state and national rates for stability

**Key Fertility Data Points:**

- State total: ~189,000 to ~197,000 females of childbearing age (10-49)
- Birth data includes suppression ("NR") for small cells for privacy
- 2018-2022 total births by year: 10,630 (2018), 10,447 (2019), 10,051 (2020), 10,111 (2021), 9,557 (2022)
- Sex ratio at birth: ~104.8 males per 100 females (51.2% male, 48.8% female)

#### Survival/Mortality Data

- **Source**: CDC Life Tables for North Dakota, 2020
- **Publication**: NVSS Report nvsr71-02
- **Files**: ND1.xlsx (Total), ND2.xlsx (Males), ND3.xlsx (Females), ND4.xlsx (Standard Errors)

**Life Table Structure (Standard life table columns):**

| Column | Description |
|--------|-------------|
| qx | Probability of dying between ages x and x+1 |
| lx | Number surviving to age x (from radix of 100,000) |
| dx | Number dying between ages x and x+1 |
| Lx | Person-years lived between ages x and x+1 |
| Tx | Total person-years lived above age x |
| ex | Expectation of life at age x |

**5-Year Survival Rates (from Projections_Base_2023.xlsx):**

| Age Group | Male | Female |
|-----------|------|--------|
| Under 5 | 0.9915 | 0.9946 |
| 5-9 | 0.9994 | 0.9994 |
| 10-14 | 0.9987 | 0.9994 |
| 15-19 | 0.9982 | 0.9983 |
| 20-24 | 0.9949 | 0.9980 |
| 25-29 | 0.9927 | 0.9972 |
| 30-34 | 0.9897 | 0.9961 |
| 35-39 | 0.9876 | 0.9950 |
| 40-44 | 0.9860 | 0.9936 |
| 45-49 | 0.9808 | 0.9914 |
| 50-54 | 0.9776 | 0.9878 |
| 55-59 | 0.9652 | 0.9817 |
| 60-64 | 0.9521 | 0.9725 |
| 65-69 | 0.9335 | 0.9593 |
| 70-74 | 0.8972 | 0.9405 |
| 75-79 | 0.8355 | 0.9092 |
| 80-84 | 0.7353 | 0.8525 |
| 85+ | 0.5428 (calculated) | 0.6998 (calculated) |

**Life Expectancy at Birth (2020):**

- Total: 76.9 years
- Males: 74.2 years
- Females: 80.0 years

**85+ Survival Rate Calculation (from spreadsheet):**

The SDC calculated the 85+ open-ended age group survival rate using:
```
85+ Survival Rate = (85+ survivors at t+5) / (90+ survivors at t+5) = 305,579 / 121,647 = 0.398 (approximate)
```
Note: They used 0.5428 for males and 0.6998 for females in actual projections.

#### Migration Data

- **Source Files**:
  - "Mig Rate 2000-2020_final.xlsx" -- Final averaged migration rates
  - "Mig Rates 2000-2020.xlsx" -- Working calculations
- **Time Periods Analyzed**:
  - 2000-2005
  - 2005-2010
  - 2010-2015
  - 2015-2020
- **Method**: Residual method (calculated as difference between actual and expected population)

### Migration Rate Calculation Methodology

The SDC calculated migration rates using the following detailed process:

#### Step 1: Calculate Expected Population (No Migration)

For each 5-year period, calculate what the population would be with zero migration:
```
Expected_Pop[t+5] = Pop[t] * Survival_Rate + Births_to_Cohort
```

#### Step 2: Calculate Migration Residual

```
Migration_Rate = (Actual_Pop[t+5] - Expected_Pop[t+5]) / Pop[t]
```

#### Step 3: Average Across Four Periods

The final migration rates used are averages of the four 5-year periods (2000-2005, 2005-2010, 2010-2015, 2015-2020).

**Example Migration Rates by Age Group (State Level, Averaged 2000-2020):**

| Age Group | Male Rate | Female Rate |
|-----------|-----------|-------------|
| Under 5 | +0.108 | -0.003 |
| 5-9 | +0.049 | +0.011 |
| 10-14 | +0.054 | +0.017 |
| 15-19 | +0.173 | +0.081 |
| 20-24 | +0.328 | +0.117 |
| 25-29 | -0.119 | -0.243 |
| 30-34 | +0.059 | +0.083 |
| 35-39 | +0.039 | -0.022 |
| 40-44 | +0.036 | +0.004 |
| 45-49 | +0.032 | -0.001 |
| 50-54 | +0.042 | +0.005 |
| 55-59 | +0.024 | -0.008 |
| 60-64 | -0.007 | -0.011 |
| 65-69 | -0.049 | -0.023 |
| 70-74 | -0.087 | -0.033 |
| 75-79 | -0.100 | -0.085 |
| 80-84 | -0.160 | -0.132 |
| 85+ | -0.148 | -0.089 |

**Key Migration Patterns Identified:**

- **Strong in-migration**: Ages 20-24 (college/workforce entry), males especially
- **Out-migration**: Ages 25-29 for females, 65+ for both sexes
- **Net male migration**: 0.034 (males), -0.019 (females) -- significant gender imbalance

#### Migration Rate Adjustments (The "60% Dampening")

From the methodology writeup, the SDC made several critical adjustments:

1. **Bakken Dampening**: "Given the significant in-migration that North Dakota experienced from 2010 to 2020, the rates were typically reduced to about 60 percent of what was found" because the Bakken Oil Boom "is unlikely to occur again"

2. **College-Age Adjustment**: "Counties with significant college age populations typically required additional adjustments as the algorithm tends to not capture the in- and out-migration of college age residents as well as it should"

3. **Male Migration Adjustment**: "The rate of male migration was further reduced than female migration as the pattern found from 2000 to 2020 when in-migration was dominated by males is unlikely to continue into the future and would have resulted in unrealistic sex ratio in future years"

4. **Bakken Region Counties**: "The rate of migration in counties in the Bakken region that experienced significant growth during the last decade also were adjusted to a lower rate"

### Projection Workbook Structure

The main projection workbook (Projections_Base_2023.xlsx) contains 45 sheets organized as follows:

| Sheet Category | Purpose |
|----------------|---------|
| Notes | Process documentation |
| 5-Year Survival Rate By Sex | Survival rates from life tables |
| Census 2010, Census 2020 | Base population data |
| Senthetic_2015_2 | Interpolated 2015 estimates |
| Mig_Rate | Averaged migration rates by age/sex/county |
| Fer 2020-2025, Fer_2025-30, etc. | Fertility calculations per period |
| Nat_Grow 2020-2025, etc. | Natural growth (births - deaths) |
| Adjustments 2020-2025, etc. | Manual adjustment factors |
| 2020-2025 Migration, etc. | Migration applied |
| 2025 Pro, 2030 Pro, etc. | Final projections by period |

**Projection Process (from Notes sheet):**

1. Start with Census base (adjusted in 5-year increments)
2. Apply fertility rate by age of mother to get ages 0-4 population
3. Apply survival rate by age group and sex by county
4. Apply migration rate by age, sex, and county
5. Apply manual adjustments for "unexpected patterns of natural growth"
6. Output next 5-year projection

### Comparison to Our Methodology

#### Where SDC Source Files Confirm Alignment

| Aspect | SDC Source Files | Our Approach |
|--------|------------------|--------------|
| Cohort-component structure | 18 age groups x 2 sexes x 53 counties = 1,908 cells | Same structure, plus 6 race categories = 11,448 cells |
| CDC life tables | ND-specific, 2020 | ND-adjusted CDC, configurable year |
| 5-year survival rates | Calculated from single-year qx | Same approach |
| Fertility by mother's age | 7 age groups (10-14 through 45-49) | Same age groups |
| Migration method | Residual from Census | Residual from Census PEP (aligned since Feb 2026) |

#### Where SDC Source Files Reveal Key Differences

| Aspect | SDC Approach (from source files) | Our Approach |
|--------|----------------------------------|--------------|
| **Migration time period** | 2000-2020 (4 periods averaged) | 2000-2024 (5 periods, BEBR averaged) |
| **Migration adjustment** | 60% dampening + manual | Algorithmic: oil dampening, male dampening, college smoothing, rate caps, convergence |
| **GQ treatment** | Combined in totals | Separated, held constant, rates computed on HH-only |
| **Manual adjustments** | Yes -- spreadsheet "Adjustments" sheets | No manual adjustments |
| **Mortality improvement** | None visible (constant rates) | Lee-Carter style 0.5%/year |
| **Race/ethnicity** | None | 6 categories |
| **Age resolution** | 5-year groups | Single-year of age |
| **Projection step** | 5-year | Annual |

### Key Formulas Extracted

**Natural Growth Calculation:**
```
Natural_Growth[county, age, sex] = Population[t] * Survival_Rate[age, sex] + Births[county, age_mother]
```

**Migration Application:**
```
Population[t+5] = Natural_Growth * (1 + Migration_Rate[county, age, sex])
```

**85+ Survival (Open-ended):**
```
85+_Survivors[t+5] = 85+_Pop[t] * 0.5428 (males) or 0.6998 (females)
```

### Data Quality Observations

From examining the source files:

1. **Small cell suppression**: Birth data uses "NR" (Not Reported) for privacy in small counties
2. **Rounding**: Some intermediate calculations appear rounded
3. **Interpolation**: 2015 population was "synthetic" (interpolated between 2010 and 2020)
4. **Date stamps**: Files dated December 2023 through January 2024, final methodology dated March 7, 2024

---

## 7. Summary Table

| Dimension | SDC 2024 | Our Project (March 2026) | Assessment |
|-----------|----------|--------------------------|------------|
| Core method | Cohort-component | Cohort-component | **Aligned** |
| Base year | Census 2020 | PEP Vintage 2025 | **Divergent** -- Ours more recent |
| Age resolution | 5-year groups | Single-year of age (ADR-048) | **Divergent** -- Ours finer |
| Projection step | 5-year | Annual | **Divergent** -- Ours finer |
| Race/ethnicity | None | 6 categories (ADR-047) | **Divergent** -- Ours more detailed |
| Geography | State + counties | State + counties + 406+ places | **Divergent** -- Ours more granular |
| State derivation | Independent | Bottom-up sum of counties (ADR-054) | **Divergent** -- Ours internally consistent |
| Migration method | Residual (Census) | Residual (Census PEP) | **Aligned** (since Feb 2026) |
| Migration periods | 2000-2020 (4 periods) | 2000-2024 (5 periods) | **Partially aligned** |
| Bakken dampening | ~60% | 40-50% period-specific (ADR-040/051) | **Aligned** in concept |
| Male migration dampening | Yes (undocumented factor) | 0.80 for 2005-2015 | **Aligned** in concept |
| College-age adjustment | Yes (undocumented) | 50/50 statewide blend, ages 15-24 (ADR-049) | **Aligned** in concept |
| GQ separation | None | Separated + held constant (ADR-055) | **Divergent** -- Ours more rigorous |
| Reservation calibration | Not documented | PEP-anchored (ADR-045) | **Divergent** |
| Migration convergence | Not documented | 5-10-5 schedule | **Divergent** |
| Age-aware rate caps | Not documented | +/-15% (15-24), +/-8% (others) (ADR-043) | **Divergent** |
| Fertility source | ND DHHS + NVSS blended | ND-adjusted CDC WONDER (ADR-053) | **Partially aligned** |
| Fertility by race | No | Yes (6 categories) | **Divergent** |
| Mortality improvement | None (constant) | 0.5%/yr Lee-Carter style | **Divergent** |
| Mortality by race | No | Yes (6 categories) | **Divergent** |
| Manual adjustments | Yes | None (fully algorithmic) | **Divergent** |
| Scenarios | 1 (single projection) | 3 (Baseline, High, Restricted) | **Divergent** |
| Automated tests | Not documented | 1,570 tests | **Divergent** |
| Backtesting | Not documented | Rolling-origin, 34 folds (ADR-057) | **Divergent** |
| Cross-validation | Not documented | Housing-unit method (ADR-060) | **Divergent** |
| Documentation | PDF report | ADRs + SOPs + metadata | **Divergent** |
| **State 2045** | **925,101** | **865,137 (Baseline)** | **~7% gap** |
| **State 2050** | **957,194** | **874,473 (Baseline)** | **~9% gap** |
| **Direction** | **Growth** | **Growth** | **Aligned** |

---

## 8. Recommendations

### 8.1 Present Both Projections as a Range

When presenting results to stakeholders, show both sets of projections as a plausible range:

- **Conservative bound**: Our Restricted Growth scenario (~828K by 2050)
- **Central estimate**: Our Baseline (~874K by 2050)
- **Optimistic bound**: SDC 2024 (~957K by 2050)
- **Aggressive bound**: Our High Growth (~1,035K by 2050)

This four-point range communicates uncertainty honestly and lets planners calibrate to
their risk tolerance.

### 8.2 Investigate County-Level Divergences

The largest divergences (Grand Forks, Ward) merit further investigation:

- Compare both models against actual Census 2025 data as it becomes available
- Evaluate whether our GQ correction and college-age smoothing are over-dampening migration
  in these counties, or whether the SDC's manual adjustments are over-compensating
- Consider whether Ward County's Minot AFB population dynamics warrant a specialized
  treatment similar to our reservation county calibration (ADR-045)

### 8.3 Leverage Our Unique Capabilities

Our model offers capabilities the SDC does not:

- **Race/ethnicity projections**: Critical for workforce planning, healthcare, education
- **Place-level projections**: City and town population forecasts for local planning
- **Scenario analysis**: Quantified uncertainty bounds for infrastructure investment decisions
- **Annual resolution**: Useful for year-by-year budget planning
- **Automated validation**: Quantifiable confidence in model performance

### 8.4 Monitor Actual Data for Model Calibration

As new Census PEP vintages become available:

- Compare both projections against actuals to assess which migration assumptions are
  proving more accurate
- If net migration consistently exceeds our Baseline assumption, consider adjusting
  the convergence schedule or dampening factors
- If net migration falls below the SDC's assumption, their projection will increasingly
  overshoot

### 8.5 Document the Convergence

The dramatic convergence between the two models -- from a 170K gap in December 2025 to a
60K gap in March 2026 -- is itself a significant finding. It demonstrates that the primary
driver of the previous divergence was the migration data source (IRS flows vs. Census
residuals), not fundamental methodological disagreements. Both models, when given similar
migration inputs, produce broadly similar trajectories, with differences attributable to
specific adjustment choices rather than structural incompatibilities.

### 8.6 Consider Future Harmonization

Given the alignment on core methodology, there may be opportunities for constructive
dialogue with the SDC:

- Share our GQ correction approach, which could improve their college-county projections
- Discuss their manual adjustment rationale, which could inform our algorithmic design
- Compare detailed age-specific migration rates to understand where the remaining
  differences arise
- Explore whether a joint or reconciled projection product would serve stakeholders better
  than two competing sets

---

*Last updated: 2026-03-02*
*Model version: 2.4.0*
*Previous version: December 2025 (obsolete -- model used IRS migration, projected state decline)*
