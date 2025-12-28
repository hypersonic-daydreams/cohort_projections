# Methodology Comparison: ND State Data Center 2024 vs. Our Project

## Summary

Both approaches use the **cohort-component method** — the demographic gold standard. However, there are significant differences in demographic detail, data sources, and methodological nuances.

---

## Where the Methodologies Align

### 1. Core Method: Cohort-Component Projection
Both use the same fundamental demographic approach:
- Age cohorts forward through time
- Apply survival rates (mortality)
- Add births (fertility applied to reproductive-age females)
- Add/subtract net migration

This is the industry-standard method used by Census Bureau, UN Population Division, and state demographic offices.

### 2. Base Population Source
| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Year | Census 2020 | Census 2020 (base year 2025 uses PEP estimates) |
| Geography | State + 53 counties | State + 53 counties + places |

Both anchor to Census 2020 as the authoritative population count.

### 3. 5-Year Age Groups for Rates
Both use **5-year age groups** for applying demographic rates, which is standard practice due to data availability and statistical stability.

### 4. Survival Rates from CDC Life Tables
| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Source | CDC life tables for ND, 2020 | SEER/CDC life tables, 2020 |
| Application | By age and sex | By age, sex, and race |

Both use official CDC life tables as the mortality foundation.

### 5. Fertility Data Sources
Both incorporate:
- ND DHHS Vital Statistics (state-specific rates)
- CDC NVSS (national rates for blending/comparison)

### 6. Migration Estimation: Residual Method
Both calculate migration as a residual:
```
Net Migration = Actual Population Change - Natural Increase (Births - Deaths)
```

The SDC uses 2000-2020 historical residuals; we use IRS county-to-county flows with similar averaging logic.

### 7. Multi-Year Averaging for Stability
| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Fertility | 2016-2022 (blended) | 5-year average (default) |
| Migration | 4 five-year periods averaged | 5-year average |

Both recognize that single-year rates are too volatile for projections.

### 8. Projection Horizon
Both project through **2045** (SDC extends to 2050), using annual or 5-year intervals.

---

## Where the Methodologies Diverge

### 1. **Demographic Detail: Race/Ethnicity**

**This is the most significant divergence.**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Race categories | **None** - total population only | **6 categories** (White NH, Black NH, AIAN NH, Asian/PI NH, Two+, Hispanic) |
| Cohorts projected | ~182 (91 ages x 2 sexes) | **1,092** (91 ages x 2 sexes x 6 races) |

**Implications:**
- Our project can project demographic diversity changes over time
- Our project can show differential growth by race/ethnicity
- SDC cannot track increasing Hispanic population or AIAN trends
- Our approach requires more data but provides richer policy-relevant outputs

### 2. **Geographic Granularity**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Levels | State, 8 regions, 53 counties | State, 53 counties, **406 places** |
| Place projections | No | Yes (threshold: 500+ population) |

Our project adds sub-county place-level projections, critical for:
- City planning
- School district enrollment forecasting
- Service delivery planning

### 3. **Migration Rate Adjustments**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Historical period | 2000-2020 (4 periods) | 2018-2022 (recent average) |
| Bakken adjustment | Rates reduced to **~60%** of historical | No explicit Bakken adjustment |
| College-age adjustment | Special treatment noted | Handled via age-pattern distribution |
| Male migration adjustment | Extra adjustment noted | 50/50 sex distribution (configurable) |

**SDC's Bakken Adjustment**: They explicitly dampen migration rates because the 2005-2015 oil boom was "atypical" — they don't expect that level of in-migration to continue.

**Our Approach**: Uses recent average (2018-2022) which already reflects post-boom normalization. We could add a similar scenario adjustment if desired.

### 4. **Age Pattern for Migration**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Method | Not specified in documentation | Simplified age multipliers or Rogers-Castro model |
| Age distribution | Implicit | Explicit 9-category multipliers (0-9: 0.3 -> 20-29: 1.0 -> 75+: 0.10) |

Our project has explicit, documented, and configurable age patterns for distributing aggregate migration to cohorts.

### 5. **Fertility Rate Processing**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Smoothing | "Rates smoothed to reduce anomalies" | Zero-fill for missing, validation thresholds |
| Blending | State + national rates blended | Single-source with multi-year averaging |
| By race | No | Yes (6 categories) |

SDC explicitly smooths and blends rates; we rely on multi-year averaging and validation rather than explicit smoothing.

### 6. **Mortality Improvement**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Approach | Not specified | Lee-Carter style: 0.5% annual improvement |
| Configurability | Unknown | Configurable factor (0-1% range) |

Our project has explicit mortality improvement built into the engine; SDC methodology doesn't specify if they project mortality improvements.

### 7. **Scenario Support**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Scenarios | Single projection (implicitly baseline) | **4 built-in scenarios**: Baseline, High Growth, Low Growth, Zero Migration |
| Sensitivity analysis | Not published | Fertility +/-10%, Migration +/-25% |

Our project has explicit scenario infrastructure for sensitivity analysis and uncertainty communication.

### 8. **Data Sources for Migration**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Domestic | Residual from Census | IRS county-to-county flows |
| International | Included in residual | Separate ACS/Census component |
| Separation | Combined | Separate domestic + international, then summed |

Our approach uses IRS administrative data for domestic migration, providing more granular origin-destination information. SDC uses demographic residual method.

### 9. **Open-Ended Age Group (90+)**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Treatment | Not specified | Explicit formula using Tx/Lx from life tables |
| Survival rate | Unknown | ~0.60-0.70 (calculated or default) |

Our project has explicit handling of the 90+ open-ended group with documented methodology.

### 10. **Validation and Quality Assurance**

| Aspect | SDC 2024 | Our Project |
|--------|----------|-------------|
| Validation | Manual review, comparison to 2018 projections | Multi-level automated validation |
| Plausibility checks | Expert judgment | Age-specific thresholds, TFR ranges, life expectancy |
| Metadata | PDF report | JSON metadata with full provenance |

Our project has extensive automated validation; SDC relies on expert review.

---

## Results Comparison

### SDC 2024 Projections

| Year | SDC State Population |
|------|---------------------|
| 2020 | 779,094 |
| 2025 | 796,989 (+2.3%) |
| 2030 | 831,543 (+6.7%) |
| 2035 | 865,397 (+11.1%) |
| 2040 | 890,424 (+14.3%) |
| 2045 | 925,101 (+18.7%) |

**Key findings they emphasize:**
- ~75% of 2010-2020 growth was from migration
- Cass County will have ~30% of state population by 2050
- Rural counties continue to decline
- Aging population through 2035

### Our Baseline Projections

| Year | Our State Population |
|------|---------------------|
| 2026 | 795,067 |
| 2030 | 789,516 (-0.7%) |
| 2035 | 782,649 (-1.6%) |
| 2040 | 771,446 (-3.0%) |
| 2045 | 754,882 (-5.1%) |

### Direct Comparison

| Year | SDC 2024 | Our Baseline | Difference |
|------|----------|--------------|------------|
| 2030 | 831,543 | 789,516 | -42,027 (-5.1%) |
| 2035 | 865,397 | 782,649 | -82,748 (-9.6%) |
| 2040 | 890,424 | 771,446 | -118,978 (-13.4%) |
| 2045 | 925,101 | 754,882 | -170,219 (-18.4%) |

**The projections diverge dramatically:**

- SDC 2024: **+16.1% growth** (797K to 925K)
- Our model: **-5.1% decline** (795K to 755K)
- Gap by 2045: **~170,000 people**

---

## Root Cause Analysis: Why the Projections Diverge

### The Dominant Factor: Migration Assumptions

The divergence is almost entirely explained by different migration assumptions.

#### SDC 2024 Migration (Net In-Migration)

| Period | Net Migration | Annual Average |
|--------|---------------|----------------|
| 2020-2025 | +7,717 | +1,543/year |
| 2025-2030 | +18,154 | +3,631/year |
| 2030-2035 | +23,539 | +4,708/year |
| 2035-2040 | +23,948 | +4,790/year |
| 2040-2045 | +29,111 | +5,822/year |
| **Total** | **+102,469** | |

#### Our Migration Data (Net Out-Migration)

| Year | Net Migration |
|------|---------------|
| 2019 | -2,584 |
| 2020 | -5,307 |
| 2021 | -6,444 |
| 2022 | -2,746 |
| **Average** | **-4,270/year** |

Over 20 years at our observed rate: **-85,400 people**

**Difference in migration alone: ~188,000 people** — this explains nearly all of the divergence.

### Why the Migration Data Differs

| Factor | SDC 2024 | Our Project |
|--------|----------|-------------|
| **Time period** | 2000-2020 (includes Bakken boom) | 2019-2022 (post-boom) |
| **Adjustment** | Dampens by ~40%, still positive | Uses actual recent data |
| **Data source** | Residual method from Census | IRS county-to-county flows |
| **Implicit assumption** | Boom migration partially continues | Post-boom reality is the new normal |

### The Bakken Factor

The SDC explicitly acknowledges the Bakken oil boom (2005-2015) was "atypical" and dampens historical migration rates to ~60% of observed levels. However, even after dampening, they still project **net in-migration**.

Our data captures what actually happened in 2019-2022: **net out-migration**. The post-boom period shows people leaving North Dakota, not arriving.

### Secondary Factor: Natural Change

SDC projects positive natural increase (births > deaths) through 2045:

| Period | Natural Change |
|--------|----------------|
| 2020-2025 | +6,102 |
| 2025-2030 | +16,404 |
| 2030-2035 | +10,321 |
| 2035-2040 | +1,083 |
| 2040-2045 | +5,572 |
| 2045-2050 | -5,526 (turns negative) |

Our model, with below-replacement fertility (TFR ~1.6) and an aging population, likely produces earlier negative natural change. However, even with identical natural change assumptions, the migration difference alone accounts for most of the divergence.

---

## Which Projection is More Realistic?

### Arguments for SDC's Optimistic View

- **Historical momentum**: North Dakota grew for decades before 2020
- **Energy sector revival**: Oil prices could recover, bringing workers back
- **Remote work trends**: Lower cost-of-living states may attract remote workers
- **Economic development**: State initiatives to attract businesses and residents
- **Expert judgment**: SDC demographers applied professional judgment to dampen boom-era rates

### Arguments for Our More Pessimistic View

- **Recent data is actual**: 2019-2022 shows real out-migration, not a projection
- **Oil boom was unique**: The Bakken boom was a once-in-a-generation event
- **Structural trends**: Rural depopulation is a nationwide phenomenon
- **Youth exodus**: Young people continue leaving for opportunities in larger metros
- **Census validation**: Recent PEP estimates show ND barely growing

### The Likely Reality

**The truth is probably somewhere in between.**

- The SDC projection may be **too optimistic** by assuming boom-era migration patterns partially continue indefinitely
- Our projection may be **too pessimistic** if the 2019-2022 out-migration represents a temporary post-boom correction rather than a permanent trend

### What Would Need to Happen for Each Scenario

**For SDC to be correct:**

- Energy sector revival bringing sustained in-migration
- Remote work attracting new residents
- Successful economic diversification
- Reversal of rural depopulation trends

**For our projection to be correct:**

- Continued post-boom out-migration
- No major economic drivers to attract migrants
- Continuation of current fertility and mortality trends
- Rural counties continue losing population to urban areas (within and outside ND)

---

## Summary Table

| Dimension | SDC 2024 | Our Project | Assessment |
|-----------|----------|-------------|------------|
| Core method | Cohort-component | Cohort-component | **Aligned** |
| Race/ethnicity | None | 6 categories | **Divergent** - Ours more detailed |
| Geography | State/county | State/county/place | **Divergent** - Ours more granular |
| Migration adjustment | 60% dampening (Bakken) | Recent average | **Divergent** - Different assumptions |
| Mortality improvement | Unknown | 0.5%/year | **Unknown** alignment |
| Scenario support | Single | 4 scenarios | **Divergent** - Ours more flexible |
| Validation | Manual | Automated | **Divergent** - Ours more systematic |
| Documentation | PDF report | ADRs + metadata | **Divergent** - Ours more explicit |

---

## Recommendations

### 1. Add a "Zero Migration" or "SDC-Aligned" Scenario

Our current scenarios (baseline, high growth, low growth) all use the same underlying out-migration data, so they all project decline at different rates. Consider adding:

- **Zero net migration scenario**: Shows natural change only (births minus deaths)
- **SDC-aligned scenario**: Uses their migration assumptions (+4,000 to +6,000/year) to bracket the uncertainty

This would give users a range from pessimistic (our baseline) to optimistic (SDC-aligned).

### 2. Document the Migration Assumption Explicitly

The migration assumption is the single most important driver of the projection. Users should understand:

- Our baseline uses 2019-2022 IRS data showing net out-migration
- This represents the post-Bakken reality
- Alternative assumptions would produce dramatically different results

### 3. Monitor Actual Migration Data

As new IRS and Census data becomes available, compare to our assumptions:

- If out-migration continues at -4,000/year, our projection is on track
- If migration turns positive, our projection is too pessimistic
- Update projections periodically with new data

### 4. Leverage Our Race/Ethnicity Detail

The SDC cannot show how ND's demographic composition will change. Our 6-category approach can answer questions they cannot:

- How will the AIAN population change?
- What's happening to Hispanic population growth?
- How do fertility/mortality differ by race?

### 5. Present Both Projections to Stakeholders

When presenting results, show both projections as a range of possibilities:

- **Optimistic bound**: SDC 2024 (assumes continued in-migration)
- **Pessimistic bound**: Our baseline (assumes continued out-migration)
- **Reality**: Likely somewhere in between, depending on economic conditions

---

## SDC Source File Analysis

This section documents detailed findings from analysis of the SDC's actual source files and working spreadsheets, providing insight into their specific methodological choices and calculations.

### Data Sources Used

#### Base Population
- **Census 2020**: Used as the authoritative base population
- **Census 2010**: Used for historical comparison and migration rate calculation
- **Population Estimates Program (PEP)**: Used cc-est2019-agesex-38 for interim estimates

#### Fertility Data
- **Source**: North Dakota Vital Statistics, Department of Health
- **Time Period**: 2016-2022 (with emphasis on 2018-2022 for rate calculation)
- **Data File**: "2018-2022 ND Res Birth for Kevin Iverson.xlsx" - Prepared September 15, 2023 by Vital Records
- **Categories**: Births by county of residence, age group of mother (Under 20, 20-25, 25-29, 30-34, 35-39, 40-44, 45+)
- **Female Population**: "Average Female Count 2018 to 2022.xlsx" - Average of 2018 and 2022 female populations by age group by county
- **National Reference**: NVSS Report (nvsr72-01.pdf) - National Vital Statistics Reports Volume 71, Number 1, dated January 31, 2023

**Fertility Rate Calculation Method:**
1. Average female population by age group (10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49) for each county from 2018-2022
2. Sum births by mother's age group for 2018-2022 period
3. Calculate 5-year fertility rate = (Total births in age group) / (Average female population in age group)
4. Rates "smoothed to reduce anomalies" - blended with state and national rates for stability

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
  - "Mig Rate 2000-2020_final.xlsx" - Final averaged migration rates
  - "Mig Rates 2000-2020.xlsx" - Working calculations
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
- **Net male migration**: 0.034 (males), -0.019 (females) - significant gender imbalance

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
| Cohort-component structure | 18 age groups x 2 sexes x 53 counties = 1,908 cells | Same structure, plus 6 race categories |
| CDC life tables | ND-specific, 2020 | SEER/CDC, configurable year |
| 5-year survival rates | Calculated from single-year qx | Same approach |
| Fertility by mother's age | 7 age groups (10-14 through 45-49) | Same age groups |

#### Where SDC Source Files Reveal Key Differences

| Aspect | SDC Approach (from source files) | Our Approach |
|--------|----------------------------------|--------------|
| **Migration time period** | 2000-2020 (4 periods averaged) | 2018-2022 (recent years) |
| **Migration method** | Residual (Census-based) | IRS county-to-county flows |
| **Migration adjustment** | 60% dampening of historical rates | No dampening (uses recent data) |
| **Manual adjustments** | Yes - spreadsheet "Adjustments" sheets | No manual adjustments |
| **Mortality improvement** | None visible (constant rates) | Lee-Carter style 0.5%/year |
| **Race/ethnicity** | None | 6 categories |
| **College adjustment** | Special handling for college counties | Handled via age pattern |

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

### Why The Projections Diverge: Source File Evidence

The source files provide definitive evidence for the ~170,000 person divergence by 2045:

1. **Migration Rate Sign**:
   - SDC's averaged rates (2000-2020) show **positive** net migration for most working-age groups
   - Our IRS data (2019-2022) shows **negative** net migration
   - This single factor accounts for ~150,000+ of the divergence

2. **Migration Rate Magnitude**:
   - Even after 60% dampening, SDC projects +4,000 to +6,000 net migrants/year
   - Our data shows -4,000 to -5,000 net migrants/year
   - 20-year difference: ~200,000 people

3. **Mortality Assumptions**:
   - SDC: Constant mortality rates throughout projection
   - Ours: 0.5% annual improvement
   - This slightly increases our projected population relative to SDC, partially offsetting migration

4. **Manual Adjustments**:
   - SDC makes manual adjustments per their "Adjustments" sheets
   - These adjustments are not fully documented but appear to smooth results
   - Our approach is algorithmic with no manual intervention

### Data Quality Observations

From examining the source files:

1. **Small cell suppression**: Birth data uses "NR" (Not Reported) for privacy in small counties
2. **Rounding**: Some intermediate calculations appear rounded
3. **Interpolation**: 2015 population was "synthetic" (interpolated between 2010 and 2020)
4. **Date stamps**: Files dated December 2023 through January 2024, final methodology dated March 7, 2024

### Recommendations Based on Source File Analysis

1. **Consider a "SDC-Calibrated" Scenario**:
   Apply their 60% dampening factor to our migration rates to produce a projection closer to theirs for comparison purposes.

2. **Document Migration Rate Discrepancy**:
   The IRS data and Census residual methods produce fundamentally different pictures of recent migration. Both have validity; the choice depends on which historical period is deemed more representative of the future.

3. **Add Mortality Improvement Toggle**:
   The SDC does not appear to include mortality improvement. Consider making this configurable (0% to 0.5%) to match different methodological choices.

4. **Validate Against 2025 Actuals**:
   When 2025 Census estimates become available, compare both projections to actual to assess which migration assumption is proving more accurate.
