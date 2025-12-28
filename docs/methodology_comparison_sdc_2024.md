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
