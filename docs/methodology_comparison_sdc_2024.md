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

The SDC 2024 projections show:

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

These patterns would be expected from our model as well, assuming similar migration assumptions.

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

1. **Consider adding a "Bakken adjustment" scenario**: If oil-driven migration is expected to be lower than recent averages, we could add a scenario that dampens western ND migration by 40% (matching SDC's implicit assumption).

2. **Compare total population trajectories**: Once our projections are complete, compare to SDC's state totals to identify if migration assumptions cause significant divergence.

3. **Use SDC for validation**: The SDC projections provide a useful benchmark for our state totals — if we diverge significantly, we should investigate why.

4. **Our racial detail is a key value-add**: The SDC cannot show how ND's demographic composition will change; our 6-category approach fills this gap.
