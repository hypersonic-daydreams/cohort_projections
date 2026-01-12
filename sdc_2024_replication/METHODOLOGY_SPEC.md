# SDC 2024 Methodology Technical Specification

**Purpose:** Document the exact computational methodology used by the North Dakota State Data Center (SDC) in their 2024 population projections, for potential replication.

**Source Documents:**
- SDC 2024 Population Projections Report (February 6, 2024)
- `Projections_Base_2023.xlsx` workbook (primary calculation engine)
- `County_Population_Projections_2023.xlsx` (county-level output)
- `Mig Rate 2000-2020_final.xlsx` (migration rate derivation)

**Date:** 2025-12-28

---

## v0.8.6 Revision Addendum (Adopted Methodology Changes)

This document primarily specifies the SDC’s 2024 cohort-survival spreadsheet methodology. For the **v0.8.6 journal-article replication and analysis pipeline**, we adopt a set of additional methodological choices and robustness enhancements (data extensions, time-base alignment, regime diagnostics, and uncertainty reporting) that are **not** part of the original SDC workbook, but are required to address the v0.8.5 critique and to make uncertainty sources auditable.

Canonical decision records (v0.8.6 adopted):

- Data extension + alignment:
  - ADR-024 `docs/governance/adrs/024-immigration-data-extension-fusion.md`
  - ADR-025 `docs/governance/adrs/025-refugee-coverage-missing-state-handling.md`
  - ADR-026 `docs/governance/adrs/026-amerasian-siv-handling-forecasting.md`
- Causal timing + robustness:
  - ADR-027 `docs/governance/adrs/027-travel-ban-extended-dynamics-supplement.md`
- Uncertainty and simulation rigor:
  - ADR-028 `docs/governance/adrs/028-monte-carlo-simulation-rigor-parallelization.md`
  - ADR-029 `docs/governance/adrs/029-wave-duration-refit-right-censoring-hazard.md`
  - ADR-032 `docs/governance/adrs/032-uncertainty-envelopes-two-band-approach.md`
- Long-run context / measurement regimes:
  - ADR-020 `docs/governance/adrs/020-extended-time-series-methodology-analysis.md`
  - ADR-030 `docs/governance/adrs/030-pep-regime-aware-modeling-long-run-series.md`
- Appendix-only covariate-conditioned near-term diagnostic:
  - ADR-031 `docs/governance/adrs/031-covariate-conditioned-near-term-forecast-anchor.md`

Important scope note:
- These v0.8.6 changes refine the **replication/analysis** and the paper’s forecasting uncertainty reporting; they do not redefine the SDC 2024 cohort-component projection engine documented below.

## 1. Projection Framework

### 1.1 Method Classification

SDC uses a **modified cohort survival component method** (their terminology). From the official report:

> "The process used in these projections is a modified version of the cohort survival component method, the most used method of projections."

This is a standard demographic projection approach where:
1. Population cohorts are aged forward through time
2. Survival rates are applied to each cohort
3. Births are added from fertility rates applied to women of childbearing age
4. Migration is added as a component after natural growth

### 1.2 Temporal Structure

| Parameter | Value |
|-----------|-------|
| Base Year | 2020 (Census April 1) |
| Projection Horizon | 2020-2050 |
| Projection Intervals | 5-year periods |
| Projection Points | 2025, 2030, 2035, 2040, 2045, 2050 |
| Number of Periods | 6 (each 5 years) |

### 1.3 Geographic Structure

| Level | Count | Aggregation |
|-------|-------|-------------|
| County | 53 | Primary projection unit |
| Region | 8 | Sum of member counties |
| State | 1 | Sum of all counties |

**Critical:** Projections are calculated at the county level and summed upward. State totals are the sum of 53 county projections.

### 1.4 Demographic Structure

| Dimension | Categories |
|-----------|------------|
| Age Groups | 18 five-year groups: 0-4, 5-9, 10-14, ..., 80-84, 85+ |
| Sex | Male, Female (projected separately) |
| Race/Ethnicity | Not included in projections |

---

## 2. Core Projection Formula

### 2.1 Cohort Survival Formula

For each 5-year projection period, each age-sex-county cohort is projected as:

```
Natural_Growth[age+5, t+5] = Population[age, t] * Survival_Rate[age, sex]
```

Where:
- `age` is the starting age group (0-4, 5-9, etc.)
- `age+5` is the next age group (cohort ages 5 years)
- `t` is the starting year
- `t+5` is the ending year (5 years later)
- `Survival_Rate` is the probability of surviving from one age group to the next

### 2.2 Migration Formula

Migration is calculated as a proportion of natural growth:

```
Migration[age, t+5] = Natural_Growth[age, t+5] * Migration_Rate[age, sex, county] * Period_Multiplier[period]
```

The `Period_Multiplier` varies by projection period (see Section 3.2).

### 2.3 Final Projection Formula

```
Population[age, t+5] = Natural_Growth[age, t+5] + Migration[age, t+5] + Adjustments[age, county, period]
```

### 2.4 Births (Age 0-4 Cohort)

The 0-4 age group is calculated separately using fertility:

```
Births[t to t+5] = SUM over age groups [
    Female_Pop[age, t] * Fertility_Rate[age]
] * 5 years

Population[0-4, t+5] = Births[t to t+5] * Survival_Rate[birth to age 0-4]
```

Note: Children born to in-migrants are treated as part of net migration, not births.

---

## 3. Component Rates

### 3.1 Fertility Rates

**Data Sources:**
- North Dakota DHHS Vital Statistics: 2016-2022 birth data by county
- CDC National Vital Statistics Reports (2021): National age-specific fertility rates
- State-level rates from ND data

**Blending Methodology:**
1. Calculate county-level age-specific fertility rates from ND vital statistics
2. Blend with state-level rates to smooth small-county anomalies
3. Further blend with national rates for additional stability
4. Rates are "smoothed to reduce anomalies in the number of births during this timeframe"

**Age Groups for Fertility:**
- Applied to females in childbearing ages (typically 15-44 or 15-49)
- 5-year age groups matching the projection structure

**Rate Application:**
- Rates are held constant across all projection periods
- No fertility decline or improvement scenarios applied

### 3.2 Survival Rates

**Data Source:**
- CDC National Center for Health Statistics
- Life Tables for North Dakota, published 2022 (for year 2020)
- Source file: `nvsr71-02.pdf`

**Life Table Files Used:**
- `ND1.xlsx`, `ND2.xlsx`, `ND3.xlsx`, `ND4.xlsx` (processed CDC data)

**Application:**
- Sex-specific survival probabilities
- 5-year survival probability: probability of surviving from age group `a` to age group `a+5`
- Rates held constant across all projection periods (no mortality improvement)

**Special Handling for 85+:**
- Open-ended age group
- Uses remaining life expectancy calculations from life tables

### 3.3 Migration Rates

**This is the most complex and consequential component.**

#### 3.3.1 Base Rate Calculation

**Method:** Census Residual (Component Method II)

```
Observed_Migration[age, sex, county, period] =
    Actual_Pop[age+5, t+5] - Expected_Pop[age+5, t+5]

Where:
    Expected_Pop[age+5, t+5] = Pop[age, t] * Survival_Rate[age, sex]
```

**Time Periods Used:**
- 2000-2005
- 2005-2010
- 2010-2015
- 2015-2020

Each period produces migration estimates for each age-sex-county cell. The four periods are averaged to produce a single migration rate.

**Cell Count:**
- 18 age groups x 2 sexes x 53 counties = 1,908 individual cells per period
- 4 periods averaged

#### 3.3.2 Migration Rate Expression

Migration rates are expressed as a percentage of the natural growth:

```
Migration_Rate[age, sex, county] =
    AVG over 4 periods [ Observed_Migration / Natural_Growth ]
```

**Sample Raw (Undampened) Rates (State Average):**

| Age Group | Male Rate | Female Rate |
|-----------|-----------|-------------|
| 0-4 | +10.78% | +7.06% |
| 5-9 | +4.87% | +4.85% |
| 10-14 | +5.36% | +5.38% |
| 15-19 | +17.34% | +15.79% |
| **20-24** | **+32.77%** | +24.49% |
| 25-29 | -11.90% | -24.32% |
| 30-34 | +5.88% | +1.13% |
| 35-39 | +3.21% | -1.45% |
| ... | ... | ... |

#### 3.3.3 Bakken Dampening

**Rationale:** The 2010-2020 period experienced the Bakken oil boom, creating historically unprecedented in-migration. SDC judged this was unsustainable for projection purposes.

**General Dampening:** Rates were "typically reduced to about 60% of what was found"

However, the implementation uses **period-specific multipliers**:

| Period | Multiplier | Notes |
|--------|------------|-------|
| 2020-2025 | 0.2 | Very low - COVID adjustment, post-boom transition |
| 2025-2030 | 0.6 | Bakken dampening |
| 2030-2035 | 0.6 | Bakken dampening |
| 2035-2040 | 0.5 | Further reduced |
| 2040-2045 | 0.7 | Increasing toward normal |
| 2045-2050 | 0.7 | Increasing toward normal |

**Effective Formula:**
```
Effective_Migration_Rate = Base_Rate * Period_Multiplier
```

Example for 20-24 Males in 2025-2030:
```
Effective_Rate = 32.77% * 0.6 = 19.66%
```

#### 3.3.4 Interpretation

The base migration rates from 2000-2020 show **net in-migration** overall:
- Strong in-migration for young adults (20-24) especially males
- Out-migration for 25-29 females (post-college departure)
- Retirement-age out-migration (65+)

After dampening, SDC still projects net in-migration, but at reduced levels.

---

## 4. Manual Adjustments

SDC applies manual adjustments beyond the formula-based calculations.

### 4.1 Adjustment Types

From the report and workbook analysis:

1. **College-Age Corrections:**
   > "Counties with significant college age populations typically required additional adjustments as the algorithm tends to not capture the in- and out-migration of college age residents as well as it should."

   Affected counties: Grand Forks (UND), Cass (NDSU), Ward (Minot State), etc.

2. **Bakken Region Sex-Ratio Balancing:**
   > "The rate of male migration was further reduced compared to female migration as the pattern found from 2000 to 2020 when in-migration was dominated by males is unlikely to continue into the future and would have resulted in unrealistic sex ratio in future years."

3. **Regional Economic Adjustments:**
   > "The western areas of the state, which are dominated by energy developments and the boom to bust cycle... is less predictable than the eastern half of the state."

### 4.2 Adjustment Magnitude

From workbook analysis:
- Approximately **~32,000 total person-adjustments per 5-year period**
- Both positive and negative adjustments by county/age/sex
- Net effect varies by period

### 4.3 Adjustment Implementation

In the workbook formula:
```
Population[age, county, t+5] = Natural_Growth + Adjustments + Migration
```

Adjustments are in a separate column/sheet and added to the projection formula.

---

## 5. Published Results Summary

### 5.1 State Population Projections

| Year | Population | 5-Year Change | Cumulative Change from 2020 |
|------|------------|---------------|----------------------------|
| 2020 | 779,094 | -- | -- |
| 2025 | 796,989 | +17,895 (+2.3%) | +2.3% |
| 2030 | 831,543 | +34,554 (+4.3%) | +6.7% |
| 2035 | 865,397 | +33,854 (+4.1%) | +11.1% |
| 2040 | 890,424 | +25,027 (+2.9%) | +14.3% |
| 2045 | 925,101 | +34,677 (+3.9%) | +18.7% |
| 2050 | 957,194 | +32,093 (+3.5%) | +22.9% |

### 5.2 Components of Change

| Period | Natural Change | Net Migration | Total Change |
|--------|----------------|---------------|--------------|
| 2020-2025 | 6,102 | 7,717 | 13,819 |
| 2025-2030 | 16,404 | 18,154 | 34,558 |
| 2030-2035 | 10,321 | 23,539 | 33,860 |
| 2035-2040 | 1,083 | 23,948 | 25,031 |
| 2040-2045 | 5,572 | 29,111 | 34,683 |
| 2045-2050 | (5,526) | 37,624 | 32,098 |

**Key Observations:**
- Natural change declines over time, turning negative by 2045-2050
- Net migration increases over time (as period multipliers increase from 0.2 to 0.7)
- Migration dominates population change (~75% of total)

---

## 6. Known Discrepancies

### 6.1 Workbook vs Published Gap

The SDC Excel workbook (`Projections_Base_2023.xlsx`) produces different totals than the published projections:

| Year | Workbook Total | Published Total | Gap |
|------|---------------|-----------------|-----|
| 2020 | 749,134 | 779,094 | -29,960 |
| 2025 | 762,238 | 796,989 | -34,751 |
| 2030 | 789,127 | 831,543 | -42,416 |
| 2035 | 817,852 | 865,397 | -47,545 |
| 2040 | 842,033 | 890,424 | -48,391 |
| 2045 | 875,651 | 925,101 | -49,450 |
| 2050 | 912,903 | 957,194 | -44,291 |

**Possible Explanations:**
1. Workbook is an earlier version before final adjustments
2. Additional post-workbook adjustments made
3. County-level aggregation includes additional factors
4. Multiple workbooks with different versions exist

This remains unresolved without SDC clarification.

---

## 7. Replication Requirements

To replicate SDC 2024 methodology exactly, the following elements are needed:

### 7.1 Must Replicate

| Element | Specification |
|---------|---------------|
| **Time Step** | 5-year intervals (2020->2025->2030->...) |
| **Age Groups** | 18 five-year groups (0-4, 5-9, ..., 85+) |
| **Geographic Level** | 53 counties, summed to state |
| **Sex** | Male and Female projected separately |
| **Projection Formula** | Pop[t+5] = Nat_Grow + Migration + Adjustments |
| **Cohort Aging** | Age groups shift up by one each period |

### 7.2 Data Inputs Required

| Input | Source | Notes |
|-------|--------|-------|
| **Base Population** | Census 2020 | By county, age group, sex |
| **Survival Rates** | CDC 2020 Life Tables for ND | Sex-specific, by age group |
| **Fertility Rates** | ND DHHS 2016-2022 blended | County-level, blended with state/national |
| **Migration Rates** | Census residual 2000-2020 | By county, age group, sex |
| **Period Multipliers** | SDC determination | 0.2, 0.6, 0.6, 0.5, 0.7, 0.7 |
| **Adjustments** | SDC manual | ~32,000/period, by county/age/sex |

### 7.3 Algorithmic Steps

For each projection period (2020-2025, 2025-2030, etc.):

1. **For each county, for each sex:**

   a. **Calculate births:**
   ```
   Births = SUM[ Female_Pop[age] * Fertility_Rate[age] ] * 5
   Pop[0-4] = Births * Infant_Survival_Rate
   ```

   b. **Age cohorts forward (for ages 5+):**
   ```
   For each age group a from 5-9 to 80-84:
       Nat_Grow[a+5] = Pop[a] * Survival[a]

   For age 85+:
       Nat_Grow[85+] = Pop[80-84] * Survival[80-84] + Pop[85+] * Survival[85+]
   ```

   c. **Add migration:**
   ```
   For each age group:
       Migration = Nat_Grow * Mig_Rate[age,sex,county] * Period_Multiplier
   ```

   d. **Add adjustments:**
   ```
   For each age group:
       Final_Pop = Nat_Grow + Migration + Adjustment[age,sex,county,period]
   ```

2. **Sum counties to state:**
   ```
   State_Pop[age,sex,year] = SUM over counties[ County_Pop[age,sex,year] ]
   ```

3. **Use final population as base for next period:**
   ```
   Pop[t] = Final_Pop for period t
   Repeat for next period
   ```

---

## 8. Comparison with Our Baseline 2026 Methodology

| Aspect | SDC 2024 | Our Baseline 2026 |
|--------|----------|-------------------|
| **Method** | Cohort-component (5-year) | Cohort-component (annual) |
| **Age Detail** | 18 five-year groups | Single-year ages (0-90+) |
| **Migration Source** | Census residual 2000-2020 | IRS county-to-county 2019-2022 |
| **Migration Direction** | Net IN-migration | Net OUT-migration |
| **Dampening** | 20%-70% by period | None |
| **Manual Adjustments** | ~32,000/period | None |
| **Fertility Trend** | Constant | Optional decline |
| **Mortality Trend** | Constant | Optional improvement |

**Impact of Differences:**
- The migration direction difference alone accounts for ~170,000 person divergence by 2045
- Our projection shows population decline (~755,000 by 2045)
- SDC projection shows population growth (~925,000 by 2045)

---

## 9. References

### 9.1 SDC Source Files

- `/data/raw/nd_sdc_2024_projections/full_report_text.md`
- `/data/raw/nd_sdc_2024_projections/source_files/results/Projections_Base_2023.xlsx`
- `/data/raw/nd_sdc_2024_projections/source_files/migration/Mig Rate 2000-2020_final.xlsx`
- `/data/raw/nd_sdc_2024_projections/source_files/life_tables/ND*.xlsx`
- `/data/raw/nd_sdc_2024_projections/source_files/fertility/*.xlsx`

### 9.2 Project Documentation

- `/docs/methodology_analysis_cohort_vs_agegroup.md`
- `/docs/governance/adrs/017-sdc-2024-methodology-comparison.md`
- `/data/processed/sdc_2024/METHODOLOGY_NOTES.md`

### 9.3 External References

- CDC NCHS Life Tables: https://www.cdc.gov/nchs/products/index.htm
- National Vital Statistics Reports: https://www.cdc.gov/nchs/products/index.htm
- Census Bureau Population Estimates Program

---

## Appendix A: Age Group Reference

| Index | Age Group | Label |
|-------|-----------|-------|
| 1 | 0-4 | Early childhood |
| 2 | 5-9 | Childhood |
| 3 | 10-14 | Pre-teen |
| 4 | 15-19 | Teen |
| 5 | 20-24 | College age |
| 6 | 25-29 | Young adult |
| 7 | 30-34 | Adult |
| 8 | 35-39 | Adult |
| 9 | 40-44 | Adult |
| 10 | 45-49 | Middle age |
| 11 | 50-54 | Middle age |
| 12 | 55-59 | Pre-retirement |
| 13 | 60-64 | Pre-retirement |
| 14 | 65-69 | Young elderly |
| 15 | 70-74 | Elderly |
| 16 | 75-79 | Elderly |
| 17 | 80-84 | Oldest old |
| 18 | 85+ | Oldest old (open) |

## Appendix B: Economic Planning Regions

| Region | Name | Counties |
|--------|------|----------|
| 1 | Williston | Williams, McKenzie, Divide |
| 2 | Minot | Ward, McHenry, Pierce, Bottineau, Renville, Burke, Mountrail |
| 3 | Devils Lake | Ramsey, Benson, Cavalier, Towner, Rolette, Nelson, Eddy |
| 4 | Grand Forks | Grand Forks, Walsh, Pembina, Nelson |
| 5 | Fargo | Cass, Richland, Ransom, Sargent, Steele, Traill |
| 6 | Jamestown | Stutsman, Barnes, Foster, Griggs, LaMoure, Dickey, Logan, McIntosh, Wells |
| 7 | Bismarck | Burleigh, Morton, McLean, Mercer, Oliver, Sheridan, Kidder, Emmons, Grant, Sioux |
| 8 | Dickinson | Stark, Dunn, Hettinger, Adams, Bowman, Golden Valley, Billings, Slope |

## Appendix C: Period Multiplier Rationale

| Period | Multiplier | Rationale |
|--------|------------|-----------|
| 2020-2025 | 0.2 | COVID-19 impacts, post-Bakken adjustment, observed early-decade out-migration |
| 2025-2030 | 0.6 | Bakken dampening - boom conditions unlikely to repeat |
| 2030-2035 | 0.6 | Continued dampening |
| 2035-2040 | 0.5 | Further reduction - conservative estimate |
| 2040-2045 | 0.7 | Gradual return toward historical patterns |
| 2045-2050 | 0.7 | Gradual return toward historical patterns |

The variable multipliers create a U-shaped migration trajectory: very low in 2020-2025, increasing through mid-projection, then stabilizing at 70% of historical rates.
