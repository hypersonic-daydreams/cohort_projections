# Census Bureau 2023 National Population Projections: Methodology Reference

**Date:** 2025-12-28
**Author:** Project Team
**Related:** [ADR-017](adr/017-sdc-2024-methodology-comparison.md)

---

## Executive Summary

The U.S. Census Bureau released the 2023 National Population Projections in November 2023, representing the first projections to incorporate 2020 Census results. This document summarizes their methodology and compares it to our North Dakota cohort projection engine.

**Key Finding:** The Census Bureau uses the same fundamental **cohort-component method** with **annual intervals** that our engine uses. The primary differences are in data granularity (national vs. state/county) and the treatment of international migration.

---

## 1. Overview

### 1.1 Projection Parameters

| Parameter | Census Bureau 2023 | Our Engine |
|-----------|-------------------|------------|
| **Base Year** | July 1, 2022 | July 1, 2023 |
| **Projection Horizon** | 2023-2100 (78 years) | 2025-2045 (20 years) |
| **Time Step** | Annual | Annual |
| **Age Detail** | Single-year ages 0-100+ | Single-year ages 0-90+ |
| **Geographic Level** | National only | State + 53 Counties |
| **Race/Ethnicity** | 6 groups | 5 groups |

### 1.2 Fundamental Equation

Both use the cohort-component identity:

```
P(t) = P(t-1) + B(t-1,t) - D(t-1,t) + M(t-1,t)
```

Where:
- P(t) = Population at time t
- B = Births during the interval
- D = Deaths during the interval
- M = Net migration during the interval

---

## 2. Fertility Methodology

### 2.1 Census Bureau Approach

| Aspect | Census Bureau | Our Engine |
|--------|--------------|------------|
| **Age Range** | Women 14-54 | Women 15-49 |
| **TFR 2023** | 1.64 | ~1.73 (ND-specific) |
| **TFR Target** | 1.52 by 2123 | Constant or declining scenarios |
| **Groups** | 6 nativity/race/Hispanic groups | By race/ethnicity |
| **Data Source** | National Vital Statistics | SEER + NCHS |

### 2.2 Census Bureau TFR by Group (2023)

From the `np2023_a5_tfr.csv` data file:

| Group | TFR 2023 | Notes |
|-------|----------|-------|
| **Total** | 1.64 | All women |
| Foreign-Born Hispanic | 2.93 | Highest |
| Foreign-Born NH Asian/Pacific Islander | 1.62 | Below replacement |
| Foreign-Born NH Other | 2.28 | Above replacement |
| Native-Born Asian/Pacific Islander | 1.27 | Lowest |
| Native-Born White | 1.52 | Below replacement |
| Native-Born Other | 1.51 | Below replacement |

### 2.3 Key Insight for ND

North Dakota's TFR (~1.73) is higher than the national average (1.64), reflecting rural/agricultural demographics. Our engine correctly uses ND-specific rates rather than national averages.

---

## 3. Mortality Methodology

### 3.1 Census Bureau Approach

| Aspect | Census Bureau | Our Engine |
|--------|--------------|------------|
| **Life Tables** | UN Model Life Tables | CDC Life Tables + SEER |
| **e0 Male 2023** | ~76.2 years | ~76.5 years (ND) |
| **e0 Female 2023** | ~82.1 years | ~81.8 years (ND) |
| **e0 Target** | 87 (M), 91 (F) by 2123 | 0.5% annual improvement option |
| **Improvement Method** | Converging to target | Linear improvement |

### 3.2 Mortality Improvement

The Census Bureau uses a sophisticated convergence model where life expectancy approaches targets over a 100+ year horizon. Our engine offers simpler linear improvement options (0%, 0.5% per year).

### 3.3 Key Insight for ND

North Dakota life expectancy is close to national average. Our survival rates from CDC/SEER are appropriate for state-level projections.

---

## 4. Migration Methodology

### 4.1 Census Bureau Approach

The Census Bureau models three distinct migration components:

| Component | Method | Data Source |
|-----------|--------|-------------|
| **Foreign-Born Immigration** | Model-based | UN population projections for sending countries |
| **Foreign-Born Emigration** | Residual method | ACS/Census comparisons |
| **Net Native Migration** | Negligible assumption | Set to ~0 |

### 4.2 Our Approach

| Aspect | Census Bureau | Our Engine |
|--------|--------------|------------|
| **Focus** | International migration | Domestic + International |
| **Data Source** | UN projections, ACS | IRS county-to-county flows |
| **Period** | Projected forward | 2019-2022 average |
| **Direction** | Net positive (national) | Net negative (ND observed) |

### 4.3 Critical Difference

**Census Bureau projects continued positive net immigration nationally**, while **our engine projects net out-migration for North Dakota** based on recent IRS data. This is the key driver of projection differences.

### 4.4 Alternative Scenarios

Census Bureau offers three alternative immigration scenarios:

| Scenario | Description |
|----------|-------------|
| **Main Series** | ~1.1M annual net immigration |
| **High Immigration** | +50% above main series |
| **Low Immigration** | Log-symmetric reduction |
| **Zero Immigration** | No international migration |

---

## 5. Race/Ethnicity Treatment

### 5.1 Birth Assignment

The Census Bureau uses the "Kid Link Method" to assign race/ethnicity to births:

1. Use 2010 Census parent-child linkages
2. Determine probability of child's race given parents' races
3. Apply to projected births

### 5.2 Categories

| Census Bureau | Our Engine |
|--------------|------------|
| Non-Hispanic White alone | Non-Hispanic White |
| Non-Hispanic Black alone | Non-Hispanic Black |
| Non-Hispanic Asian alone | Non-Hispanic Asian |
| Non-Hispanic All Other | Non-Hispanic Other |
| Hispanic | Hispanic |
| Two or More Races | (included in Other) |

---

## 6. Comparison with SDC 2024

| Aspect | Census Bureau 2023 | SDC 2024 | Our Engine |
|--------|-------------------|----------|------------|
| **Method** | Cohort-component | Cohort-component | Cohort-component |
| **Time Step** | Annual | 5-year | Annual |
| **Geographic** | National | ND State + Counties | ND State + Counties |
| **Migration** | International only | Domestic + Int'l | Domestic + Int'l |
| **Migration Direction** | Net positive | Net positive (dampened) | Net negative |

---

## 7. Data Files Downloaded

The following Census Bureau files have been downloaded to `data/raw/census_bureau_methodology/`:

### 7.1 Methodology Documents

| File | Description | Size |
|------|-------------|------|
| `methodstatement23.pdf` | Main 2023 National Population Projections methodology | 1.1 MB |
| `sptoolkitusersguide.pdf` | Subnational Projections Toolkit User's Guide | 699 KB |
| `methods-statement-v2023.pdf` | Population estimates methodology (2020-2023) | 422 KB |

### 7.2 Data Files

| File | Description | Size |
|------|-------------|------|
| `np2023_a1_fertility_rates.csv` | Age-specific fertility rates by group, 2023-2100 | 88 KB |
| `np2023_a3_life_expectancy.csv` | Life expectancy at birth by group, 2023-2100 | 29 KB |
| `np2023_a4_survival_ratios.csv` | Survival ratios by age/sex/group, 2023-2100 | 1.2 MB |
| `np2023_a5_tfr.csv` | Total fertility rates by group, 2023-2100 | 3 KB |

---

## 8. Implications for Our Projections

### 8.1 Validation

Our cohort-component methodology aligns with the Census Bureau's approach:
- Annual time step (same)
- Single-year age detail (same)
- Component-based accounting (same)

### 8.2 Differences to Document

When presenting our projections, we should note:

1. **Migration is state-specific**: We use IRS domestic migration data showing net out-migration for ND, unlike national projections showing net in-migration

2. **Fertility is state-specific**: ND TFR (~1.73) exceeds national average (1.64)

3. **Time horizon is shorter**: We project 20 years (2025-2045) vs. 78 years (2023-2100)

### 8.3 Potential Enhancements

Consider adopting from Census Bureau methodology:

1. **Mortality improvement curves**: Their convergence model could replace our linear improvement
2. **Race-specific rates**: Their 6-group breakdown offers more granularity
3. **Alternative scenarios**: Formalize high/low scenarios with multipliers

---

## 9. References

### 9.1 Census Bureau Sources

- [2023 National Population Projections Tables: Main Series](https://www.census.gov/data/tables/2023/demo/popproj/2023-summary-tables.html)
- [2023 National Population Projections Datasets](https://www.census.gov/data/datasets/2023/demo/popproj/2023-popproj.html)
- [Population Projections Technical Documentation](https://www.census.gov/programs-surveys/popproj/technical-documentation.html)

### 9.2 Project Documentation

- [ADR-017: SDC 2024 Methodology Comparison](adr/017-sdc-2024-methodology-comparison.md)
- [Methodology Analysis: Cohort vs Age-Group](methodology_analysis_cohort_vs_agegroup.md)

---

## Appendix: TFR Projections Over Time

From `np2023_a5_tfr.csv`:

| Year | Total TFR | FB Hispanic | NB White |
|------|-----------|-------------|----------|
| 2023 | 1.64 | 2.93 | 1.52 |
| 2030 | 1.62 | 2.82 | 1.52 |
| 2040 | 1.60 | 2.67 | 1.52 |
| 2050 | 1.58 | 2.52 | 1.52 |
| 2060 | 1.56 | 2.37 | 1.52 |
| 2100 | 1.52 | 1.52 | 1.52 |

The Census Bureau projects fertility convergence across all groups to a common TFR of 1.52 by 2123.
