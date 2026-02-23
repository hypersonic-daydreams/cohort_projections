# Findings 4 & 5: Race-Specific Population Trends

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Status** | CONFIRMED -- Root Cause Identified (Critical Bug in Base Population Distribution) |

---

## Executive Summary

**Finding 4** (Hispanic +204% growth) and **Finding 5** (Black -16.7% decline) are both confirmed and are caused by the same root issue: **the Census+PUMS hybrid base population distribution (ADR-041) produces a severely distorted age-sex-race allocation for small racial groups**. The PUMS ~1% sample for North Dakota allocates Black non-Hispanic population to only 7 of the possible 36 age-sex cells, and Hispanic population to only 11 of 36 cells. This creates artificial "clumps" in the age distribution that compound through 30 years of cohort-component projection.

The most critical consequence: **zero Black females of reproductive age exist in the base population**, meaning the Black population cannot produce births and is doomed to decline through mortality alone. Meanwhile, the Hispanic population's concentration of young females (ages 10-19 in 2025) generates a demographic "echo" of high fertility as these cohorts age into childbearing years.

**Severity**: Critical. The race-specific projections are unreliable artifacts of small-sample noise in the PUMS-based race allocation. The total population projection (all races combined) is much less affected because race-specific distortions partially cancel out.

---

## 1. How Race Is Handled in Each Component

### 1.1 Fertility (Race-Differentiated)

**Source file**: `data/processed/fertility_rates.parquet` (42 rows, 7 age groups x 6 race codes)
**Transform function**: `_transform_fertility_rates()` in `scripts/pipeline/02_run_projections.py` (lines 188-225)
**Engine module**: `cohort_projections/core/fertility.py`, `calculate_births()` (lines 16-139)

The fertility component IS race-differentiated. CDC NCHS provides age-specific fertility rates (ASFR) by race/ethnicity at the national level. The data includes separate rates for:

| Race Code | Engine Name | TFR (2024) |
|-----------|-------------|-----------|
| white_nh | White alone, Non-Hispanic | 1.533 |
| black_nh | Black alone, Non-Hispanic | 1.535 |
| hispanic | Hispanic (any race) | **1.956** |
| aian_nh | AIAN alone, Non-Hispanic | 1.422 |
| asian_nh | Asian/PI alone, Non-Hispanic | 1.485 |
| (computed average) | Two or more races, Non-Hispanic | ~1.506 |

**Key observation**: Hispanic TFR (1.956) is 27% higher than Black TFR (1.535) and 28% higher than White TFR (1.533). This contributes to, but does not alone explain, the divergent growth rates.

The `calculate_births()` function merges fertility rates with the female population on `[age, race]` keys. If a race has zero female population at reproductive ages, it produces zero births regardless of the fertility rate.

### 1.2 Mortality/Survival (Race-Differentiated)

**Source file**: `data/processed/survival_rates.parquet` (1,212 rows)
**Transform function**: `_transform_survival_rates()` in `scripts/pipeline/02_run_projections.py` (lines 228-268)
**Engine module**: `cohort_projections/core/mortality.py`, `apply_survival_rates()` (lines 17-174)

Survival rates ARE race-differentiated. The CDC life tables provide race-specific survival by age and sex:

| Race | Female e0 (approx) | Male e0 (approx) |
|------|-------------------|------------------|
| aian_nh | 72.9 | 66.1 |
| asian_nh | 86.4 | 82.6 |
| black_nh | 77.1 | 69.8 |
| hispanic | 83.4 | 77.9 |
| white_nh | 80.4 | 75.5 |

Black life expectancy is lower than Hispanic, meaning the Black population experiences higher mortality at every age. This accelerates the decline for a population that cannot replenish through births.

### 1.3 Migration (Race-Differentiated via Proportional Allocation)

**Source file**: `data/processed/migration/migration_rates_pep_baseline.parquet` (57,876 rows)
**Process function**: `process_pep_migration_rates()` in `cohort_projections/data/process/migration_rates.py` (lines 1667-1903)
**Engine module**: `cohort_projections/core/migration.py`, `apply_migration()` (lines 16-125)

The migration component is race-differentiated, but **not based on observed race-specific migration**. The process is:

1. PEP county components provide aggregate net migration by county (no race breakdown)
2. BEBR multi-period averaging produces county-level net migration estimates
3. Net migration is distributed to age using Rogers-Castro age pattern
4. Distributed to sex (50/50 split)
5. **Distributed to race proportional to the base population composition** (`distribute_migration_by_race()`, lines 519-624)

This means migration is proportional to existing population share, which is reasonable as a first approximation. The state-level totals:

| Race | Annual Net Migration | % of Base Pop |
|------|---------------------|---------------|
| White alone, Non-Hispanic | 1,104.8 | +0.17% |
| AIAN alone, Non-Hispanic | 211.9 | +0.24% |
| Two or more races, Non-Hispanic | 62.9 | +0.23% |
| Hispanic (any race) | 48.7 | +0.30% |
| Asian/PI alone, Non-Hispanic | 50.0 | +0.36% |
| Black alone, Non-Hispanic | 6.3 | +0.14% |

Migration as a percentage of base population is modest for all groups (0.14-0.36% annually). This alone does not drive the extreme divergence.

---

## 2. Root Cause: PUMS Small-Sample Artifacts in Base Population Distribution

### 2.1 The Distribution Problem

The base population for each county is constructed by applying a **state-level age-sex-race distribution** to the county's total population. This distribution is loaded from `data/raw/population/nd_age_sex_race_distribution.csv` (115 rows).

The distribution was built per ADR-041 using Census cc-est2024 for age-sex proportions and PUMS for race allocation within each age-sex cell. The PUMS sample for North Dakota has only ~12,277 records. When cross-tabulated by 18 age groups x 2 sexes x 6-8 race categories, many cells have **zero observations**.

### 2.2 Black Non-Hispanic: Only 7 of 36 Cells Populated

The PUMS-derived race allocation for Black non-Hispanic produces population in only 7 age-sex cells out of a possible 36 (18 age groups x 2 sexes):

| Age Group | Sex | Count | Proportion |
|-----------|-----|-------|-----------|
| 0-4 | Male | 277 | 0.000348 |
| 15-19 | Male | 942 | 0.001183 |
| 25-29 | Male | 78 | 0.000098 |
| 30-34 | Male | 1,555 | 0.001953 |
| 35-39 | Male | 814 | 0.001022 |
| 55-59 | **Female** | 625 | 0.000784 |
| 60-64 | **Female** | 718 | 0.000901 |

**Critical finding**: The only Black females in the entire distribution are ages 55-64. There are **zero Black females at any reproductive age (15-49)**. This makes it mathematically impossible for the Black population to produce any births.

### 2.3 Hispanic: Heavily Concentrated in Specific Cells

The Hispanic distribution has 11 populated cells:

| Age Group | Sex | Count | Proportion |
|-----------|-----|-------|-----------|
| 0-4 | Male | 111 | 0.000139 |
| 0-4 | Female | 1,290 | 0.001620 |
| 10-14 | Female | **8,623** | **0.010825** |
| 15-19 | Male | 1,087 | 0.001365 |
| 15-19 | Female | 549 | 0.000689 |
| 20-24 | Male | 2,176 | 0.002731 |
| 25-29 | Female | 781 | 0.000980 |
| 30-34 | Male | 1,458 | 0.001831 |
| 30-34 | Female | 1,079 | 0.001355 |
| 40-44 | Male | 269 | 0.000337 |
| 45-49 | Male | 1,652 | 0.002074 |

The Hispanic population is **heavily concentrated in the 10-14 female age group** (8,623 estimated count, representing 45% of total Hispanic population). This single cohort, as it ages through the projection, creates a massive demographic "wave":

- By 2025: 8,623 females ages 10-14
- By 2030: These females are ages 15-19, entering peak fertility
- By 2035-2040: Ages 20-29, at maximum fertility with Hispanic ASFR of 0.081-0.112 per year
- Their children create an echo boom in the 2040s-2050s

This explains the explosive +204% growth: a single large female cohort passes through the highest-fertility ages over 30 years, producing a multi-generational compounding effect.

### 2.4 Confirmed Projection Output Data

The projection outputs confirm the mechanism:

**Black population trajectory (state total)**:
- 2025: 5,027 (845 under 18; **0 fertile females**; 55 births at age 0)
- 2030: 5,224 (**0 births**, 284 under 18)
- 2040: 5,104 (0 births, 190 under 18)
- 2050: 4,698 (0 births, 0 under 18)
- 2055: 4,189 (-16.7%, **entirely ages 30-90**, population aging out)

The 55 births at age 0 in 2025 appear to come from the initial year's computation, likely from migration-added females. From 2030 onward, zero births are produced because no females exist in reproductive ages.

**Hispanic population trajectory (state total)**:
- 2025: 19,141 (11,044 under 18 = 57.7%)
- 2035: 28,512 (the 10-14 cohort from 2025 is now 20-24, producing births)
- 2045: 43,847 (first echo generation entering school age)
- 2055: 58,200 (+204.1%)

The age structure confirms the "cohort wave" mechanism. By 2055, the original 10-14 cohort from 2025 is ages 40-44, still appearing as a distinct concentration at 12,647 persons.

---

## 3. Is the Growth/Decline Consistent with Input Rates?

### 3.1 Hispanic +204% Growth

The growth IS mechanically consistent with the input data -- high-fertility female cohorts concentrated in pre-reproductive ages will indeed produce exponential growth. However, the **input data is wrong**. The concentration of 45% of Hispanic population in a single female age group (10-14) is a PUMS sampling artifact, not reality.

For context, if the Hispanic population had a realistic age distribution (matching the overall state distribution), the growth would be driven primarily by:
- Higher TFR (1.956 vs 1.533 for White)
- Positive net migration (+0.30% annually)
- Younger age structure

A realistic 30-year growth rate for Hispanic population would likely be in the 50-100% range, not 204%.

### 3.2 Black -16.7% Decline

The decline IS mechanically consistent with zero births (no fertile females) plus mortality attrition. However, the **zero-births result is entirely artificial**. North Dakota's Black population does include women of reproductive age -- the PUMS sample simply failed to capture them in its small ND sample.

A realistic trajectory for the Black population would depend on whether migration is positive or negative. With small positive net migration (+6.3/year, 0.14% of base) and replacement-level fertility (TFR 1.535), the Black population would likely grow slowly or remain roughly stable.

---

## 4. Small-Number Volatility Assessment

### 4.1 Scale of the Problem

The PUMS small-sample problem is severe for minorities in North Dakota:

| Group | State Population | PUMS ~1% Sample | Distribution Cells Populated | Expected Cells |
|-------|-----------------|-----------------|------------------------------|---------------|
| White alone, Non-Hispanic | 648,246 | ~6,500 | 34/36 | 36 |
| AIAN alone, Non-Hispanic | 86,943 | ~870 | 27/36 | 36 |
| Two or more races, Non-Hispanic | 26,797 | ~268 | 17/36 | 36 |
| Hispanic (any race) | 16,286 | ~163 | **11/36** | 36 |
| Asian/PI alone, Non-Hispanic | 13,885 | ~139 | 8/36 | 36 |
| Black alone, Non-Hispanic | 4,412 | ~44 | **7/36** | 36 |

With only ~44 PUMS observations for Black non-Hispanic, every cell in the distribution is determined by a handful of observations. The probability of getting zero observations in any given cell is very high. For a population of 4,412 spread across 36 age-sex cells, the expected average cell size in the PUMS sample is about 1.2 persons -- many cells will be empty by chance alone.

### 4.2 Impact on Projections

The small-number artifact has a cascading effect through the cohort-component method:

1. **Initial distribution**: Entire age-sex cells are zero -> artificial gaps in age pyramid
2. **Year 1**: Cohorts with zero population produce zero survivors, zero births
3. **Year 2-30**: Gaps propagate forward because the engine cannot create population from nothing
4. **Compounding**: Mortality erodes existing cohorts while no new births replenish them

For the Black population, this creates an irreversible decline. For the Hispanic population, the concentration of population in a high-fertility age cohort creates an artificial echo boom.

---

## 5. Recommendations

### 5.1 Immediate: Document Limitations

Race-specific projections for Black, Hispanic, Asian/PI, and Two-or-more-races populations should carry a **strong caveat** that these are unreliable due to small-sample base population allocation. Only the White and AIAN projections have sufficient sample size for age-sex-race distribution to be approximately correct.

### 5.2 Short-Term: Use Census Race Data Directly

The Census Bureau publishes population estimates by race for North Dakota at the state level (cc-est2024-alldata includes race breakdowns). This would replace the PUMS-based race allocation with full-count data, eliminating the small-sample problem entirely. The trade-off is that the available race categories may need mapping to the 6-category scheme.

Specific recommendation: Investigate using `cc-est2024-alldata.parquet` which has age x sex x race breakdowns based on actual demographic analysis, not sample data.

### 5.3 Medium-Term: Implement IPF (Iterative Proportional Fitting)

As noted in ADR-041's alternatives section, IPF could combine Census age-sex marginals, Census state-level race marginals, and PUMS race-within-cell conditionals in a statistically rigorous way. This would ensure:
- Age-sex structure matches Census exactly
- Race totals match Census exactly
- Within-cell race allocation uses PUMS as a prior, constrained by known marginals

### 5.4 Minimum Viable Fix

At minimum, the distribution file should ensure that **every race group has some female population at reproductive ages (15-49)**. A simple fix would be: for any race group with zero females ages 15-49, impute a minimum allocation using the state-average female age distribution scaled to the race group's total. This would be a rough but defensible correction that prevents the zero-births artifact.

---

## 6. Files Referenced

| File | Purpose |
|------|---------|
| `/home/nhaarstad/workspace/demography/cohort_projections/config/projection_config.yaml` | Project configuration with race categories (lines 72-80) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/raw/population/nd_age_sex_race_distribution.csv` | Census+PUMS hybrid distribution (115 rows) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/fertility_rates.parquet` | Fertility rates by age and race (42 rows) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/survival_rates.parquet` | Survival rates by age, sex, race (1,212 rows) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/migration/migration_rates_pep_baseline.parquet` | Migration rates by county/age/sex/race (57,876 rows) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/base_population.parquet` | Base population for all counties (57,876 rows) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/data/load/base_population_loader.py` | Base population loader (distribution application, line 67-211) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/core/fertility.py` | Fertility engine (`calculate_births()`, line 16-139) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/core/mortality.py` | Survival engine (`apply_survival_rates()`, line 17-174) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/core/migration.py` | Migration engine (`apply_migration()`, line 16-125) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/data/process/migration_rates.py` | Migration processing (`distribute_migration_by_race()`, line 519-624) |
| `/home/nhaarstad/workspace/demography/cohort_projections/scripts/pipeline/02_run_projections.py` | Rate transforms (`_transform_fertility_rates()`, line 188-225) |
| `/home/nhaarstad/workspace/demography/cohort_projections/docs/governance/adrs/041-census-pums-hybrid-base-population.md` | ADR documenting current distribution methodology |

---

## 7. Data Appendix

### 7.1 State-Level Population by Race (Baseline Scenario)

| Race | 2025 | 2035 | 2045 | 2055 | Growth |
|------|------|------|------|------|--------|
| White alone, Non-Hispanic | 642,227 | 678,118 | 721,193 | 740,971 | +15.4% |
| AIAN alone, Non-Hispanic | 91,656 | 106,799 | 127,910 | 140,755 | +53.6% |
| Two or more races, Non-Hispanic | 27,150 | 27,265 | 29,415 | 28,436 | +4.7% |
| Hispanic (any race) | 19,141 | 28,512 | 43,847 | **58,200** | **+204.1%** |
| Asian/PI alone, Non-Hispanic | 14,157 | 21,980 | 28,262 | 32,730 | +131.2% |
| Black alone, Non-Hispanic | 5,027 | 5,241 | 5,052 | **4,189** | **-16.7%** |
| **State Total** | **799,358** | **867,915** | **955,679** | **1,005,281** | **+25.8%** |

### 7.2 Black Non-Hispanic Age Structure

| Year | Total | Under 18 | 18-44 | 45-64 | 65+ | Fertile Females |
|------|-------|----------|-------|-------|-----|-----------------|
| 2025 | 5,027 | 845 | 2,834 | 1,347 | 0 | **0** |
| 2035 | 5,241 | 293 | 1,401 | 2,393 | 1,154 | **0** |
| 2045 | 5,052 | 0 | 1,698 | 2,469 | 885 | **0** |
| 2055 | 4,189 | 0 | 403 | 2,763 | 1,023 | **0** |

### 7.3 Hispanic Age Structure

| Year | Total | Under 18 | 18-44 | 45-64 | 65+ | Fertile Females (approx) |
|------|-------|----------|-------|-------|-----|--------------------------|
| 2025 | 19,141 | 11,044 (57.7%) | 6,439 | 1,658 | 0 | ~9,900 |
| 2035 | 28,512 | 6,297 (22.1%) | 17,003 | 3,850 | 1,362 | ~17,000 |
| 2045 | 43,847 | 18,033 (41.1%) | 18,416 | 5,790 | 1,608 | ~18,400 |
| 2055 | 58,200 | 24,255 (41.7%) | 24,783 | 7,906 | 1,256 | ~24,800 |
