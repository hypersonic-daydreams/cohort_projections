# Data Validation Report
## North Dakota Cohort Projection System

**Generated:** 2026-02-02 18:58:13
**Validator:** `scripts/validate_data.py`

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Checks | 34 |
| Passed | 34 (100.0%) |
| Failed | 0 (0.0%) |
| Informational | 0 (0.0%) |

**Overall Status:** PASS - All critical checks passed

---

## File-by-File Validation

### 1. Fertility Data (`data/raw/fertility/asfr_processed.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Required columns | ['age', 'race_ethnicity', 'asfr', 'year'] | ['age', 'race_ethnicity', 'asfr', 'year'] | PASS |
| Row count | 42 | 42 | PASS |
| Age groups count | 7 | 7 | PASS |
| Race categories count | 6 | 6 | PASS |
| ASFR values range (0-200) | 0-200 | 0.4-112.3 | PASS |

**Notes:** ['total', 'white_nh', 'black_nh', 'hispanic', 'aian_nh', 'asian_nh']

---

### 2. Mortality Data (`data/raw/mortality/survival_rates_processed.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Required columns | ['age', 'sex', 'race_ethnicity', 'qx', 'survival_rate', 'lx'] | ['age', 'sex', 'race_ethnicity', 'qx', 'survival_rate', 'lx'] | PASS |
| Row count | 1212 | 1212 | PASS |
| Age range count | 101 | 101 | PASS |
| Sex categories | 2 | 2 | PASS |
| Race categories count | 6 | 6 | PASS |
| qx range (0-1) | 0-1 | 0.000040-1.000000 | PASS |
| survival_rate = 1 - qx | < 0.0001 | 0.00000000 | PASS |

**Notes:** Contains survival rates for ages 0-100, both sexes, and 6 race/ethnicity categories.

---

### 3. Migration Data (`data/raw/migration/nd_migration_processed.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Required columns | ['county_fips', 'county_name', 'year', 'inflow_n2', 'outflow_n2', 'net_migration'] | ['county_fips', 'county_name', 'year', 'inflow_n2', 'outflow_n2', 'net_migration', 'inflow_domestic', 'outflow_domestic'] | PASS |
| Row count | 212 | 212 | PASS |
| County count | 53 | 53 | PASS |
| Year count | 4 | 4 | PASS |
| FIPS starts with 38 | All | 212/212 | PASS |
| net_migration = inflow - outflow | < 1 | 0 | PASS |

**Notes:** Contains IRS migration data for years 2019-2022.

---

### 4. County Population Data (`data/raw/population/nd_county_population.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Required columns | ['county_fips', 'county_name', 'population_2024'] | ['county_fips', 'county_name', 'population_2024', 'births_2024', 'deaths_2024', 'net_migration_2024'] | PASS |
| Row count | 53 | 53 | PASS |
| Total population (~796K) | 750,000-850,000 | 796,568 | PASS |
| FIPS starts with 38 | All | 53/53 | PASS |
| No duplicate counties | 0 | 0 | PASS |

**Notes:** 2024 population estimates for all 53 North Dakota counties.

---

### 5. Population Distribution Data (`data/raw/population/nd_age_sex_race_distribution.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Required columns | ['age_group', 'sex', 'race_ethnicity', 'proportion'] | ['age_group', 'sex', 'race_ethnicity', 'estimated_count', 'proportion'] | PASS |
| Proportions sum to 1.0 | 0.99-1.01 | 1.000000 | PASS |
| Age groups present | >= 15 | 18 | PASS |
| Sex categories | 2 | 2 | PASS |
| Race categories present | >= 4 | 8 | PASS |
| Proportions range (0-1) | 0-1 | 0.000081-0.049361 | PASS |

**Notes:** Proportional distribution of population by age, sex, and race/ethnicity.

---

### 6. Geographic Data

#### 6a. Counties (`data/raw/geographic/nd_counties.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| ND counties present | 53 | 53 | PASS |
| Total rows (national) | >= 3000 | 3144 | PASS |

**Notes:** National county file with population estimates. Filter on state_fips=38 for ND.

#### 6b. Places (`data/raw/geographic/nd_places.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| ND places count | >= 300 | 2896 | PASS |

**Notes:** North Dakota cities, towns, and census-designated places.

#### 6c. Metro Crosswalk (`data/raw/geographic/metro_crosswalk.csv`)

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Total entries | >= 1000 | 1915 | PASS |
| ND counties in crosswalk | >= 0 | 14 | PASS |

**Notes:** National CBSA/metro area crosswalk file.

---

## Data Quality Summary

### Data Completeness

| Data Category | Status | Completeness |
|---------------|--------|--------------|
| Fertility (ASFR) | Complete | 7 age groups x 6 race categories |
| Mortality (Life Tables) | Complete | 101 ages x 2 sexes x 6 races |
| Migration (IRS) | Complete | 53 counties x 4 years |
| Population (County) | Complete | All 53 ND counties |
| Population (Distribution) | Complete | Age/sex/race proportions |
| Geographic | Complete | Counties, places, metro areas |

### Issues Found

No critical issues found.

---

## Recommendations

1. All validations passed. Data is ready for projection modeling.
2. Consider implementing automated validation as part of data pipeline.
3. Document any manual adjustments made to source data.

---

## Technical Details

### File Locations

| File | Path |
|------|------|
| Fertility | `data/raw/fertility/asfr_processed.csv` |
| Mortality | `data/raw/mortality/survival_rates_processed.csv` |
| Migration | `data/raw/migration/nd_migration_processed.csv` |
| County Population | `data/raw/population/nd_county_population.csv` |
| Population Distribution | `data/raw/population/nd_age_sex_race_distribution.csv` |
| Counties | `data/raw/geographic/nd_counties.csv` |
| Places | `data/raw/geographic/nd_places.csv` |
| Metro Crosswalk | `data/raw/geographic/metro_crosswalk.csv` |

### Data Sources

- **Fertility:** CDC NCHS Natality Data (WONDER)
- **Mortality:** CDC NCHS National Vital Statistics System
- **Migration:** IRS Statistics of Income (SOI) Migration Data
- **Population:** US Census Bureau Population Estimates Program
- **Geographic:** US Census Bureau TIGER/Line and OMB CBSA definitions

---

*Report generated by the North Dakota Cohort Projection System validation script.*
