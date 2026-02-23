# Fertility Rate Data Sources

## Research Date
2025-12-28

## Project Requirements

The cohort_projections project requires Age-Specific Fertility Rates (ASFR) by race/ethnicity for North Dakota population projections.

**Required Format:**
- age: 15-49 (single years or 5-year groups)
- race_ethnicity: 6 categories per ADR-007:
  1. White alone, Non-Hispanic
  2. Black alone, Non-Hispanic
  3. AIAN alone, Non-Hispanic (American Indian/Alaska Native)
  4. Asian/PI alone, Non-Hispanic (Asian/Pacific Islander)
  5. Two or more races, Non-Hispanic
  6. Hispanic (any race)
- asfr: births per 1,000 women
- year: Most recent available (2020-2023 preferred)

---

## Downloaded Data Files

### 1. cdc_birth_rates_race_age.csv (PRIMARY - RECOMMENDED)

**Source:** CDC NCHS Vital Statistics Rapid Release - Natality Dashboard
**URL:** https://data.cdc.gov/api/views/76vv-a7x8/rows.csv?accessType=DOWNLOAD
**Downloaded:** 2025-12-28

**Content:**
- Quarterly age-specific birth rates from 2023 Q1 to 2025 Q2
- Age groups: 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45+
- Rates per 1,000 women

**Race/Ethnicity Categories Available:**
- All races and origins
- Hispanic
- Non-Hispanic Black
- Non-Hispanic White

**GAPS:** Missing AIAN, Asian/PI, and Two or More Races categories

**Column Mapping:**
| Source Column | Target Column |
|---------------|---------------|
| Year and Quarter | year (extract year) |
| Indicator | age |
| Race Ethnicity Category | race_ethnicity |
| Rate | asfr |

**Preprocessing Required:**
1. Filter for `Topic Subgroup == "Age-specific Birth Rates"`
2. Extract year from "Year and Quarter" column
3. Calculate annual averages from quarterly data
4. Map race categories to project standard

**Sample Data:**
```
Year and Quarter,Topic,Topic Subgroup,Indicator,Race Ethnicity Category,Rate
2025 Q2,Birth Rates,Age-specific Birth Rates,25-29 years,Non-Hispanic White,89.10
2025 Q2,Birth Rates,Age-specific Birth Rates,25-29 years,Non-Hispanic Black,79.00
2025 Q2,Birth Rates,Age-specific Birth Rates,25-29 years,Hispanic,110.20
```

---

### 2. cdc_birth_rates_state_age.csv

**Source:** CDC/NCHS
**URL:** https://data.cdc.gov/api/views/y268-sna3/rows.csv?accessType=DOWNLOAD
**Downloaded:** 2025-12-28

**Content:**
- State-level teen birth rates (15-17, 18-19 year age groups only)
- Years 1990-2019
- Includes North Dakota data

**Limitation:** Only teen age groups (15-19), not full reproductive age range

**Use Case:** North Dakota-specific teen fertility adjustment

---

### 3. cohort_table01.csv, cohort_table10_white.csv, cohort_table11_black.csv

**Source:** CDC NCHS NVSS Cohort Fertility Tables
**URL:** https://ftp.cdc.gov/pub/Health_Statistics/NCHS/nvss/birth/cohort/
**Downloaded:** 2025-12-28

**Content:**
- Historical cohort fertility data (1960-2005)
- Central birth rates by age, race, and birth order
- Single-year ages 14-49

**Race Categories:**
- Table01: All races combined
- Table10: White women only
- Table11: Black women only

**Limitation:** Historical data only (ends in 2005), not current rates

---

### 4. nchs_births_fertility_rates.csv

**Source:** NCHS Births and General Fertility Rates
**URL:** https://data.cdc.gov/api/views/e6fc-ccez/rows.csv?accessType=DOWNLOAD
**Downloaded:** 2025-12-28

**Content:**
- Annual national births and general fertility rates since 1909
- Columns: Year, Birth Number, General Fertility Rate, Crude Birth Rate

**Limitation:** No age or race breakdown - aggregate rates only

---

### 5. nvsr73-02_births_2022.pdf, nvsr74-01_births_2023.pdf

**Source:** National Vital Statistics Reports
**URLs:**
- https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-02.pdf (2022 Final Data)
- https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-1.pdf (2023 Final Data)
**Downloaded:** 2025-12-28

**Content:**
- Comprehensive tables with age-specific birth rates by race/ethnicity
- Includes all required race categories (White, Black, AIAN, Asian, NHPI, Hispanic)
- Table 2: "Birth rates, by age of mother: United States, 2010-2022/2023, and by age and race and Hispanic origin of mother"

**Format:** PDF - requires manual extraction or table parsing

**Key Tables to Extract:**
- Table 2: Age-specific birth rates by race/ethnicity (5-year age groups)
- Table 11: Total fertility rates by race/ethnicity

---

## Data Access Methods

### CDC WONDER (RECOMMENDED for complete data)

**URL:** https://wonder.cdc.gov/natality-current.html

**Query Parameters for ASFR by Race/Ethnicity:**
1. Group Results By: Year, Mother's Single Race 6, Mother's Age
2. Year: 2020-2023 (select range)
3. Single Race 6 categories:
   - American Indian or Alaska Native
   - Asian
   - Black or African American
   - Native Hawaiian or Other Pacific Islander
   - White
   - More than one race
4. Calculate rates: births per 1,000 population

**Race Mapping to Project Categories:**
| CDC WONDER | Project Category |
|------------|------------------|
| White + Non-Hispanic | White alone, Non-Hispanic |
| Black or African American + Non-Hispanic | Black alone, Non-Hispanic |
| American Indian or Alaska Native + Non-Hispanic | AIAN alone, Non-Hispanic |
| Asian + Non-Hispanic | Asian/PI alone, Non-Hispanic |
| Native Hawaiian or Other Pacific Islander + Non-Hispanic | Asian/PI alone, Non-Hispanic |
| More than one race + Non-Hispanic | Two or more races, Non-Hispanic |
| Hispanic ethnicity (any race) | Hispanic (any race) |

**Export:** Results can be exported as tab-delimited text

**Note:** CDC WONDER requires agreeing to data use terms and manual query execution. No direct API available for bulk download.

---

### Human Fertility Database (Alternative)

**URL:** https://www.humanfertility.org/Country/Country?cntr=USA

**Available Data:**
- Age-specific fertility rates (ASFR) 1933-2023
- Single-year ages
- Period and cohort data
- High quality, consistent methodology

**Limitation:** No race/ethnicity breakdown - total population only

**Data Files (require registration):**
- USA.zip: All available US data
- asfr.zip: ASFR data for all countries

---

## Recommended Data Acquisition Strategy

### Option A: Manual CDC WONDER Query (Most Complete)

1. Navigate to https://wonder.cdc.gov/natality-expanded-current.html
2. Query age-specific birth rates by:
   - Year: 2020-2023
   - Mother's Age: 5-year groups (15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49)
   - Mother's Single Race 6: All categories
   - Mother's Hispanic Origin: Hispanic, Non-Hispanic
3. Export results
4. Process to combine NHPI with Asian for Asian/PI category
5. Save as processed CSV

### Option B: Extract from NVSR PDFs (Official Published Rates)

1. Extract Table 2 from nvsr74-01_births_2023.pdf
2. Parse using Python (tabula-py or camelot)
3. Map to project race categories
4. Interpolate 5-year age groups to single years if needed

### Option C: Use Downloaded CDC API Data + Supplement (Quickest)

1. Use cdc_birth_rates_race_age.csv for White, Black, Hispanic rates
2. Supplement AIAN and Asian/PI rates from:
   - NVSR PDF tables
   - CDC WONDER query
   - Or national averages as proxy

---

## AIAN and Asian/PI Rate Estimates

For missing race categories, these national 2022 rates can be used as proxies:

### American Indian/Alaska Native (AIAN)
| Age Group | ASFR per 1,000 |
|-----------|----------------|
| 15-19 | 26.6 |
| 20-24 | 72.0 |
| 25-29 | 85.2 |
| 30-34 | 63.8 |
| 35-39 | 29.7 |
| 40-44 | 6.7 |
| 45-49 | 0.4 |

### Asian/Pacific Islander
| Age Group | ASFR per 1,000 |
|-----------|----------------|
| 15-19 | 3.4 |
| 20-24 | 26.1 |
| 25-29 | 72.8 |
| 30-34 | 111.0 |
| 35-39 | 65.8 |
| 40-44 | 16.5 |
| 45-49 | 1.3 |

*Source: National Vital Statistics Reports, Births: Final Data for 2022*

---

## North Dakota-Specific Data (ADR-053, Implemented 2026-02-23)

### ND CDC WONDER Birth Files

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| cdc_wonder_nd_births_2020_2023.txt | ND births by race×age×Hispanic, 2020-2023 pooled | 74 | 2026-02-23 |
| cdc_wonder_national_births_2020_2023.txt | National births, same dimensions (fallback) | 151 | 2026-02-23 |

**CDC WONDER Query Parameters:**
- Database: Natality, 2016-2024 (Expanded)
- Group By: Mother's Single Race 6, Mother's Hispanic Origin, Mother's Age 9
- State: North Dakota (ND file) / All states (national file)
- Years: 2020-2023 combined (not grouped by year — pooled for stability)
- Format: Tab-delimited text export

**Race Mapping (CDC WONDER → Project):**
| Single Race 6 | Hispanic Origin | Project Code |
|----------------|-----------------|-------------|
| White | Not Hispanic or Latino | white_nh |
| Black or African American | Not Hispanic or Latino | black_nh |
| American Indian or Alaska Native | Not Hispanic or Latino | aian_nh |
| Asian | Not Hispanic or Latino | asian_nh |
| Native Hawaiian or Other Pacific Islander | Not Hispanic or Latino | asian_nh (combined) |
| More than one race | Not Hispanic or Latino | two_or_more_nh |
| Any race | Hispanic or Latino | hispanic |

**Handling "Unknown or Not Stated" Hispanic Origin:**
CDC WONDER reports births where Hispanic origin is unknown. These are distributed
proportionally to the known Hispanic/Non-Hispanic split within each race × age cell.
This avoids undercount bias (dropping them) or overcount bias (assigning all to one group).

### ND ASFR Output File

**nd_asfr_processed.csv** — ND-specific age-specific fertility rates

| Column | Type | Description |
|--------|------|-------------|
| age | string | 5-year age group (15-19, 20-24, ..., 45-49) |
| race_ethnicity | string | Race code (total, white_nh, black_nh, hispanic, aian_nh, asian_nh, two_or_more_nh) |
| asfr | float | Births per 1,000 women per year |
| year | int | Reference year (2023) |

- 49 rows (7 age groups × 7 race categories)
- Format matches asfr_processed.csv schema for pipeline compatibility
- Referenced in config: `pipeline.fertility.input_file: "data/raw/fertility/nd_asfr_processed.csv"`

**Population Denominators:**
Female population by age × race from `data/raw/population/cc-est2024-alldata-38.csv`
(Census County Characteristics, FIPS 38). Summed across all 53 ND counties and
years 2-5 (July 2020 through July 2023) to match birth pooling window.

**Suppressed Cells:**
5 of 49 cells had <10 births across the 4-year window and use national ASFR
as fallback: Asian NH 15-19, and several 45-49 age group cells.

**Processing Script:** `scripts/data/build_nd_fertility_rates.py`

### Validation

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ND TFR (total) | 1.863 | 1.85-1.90 | Within 1% |
| Average annual births | 9,804 | ~9,647 (actual 2023) | Within 1.6% |
| ND/National TFR ratio | 1.15 | ~1.14-1.17 per GFR data | Consistent |

## Historical Notes

Prior to ADR-053 (2026-02-23), the project used national ASFR from
`asfr_processed.csv` (SEER-derived, 2024 vintage). That file remains in the
repository as a reference and fallback source. The switch to ND-specific rates
increased projected births by ~15%, correcting a systematic undercount.

---

## Data Quality Notes

1. **Race/Ethnicity Categories:**
   - CDC data uses Hispanic as ethnic category, crossing with race
   - Project treats Hispanic as mutually exclusive category per ADR-007
   - Non-Hispanic extraction required from CDC data

2. **Age Group Interpolation:**
   - 5-year age groups standard in source data
   - Single-year interpolation may be needed
   - Use Beers interpolation or graduation methods

3. **Rate Units:**
   - Downloaded data uses births per 1,000 women
   - Matches project requirements (no conversion needed)

4. **Data Currency:**
   - CDC quarterly data current through Q2 2025
   - NVSR annual reports available through 2023 final data

---

## References

1. **CDC NCHS NVSS Birth Data:** https://www.cdc.gov/nchs/nvss/births.htm
2. **CDC WONDER Natality:** https://wonder.cdc.gov/natality.html
3. **NCHS Data Visualization - Natality Trends:** https://www.cdc.gov/nchs/data-visualization/natality-trends/
4. **Human Fertility Database:** https://www.humanfertility.org/
5. **NVSR Births Final Data 2023:** https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-1.pdf
6. **NVSR Births Final Data 2022:** https://www.cdc.gov/nchs/data/nvsr/nvsr73/nvsr73-02.pdf
7. **CDC Cohort Fertility Tables:** https://www.cdc.gov/nchs/nvss/cohort_fertility_tables.htm
