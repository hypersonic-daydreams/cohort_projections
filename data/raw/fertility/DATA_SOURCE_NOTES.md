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

## North Dakota-Specific Considerations

1. **State rates vs. National rates:**
   - ND-specific rates are ideal but rarely available by race
   - National rates are acceptable per ADR-001
   - Consider adjustment factors based on ND teen birth rate data

2. **AIAN Population:**
   - AIAN is 5% of ND population (significant)
   - May have different fertility patterns than national AIAN
   - Consider using Northern Plains AIAN rates if available from IHS data

3. **Small Population Adjustment:**
   - Per ADR-001, use 5-year averaging for stability
   - Population-weighted averaging preferred

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
