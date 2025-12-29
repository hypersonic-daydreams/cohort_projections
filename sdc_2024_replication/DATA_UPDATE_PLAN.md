# Data Update Plan: SDC Methodology with Updated Data

**Purpose:** Document available updated data sources for creating an "SDC methodology with updated data" projection variant.

**Date:** 2025-12-28

**Concept:** Use the SDC 2024 methodology (cohort-component with 5-year intervals, 60% Bakken dampening, etc.) but substitute updated data sources where available.

---

## 1. Base Population

### SDC 2024 Used
- **Census 2020 (April 1)** as base population
- County-level by 5-year age groups and sex
- 53 North Dakota counties

### Updated Data Available

#### Census Vintage 2024 Population Estimates
- **Release Date:** June 26, 2025
- **Coverage:** County-level populations by age, sex, race, and Hispanic origin through July 1, 2024
- **URL:** https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
- **Format:** CSV/Excel files available for download

#### Key Differences from SDC 2024
| Aspect | SDC 2024 | Updated Option |
|--------|----------|----------------|
| Base Year | 2020 (Census) | 2024 (Vintage 2024 estimates) |
| Reference Date | April 1, 2020 | July 1, 2024 |
| Years of Actual Data | 0 (starts at Census) | 4 years of post-Census estimates |

#### Benefits of Using Vintage 2024
1. **Captures post-COVID reality**: The 2020 Census was conducted during COVID-19, and subsequent estimates reflect population adjustments
2. **Incorporates actual 2020-2024 trends**: Rather than projecting from 2020, we start with known 2024 population
3. **Aligns with ND's record population**: Vintage 2024 shows ND at 796,568 (vs. 779,094 in 2020 Census)

#### Files to Obtain
- `cc-est2024-agesex-38.csv` - County characteristics by age and sex for North Dakota (FIPS 38)
- State-level totals for validation

---

## 2. Fertility Rates

### SDC 2024 Used
- **Source:** ND DHHS Vital Statistics (2018-2022 births)
- **National Blend:** CDC NVSS national rates (nvsr72-01.pdf, January 2023)
- Rates by 5-year age groups of mother (10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49)
- County-level with state/national blending for stability

### Updated Data Available

#### CDC WONDER Natality Data (2023 Final)
- **Release Date:** Final 2023 data available as of mid-2025
- **URL:** https://wonder.cdc.gov/natality-current.html
- **Coverage:** 2007-2024 (provisional 2024, final through 2023)
- **State-level:** Full age-specific fertility rates available
- **County-level:** Only for counties with 100,000+ population (excludes most ND counties)

#### NCHS Vital Statistics Reports
- **Final 2023 Data:** National Vital Statistics Reports, Volume 74, Number 1 (March 2025)
- **URL:** https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-1.pdf
- **Contains:** National and state-level age-specific birth rates for 2023
- **State Totals:** North Dakota had 9,557 births in 2022; 2023 data available

#### 2024 Provisional Data
- **Source:** Vital Statistics Rapid Release
- **URL:** https://www.cdc.gov/nchs/data/vsrr/vsrr038.pdf
- **Note:** Provisional only - may be revised

#### Key Differences from SDC 2024
| Aspect | SDC 2024 | Updated Option |
|--------|----------|----------------|
| Birth Years | 2018-2022 | 2020-2024 or 2021-2023 |
| National Reference | 2021 NVSS | 2023 NVSS |
| Trend Captured | Pre/early COVID | Full COVID + recovery |

#### North Dakota-Specific Data
To replicate SDC's county-level blending approach, we would need:
1. **ND DHHS Vital Statistics:** Request 2020-2024 birth data by county and mother's age (similar to "2018-2022 ND Res Birth for Kevin Iverson.xlsx")
2. **County female population:** From Vintage 2024 population estimates

#### Files to Obtain
- CDC WONDER query: ND births by mother's age group, 2020-2023
- NVSS state-level fertility rates for blending
- ND DHHS request: County-level birth data 2020-2024 (if available)

---

## 3. Survival Rates

### SDC 2024 Used
- **Source:** CDC NCHS Life Tables for North Dakota, 2020
- **Publication:** NVSR 71-02
- **Files:** ND1.xlsx (Total), ND2.xlsx (Males), ND3.xlsx (Females)
- Sex-specific, single-year age to 5-year survival probability conversion

### Updated Data Available

#### U.S. State Life Tables, 2022
- **Release Date:** December 4, 2025
- **Publication:** NVSR Volume 74, Number 12
- **URL:** https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-12.pdf
- **Supplemental Files:** https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/74-12/
- **Contains:**
  - ND-1: Life table for total population, North Dakota, 2022
  - ND-2: Life table for males, North Dakota, 2022
  - ND-3: Life table for females, North Dakota, 2022

#### U.S. National Life Tables, 2023
- **Release Date:** July 15, 2025
- **Publication:** NVSR Volume 74, Number 6
- **URL:** https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-06.pdf
- **Note:** National only - state-specific 2023 tables not yet available

#### Key Differences from SDC 2024
| Aspect | SDC 2024 | Updated Option |
|--------|----------|----------------|
| Life Table Year | 2020 | 2022 (state) or 2023 (national) |
| Publication | NVSR 71-02 | NVSR 74-12 (state 2022) |
| COVID Impact | Pre-COVID mortality | Full COVID impact + recovery |

#### Life Expectancy Comparison
| Measure | ND 2020 | U.S. 2022 | U.S. 2023 |
|---------|---------|-----------|-----------|
| Total | 76.9 years | 77.5 years | 78.4 years |
| Male | 74.2 years | ~74.8 years | 75.8 years |
| Female | 80.0 years | ~80.2 years | 81.1 years |

Note: 2022 and 2023 show mortality improvement from COVID-era lows.

#### Files to Obtain
- ND State Life Tables 2022: Download from NVSR 74-12 supplemental files
- Excel files: ND-1.xlsx, ND-2.xlsx, ND-3.xlsx (2022 versions)

---

## 4. Migration Rates

### SDC 2024 Used
- **Method:** Census residual (Component Method II)
- **Time Period:** 2000-2020 (four 5-year periods averaged)
- **Adjustment:** 60% "Bakken dampening" plus period-specific multipliers (0.2 to 0.7)
- County-level by age group and sex

### Migration Rate Options

#### Option A: Keep SDC's 2000-2020 Rates (Recommended for Replication)
- **Rationale:** The SDC methodology explicitly uses historical 2000-2020 migration patterns
- **Advantage:** True methodological replication - only updates other components
- **Data Required:** Use existing extracted rates from `data/processed/sdc_2024/migration_rates_sdc_2024.csv`

#### Option B: Extend to 2000-2024 (Census Residual Method)
- **Rationale:** Incorporate more recent migration experience
- **Data Required:**
  - Census 2020 population by county/age/sex (have this)
  - Vintage 2024 population estimates by county/age/sex (available)
  - Calculate 2020-2024 residual migration and average with 2000-2020 periods
- **Challenge:** Would need to calculate new period migration rates

#### Option C: Use IRS SOI Data (Different Methodology)
- **Rationale:** Administrative data provides direct migration measurement
- **Latest Available:** 2021-2022 filing year (county-to-county flows)
- **URL:** https://www.irs.gov/statistics/soi-tax-stats-migration-data-2021-2022
- **Note:** This is a different methodology than Census residual - not a true SDC replication

### Key Migration Data Status
| Source | Latest Available | Coverage |
|--------|-----------------|----------|
| IRS SOI | 2021-2022 | County-to-county flows |
| Census ACS | 2018-2022 (5-year) | State-to-county flows |
| Census Residual | 2020-2024 (calculable) | Requires population estimates |

### Recommendation for "SDC with Updated Data"

**Use Option A (SDC's original rates)** for these reasons:
1. Migration rates are the most controversial/uncertain component
2. SDC's rates already incorporate 2015-2020 (partial post-Bakken period)
3. The 60% dampening was SDC's judgment call - hard to replicate their reasoning
4. Changing migration rates would fundamentally change results, obscuring the impact of other updates

The "SDC methodology with updated data" variant should show what happens when we update fertility, mortality, and base population while keeping SDC's migration assumptions constant.

---

## 5. Summary: Data Update Matrix

| Component | SDC 2024 Data | Updated Data Available | Update Priority |
|-----------|---------------|------------------------|-----------------|
| **Base Population** | Census 2020 | Vintage 2024 (July 2024) | HIGH - 4 years newer |
| **Fertility Rates** | 2018-2022 ND births | 2020-2023/2024 NCHS/WONDER | MEDIUM - 2 years newer |
| **Survival Rates** | 2020 ND Life Tables | 2022 ND Life Tables | MEDIUM - 2 years newer |
| **Migration Rates** | 2000-2020 residual | Keep original | LOW - preserve methodology |

---

## 6. Data Collection Steps

### Step 1: Base Population (Vintage 2024)
1. Download from Census: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
2. Filter for North Dakota (FIPS 38)
3. Extract county-level population by 5-year age groups and sex
4. Save as `base_population_vintage_2024.csv`

### Step 2: Survival Rates (2022 Life Tables)
1. Download ND state life tables from: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/74-12/
2. Extract survival rates by age and sex
3. Convert to 5-year survival probabilities (same methodology as SDC)
4. Save as `survival_rates_nd_2022.csv`

### Step 3: Fertility Rates (2020-2023)
1. Query CDC WONDER for ND births by mother's age (2020-2023)
2. Obtain female population by age from Vintage 2024 estimates
3. Calculate age-specific fertility rates
4. Blend with national rates per SDC methodology
5. Save as `fertility_rates_nd_2020_2023.csv`

### Step 4: Migration Rates
1. Use existing SDC rates: `data/processed/sdc_2024/migration_rates_sdc_2024.csv`
2. Apply same 60% dampening and period multipliers
3. No additional data collection needed

---

## 7. Expected Impact of Updates

### Base Population Impact
- **SDC 2024:** Starts at 779,094 (Census 2020)
- **Updated:** Starts at ~796,568 (Vintage 2024)
- **Effect:** Higher starting point (+17,474) will propagate through all projection years

### Survival Rate Impact
- **2020 ND Life Tables:** Reflect COVID-era mortality
- **2022 ND Life Tables:** Reflect mortality improvement post-COVID
- **Effect:** Slightly more survivors per cohort -> modestly higher population

### Fertility Rate Impact
- **2018-2022:** Captures pre-COVID and early-COVID fertility
- **2020-2023:** Captures full COVID impact and any recovery
- **National trends:** Continued fertility decline (~2% decline from 2022 to 2023)
- **Effect:** Likely slightly lower births -> modestly lower population

### Net Effect
The updates should produce projections that:
1. Start ~17,000 higher due to updated base population
2. Show slightly more favorable mortality (2022 vs 2020 life tables)
3. Show slightly lower fertility (2020-2023 vs 2018-2022)
4. Still project growth (due to SDC's in-migration assumptions)

The "SDC methodology with updated data" will likely project population between our baseline (decline) and the original SDC 2024 (strong growth), because:
- Higher base population than SDC 2024
- Same positive migration assumptions as SDC 2024
- Updated demographic rates

---

## 8. File Naming Convention

For the updated data variant:

```
sdc_2024_replication/
├── data/
│   ├── base_population_by_county.csv          # Original (Census 2020)
│   ├── base_population_vintage_2024.csv       # Updated (Vintage 2024)
│   ├── fertility_rates_by_county.csv          # Original (2018-2022)
│   ├── fertility_rates_2020_2023.csv          # Updated (2020-2023)
│   ├── survival_rates_by_county.csv           # Original (2020 Life Tables)
│   ├── survival_rates_nd_2022.csv             # Updated (2022 Life Tables)
│   └── migration_rates_by_county.csv          # Keep original (SDC 2000-2020)
├── output/
│   ├── sdc_replication_population.csv         # Original replication
│   └── sdc_updated_data_population.csv        # Updated data variant
└── DATA_UPDATE_PLAN.md                        # This document
```

---

## 9. References

### Census Bureau
- Vintage 2024 County Characteristics: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html
- Population Estimates Press Release: https://www.census.gov/newsroom/press-releases/2025/population-estimates-age-sex.html

### CDC/NCHS
- State Life Tables 2022: https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-12.pdf
- National Life Tables 2023: https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-06.pdf
- CDC WONDER Natality: https://wonder.cdc.gov/natality-current.html
- NVSS Birth Data: https://www.cdc.gov/nchs/nvss/births.htm

### IRS
- SOI Migration Data: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- 2021-2022 Migration Data: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2021-2022

### North Dakota
- ND State Data Center: https://www.ndsu.edu/sdc/
- ND DHHS Vital Statistics: https://www.health.nd.gov/vital

---

## 10. Next Steps

1. **Review this plan** - Confirm approach with project requirements
2. **Download Census Vintage 2024** - Base population data
3. **Download CDC 2022 Life Tables** - ND-specific survival rates
4. **Query CDC WONDER** - 2020-2023 birth data for ND
5. **Create updated data files** - Transform to SDC format
6. **Run updated projection** - Using modified data inputs
7. **Compare results** - Document impact of data updates

---

*This document is RESEARCH ONLY. No data has been downloaded yet.*
