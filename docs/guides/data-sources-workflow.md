# Data Sources Workflow Guide

Complete guide for acquiring, updating, and managing data sources for the cohort projection system.

**Related**: [configuration-reference.md](./configuration-reference.md) | [AGENTS.md](../../AGENTS.md)

---

## Overview

The projection system requires five categories of data:

| Category | Purpose | Update Frequency |
|----------|---------|------------------|
| Geographic | County/place reference data | Annually (after Census releases) |
| Population | Base population by age/sex/race | Annually |
| Fertility | Age-specific fertility rates | Every 3-5 years |
| Mortality | Life tables / survival rates | Every 3-5 years |
| Migration | Domestic and international flows | Annually |

---

## Data Acquisition Methods

### 1. Local Sibling Repositories (Automated)

Use the fetch script to copy data from sibling repositories on your machine:

```bash
# See what data is available locally
python scripts/fetch_data.py --list

# Dry run (show what would be fetched)
python scripts/fetch_data.py --dry-run

# Fetch all available data
python scripts/fetch_data.py

# Fetch specific category only
python scripts/fetch_data.py --category geographic
python scripts/fetch_data.py --category population

# Force re-fetch existing files
python scripts/fetch_data.py --force
```

**Required sibling repositories:**

| Repository | Data Provided |
|------------|---------------|
| `~/projects/popest` | Census Population Estimates (county, state, cities) |
| `~/projects/ndx-econareas` | County reference data, CBSA crosswalks |
| `~/maps` | ACS PUMS microdata |

### 2. Manual Downloads (External Sources)

Some data must be downloaded manually from external sources.

---

## Data Source Details

### Geographic Data

#### North Dakota Counties (53 counties)

**Source**: Census Bureau Population Estimates Program
**URL**: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
**Local Path**: `data/raw/geographic/nd_counties.csv`

**Required columns**:
- `county_fips` - 5-digit FIPS code (e.g., "38101")
- `county_name` - County name (e.g., "Cass County")
- `state_fips` - State FIPS code ("38" for North Dakota)

**Update procedure**:
1. Download county estimates file for the latest year
2. Filter to STATE=38 (North Dakota)
3. Save to `data/raw/geographic/nd_counties.csv`

#### North Dakota Places (406 incorporated places)

**Source**: Census Bureau Population Estimates Program - Incorporated Places
**URL**: https://www.census.gov/data/tables/time-series/demo/popest/2020s-total-cities-and-towns.html
**Local Path**: `data/raw/geographic/nd_places.csv`

**Required columns**:
- `place_fips` - 7-digit FIPS code (e.g., "3825700" for Fargo)
- `place_name` - Place name (e.g., "Fargo city")
- `county_fips` - Containing county FIPS code
- `state_fips` - State FIPS code ("38")

**Update procedure**:
1. Download file `sub-est20XX_38.csv` (already filtered to North Dakota)
2. Ensure place-to-county relationships are included
3. Save to `data/raw/geographic/nd_places.csv`

---

### Population Data

#### County Population Estimates

**Source**: Census Bureau Population Estimates Program
**URL**: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html
**Local Path**: `data/raw/population/co-est20XX-alldata.csv`

**Required columns**:
- `STATE`, `COUNTY` - FIPS codes
- `STNAME`, `CTYNAME` - Names
- `POPESTIMATE20XX` - Population estimates
- `BIRTHS20XX`, `DEATHS20XX`, `NETMIG20XX` - Demographic components

**Update procedure**:
1. Download `co-est20XX-alldata.csv` from Census Bureau
2. Place in `data/raw/population/`
3. Run `python scripts/pipeline/01_process_demographic_data.py`

#### Age-Sex-Race Distribution (ACS PUMS)

**Source**: Census Bureau American Community Survey PUMS
**URL**: https://www.census.gov/programs-surveys/acs/microdata.html
**Local Path**: `data/raw/population/pums_person.parquet`

**Required columns**:
- `AGEP` - Age
- `SEX` - Sex (1=Male, 2=Female)
- `RAC1P` - Race
- `HISP` - Hispanic origin
- `PWGTP` - Person weight
- `ST` - State (filter to 38 for North Dakota)

**Update procedure**:
1. Download 5-year ACS PUMS person file
2. Filter to North Dakota (ST=38)
3. Apply person weights for population estimates
4. Process into age-sex-race distribution

---

### Fertility Data

#### SEER Age-Specific Fertility Rates

**Source**: SEER*Stat (Surveillance, Epidemiology, and End Results Program)
**URL**: https://seer.cancer.gov/seerstat/
**Local Path**: `data/raw/fertility/seer_asfr_2018_2022.csv`

**Required columns**:
- `age` - Single year of age (15-49)
- `race_ethnicity` - Race/ethnicity category
- `asfr` - Age-specific fertility rate
- `year` - Year

**Manual download instructions**:
1. Install SEER*Stat software from https://seer.cancer.gov/seerstat/
2. Request access to SEER*Stat Database
3. Query: Natality > U.S. > Age of Mother, Race/Ethnicity
4. Export as CSV with single-year age groups (15-49)

#### Alternative: CDC WONDER Natality

**Source**: CDC WONDER
**URL**: https://wonder.cdc.gov/natality.html
**Local Path**: `data/raw/fertility/cdc_nvss_fertility.csv`

**Download procedure**:
1. Go to https://wonder.cdc.gov/natality.html
2. Select years 2018-2022
3. Group by: Age of Mother (single years), Race/Ethnicity
4. Request births and population to calculate rates
5. Export and calculate: `asfr = births / (female_population * 1000)`

---

### Mortality Data

#### Life Tables (Survival Rates)

**Source**: CDC National Center for Health Statistics
**URL**: https://www.cdc.gov/nchs/products/life_tables.htm
**Local Path**: `data/raw/mortality/seer_lifetables_2020.csv`

**Required columns**:
- `age` - Age at start of interval
- `sex` - Sex
- `race_ethnicity` - Race/ethnicity category
- `qx` - Probability of death between age x and x+1
- `lx` - Number surviving to exact age x (radix of 100,000)

**Key formulas**:
- `survival_rate = 1 - qx`
- Or: `survival_rate = lx[x+1] / lx[x]`

**Download procedure**:
1. Go to https://www.cdc.gov/nchs/products/life_tables.htm
2. Download "United States Life Tables" for desired year
3. Extract life table values from PDF/Excel
4. Calculate survival rates

---

### Migration Data

#### IRS County-to-County Migration Flows

**Source**: IRS Statistics of Income - Migration Data
**URL**: https://www.irs.gov/statistics/soi-tax-stats-migration-data
**Local Path**: `data/raw/migration/irs_county_flows_2018_2022.csv`

**Required columns**:
- `y1_statefips`, `y1_countyfips` - Origin geography
- `y2_statefips`, `y2_countyfips` - Destination geography
- `n1` - Number of returns (approximate number of tax filers)
- `n2` - Number of exemptions (approximate persons)
- `AGI` - Adjusted gross income

**Download procedure**:
1. Go to https://www.irs.gov/statistics/soi-tax-stats-migration-data
2. Download county-to-county migration files for years 2018-2022
3. Filter to flows involving North Dakota (state FIPS 38)
4. Combine multiple years into single file

**Important notes**:
- IRS data covers only tax filers
- Adjust for non-filers using population coverage ratios
- Data typically released 18 months after tax year

#### International Migration (ACS)

**Source**: Census Bureau American Community Survey
**URL**: https://data.census.gov/table?t=Foreign+Born
**Local Path**: `data/raw/migration/acs_international_migration.csv`

**Download procedure**:
1. Go to data.census.gov
2. Query Table B05005: Place of Birth by Year of Entry
3. Filter to North Dakota geographies
4. Export to CSV

---

## Processing Pipeline

After acquiring raw data, run the processing pipeline:

```bash
# Step 1: Process demographic data
python scripts/pipeline/01_process_demographic_data.py

# Step 2: Run projections
python scripts/pipeline/02_run_projections.py

# Step 3: Export results
python scripts/pipeline/03_export_results.py
```

### Processed Data Outputs

| Raw Input | Processed Output |
|-----------|------------------|
| Population estimates | `data/processed/nd_county_population.csv` |
| ACS PUMS | `data/processed/nd_age_sex_race_distribution.csv` |
| SEER fertility | `data/processed/asfr_processed.csv` |
| CDC life tables | `data/processed/survival_rates_processed.csv` |
| IRS migration | `data/processed/nd_migration_processed.csv` |

---

## Data Validation

After processing, validate data integrity:

```bash
python scripts/validate_data.py
```

The validation script checks:
- Required columns present
- No missing values in key fields
- Population values are non-negative
- FIPS codes are properly formatted
- All 53 counties represented
- Rates are within plausible ranges

---

## Data Update Schedule

| Data Type | Source | Typical Release | Recommended Update |
|-----------|--------|-----------------|-------------------|
| Population Estimates | Census Bureau | May (prior year) | Annually |
| ACS PUMS | Census Bureau | December | Every 5 years |
| Fertility Rates | SEER/CDC | Varies | Every 3-5 years |
| Life Tables | CDC NCHS | Annually | Every 3-5 years |
| IRS Migration | IRS SOI | Fall (2 years prior) | Annually |

---

## Troubleshooting

### Common Issues

**"Data file not found"**
- Run `python scripts/fetch_data.py --list` to see what's available
- Check sibling repository paths in `config/data_sources.yaml`

**"Missing required columns"**
- Verify column names match what's expected
- Check `config/data_sources.yaml` for required column names
- Census Bureau may have changed column naming conventions

**"FIPS code validation failed"**
- Ensure FIPS codes are strings, not integers
- State FIPS should be 2 digits (e.g., "38" not "38.0")
- County FIPS should be 5 digits (e.g., "38101" not "38101.0")

---

## Contact Information

### Data Provider Resources

| Provider | Contact | Resources |
|----------|---------|-----------|
| Census Bureau | ask.census.gov | Data webinars, API documentation |
| CDC NCHS | nchs@cdc.gov | Life tables methodology |
| IRS SOI | irs.gov/statistics | Migration data methodology |
| SEER | seer.cancer.gov | SEER*Stat user guides |

### Internal Resources

- Configuration: `config/data_sources.yaml`
- ADR: `docs/governance/adrs/016-raw-data-management-strategy.md`
- Sibling repos: See fetch script comments

---

*Last Updated: 2026-02-02*
