# Data Directory

This directory contains the input data, processed rates, and projection outputs for the North Dakota Population Projection System. **Raw data files are NOT included in the GitHub repository** due to size and licensing considerations.

---

## Directory Structure

```
data/
├── raw/              # Raw input data from external sources (NOT in git)
│   ├── fertility/    # SEER fertility rate data
│   ├── mortality/    # SEER/CDC life tables
│   ├── migration/    # IRS migration flows + international migration
│   └── geographic/   # Geographic reference data (counties, places)
├── processed/        # Processed demographic rates (NOT in git)
│   ├── fertility/    # Processed fertility rate tables
│   ├── mortality/    # Processed survival rate tables
│   ├── migration/    # Processed migration rate tables
│   └── validation/   # Validation reports
├── projections/      # Projection outputs (NOT in git)
│   ├── state/        # State-level projections
│   ├── county/       # County-level projections
│   ├── places/       # Place-level projections
│   └── scenarios/    # Alternative scenario outputs
└── metadata/         # Processing metadata and data dictionaries
```

---

## Required Data Sources

### 1. Fertility Rates (SEER)

**Source**: SEER (Surveillance, Epidemiology, and End Results Program)

**What you need**: Age-specific fertility rates by race for North Dakota

**Where to get it**:
- **SEER*Stat Software**: https://seer.cancer.gov/seerstat/
- **Download**: https://seer.cancer.gov/seerstat/download/
- **Data source**: US Fertility Data

**Steps to obtain**:
1. Download and install SEER*Stat (free registration required)
2. Request access to US Fertility data files
3. Extract fertility rates for:
   - Geography: North Dakota (FIPS 38)
   - Years: 2018-2022 (5-year average for stability)
   - Age groups: Single-year ages 15-49
   - Race categories: All available (will be harmonized to 6 categories)

**Expected format**: CSV or TXT file with columns:
- Year
- State/County FIPS
- Age
- Race code
- Births
- Female population
- Fertility rate (births per 1,000 women)

**Save to**: `data/raw/fertility/seer_asfr_2018_2022.csv`

**Alternative sources**:
- **CDC WONDER Natality**: https://wonder.cdc.gov/natality.html
  - More accessible but less detailed race categories
  - Free, no registration required
- **NCHS Vital Statistics**: https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm
  - Birth files by state

---

### 2. Life Tables / Mortality Rates (SEER or CDC)

**Source**: SEER or CDC (National Center for Health Statistics)

**What you need**: Complete life tables by sex and race for North Dakota

**Where to get it**:

**Option A: SEER Life Tables** (Recommended)
- **SEER*Stat Software**: https://seer.cancer.gov/seerstat/
- **Data**: US Mortality data with abridged life tables

**Option B: CDC NVSS Life Tables**
- **State life tables**: https://www.cdc.gov/nchs/products/life_tables.htm
- **Downloads**: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/

**Steps to obtain**:
1. Access SEER*Stat or download CDC state life tables
2. Extract life tables for:
   - Geography: North Dakota
   - Year: 2020 (most recent complete data)
   - Age groups: Single-year ages 0-90+ (or abridged: 0, 1-4, 5-9, ..., 85+)
   - Sex: Male and Female
   - Race: All available categories

**Expected format**: CSV or TXT with life table functions:
- Age (x)
- lx (number surviving to age x)
- qx (probability of dying between age x and x+1)
- Lx (person-years lived between age x and x+1)
- Tx (total person-years lived after age x)
- ex (life expectancy at age x)

**Save to**: `data/raw/mortality/seer_lifetables_2020.csv`

**Alternative**: Use national life tables and adjust for state mortality patterns

---

### 3. Migration Flows (IRS)

**Source**: IRS Statistics of Income (SOI)

**What you need**: County-to-county migration flows for North Dakota counties

**Where to get it**:
- **IRS SOI Migration Data**: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- **Direct link**: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2021-2022

**Steps to obtain**:
1. Download county-to-county migration flow files
2. Extract flows for:
   - All ND counties (38xxx) as origin or destination
   - Years: 2018-2022 (multi-year average)
   - Both inflows and outflows

**Expected format**: CSV with columns:
- Year
- State/County of origin (FIPS)
- State/County of destination (FIPS)
- Number of returns (households)
- Number of exemptions (people)
- Aggregate income

**Save to**: `data/raw/migration/irs_county_flows_2018_2022.csv`

**Important notes**:
- IRS data is aggregate (no age, sex, or race detail)
- Processor will distribute using standard age patterns
- Need to manually compile multi-year file from annual releases

---

### 4. International Migration (Census/ACS)

**Source**: American Community Survey (ACS) or Census Bureau

**What you need**: International migration estimates for North Dakota

**Where to get it**:

**Option A: ACS Public Use Microdata (PUMS)**
- **Data**: https://www.census.gov/programs-surveys/acs/microdata.html
- Filter for North Dakota, recent movers from abroad

**Option B: Census Population Estimates**
- **International migration component**: https://www.census.gov/programs-surveys/popest/data/tables.html
- State-level international migration estimates

**Option C: BigQuery (Already integrated)**
- Query ACS tables for migration data
- Use scripts in `scripts/setup/03_explore_census_data.py`

**Expected format**: CSV with:
- Year
- State FIPS
- Total international in-migrants
- (Optional) Age/sex/race distribution if available

**Save to**: `data/raw/migration/acs_international_migration.csv`

---

### 5. Geographic Reference Data (Census TIGER)

**Source**: Census Bureau TIGER/Line Shapefiles

**What you need**:
- North Dakota county list with FIPS codes
- North Dakota incorporated places list with FIPS codes
- Place-to-county mapping

**Where to get it**:

**Option A: Census TIGER/Line**
- **Download**: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
- Select year: 2020 (matches Census)
- Select layer: Counties, Places

**Option B: Manual creation**
- Create simple CSV files with FIPS codes and names
- Use Census geography documentation

**Option C: BigQuery (Already integrated)**
- Query `bigquery-public-data.census_bureau_tiger` tables
- Geographic reference built-in

**Expected format**:

`data/raw/geographic/nd_counties.csv`:
```csv
fips,name,population_2020
38001,Adams County,2200
38003,Barnes County,10853
...
```

`data/raw/geographic/nd_places.csv`:
```csv
fips,name,county_fips,population_2020
3825700,Fargo,38017,125990
3807200,Bismarck,38015,73622
...
```

---

## Data Processing Pipeline

Once you have the raw data files in place:

### Step 1: Process Demographic Rates

```bash
# Process all three components
python scripts/pipeline/01_process_demographic_data.py --all

# Or process individually
python scripts/pipeline/01_process_demographic_data.py --fertility
python scripts/pipeline/01_process_demographic_data.py --survival
python scripts/pipeline/01_process_demographic_data.py --migration
```

This will create processed rate tables in `data/processed/`:
- `fertility_rates.parquet` - 210 rows (35 ages × 6 races)
- `survival_rates.parquet` - 1,092 rows (91 ages × 2 sexes × 6 races)
- `migration_rates.parquet` - 1,092 rows (91 ages × 2 sexes × 6 races)

### Step 2: Run Projections

```bash
# Run all geographies
python scripts/pipeline/02_run_projections.py --all

# Or specific geographies
python scripts/pipeline/02_run_projections.py --counties
python scripts/pipeline/02_run_projections.py --fips 38101 38015
```

### Step 3: Export Results

```bash
# Export all results
python scripts/pipeline/03_export_results.py --all

# Create Excel exports
python scripts/pipeline/03_export_results.py --all --formats excel
```

---

## Alternative Data Sources

### If SEER Access is Difficult

**Fertility rates**:
- CDC WONDER Natality files (easier access, less race detail)
- Use national rates and scale to state birth counts

**Life tables**:
- CDC state life tables (publicly available)
- Use national life tables by race (more detailed) and adjust

**Migration**:
- Use ACS county-to-county flow tables (5-year estimates)
- Supplement with Population Estimates component data

### Synthetic Data for Testing

For testing purposes without real data:

```python
# Generate synthetic data
from examples import process_fertility_example, process_survival_example, process_migration_example

# Each example generates synthetic data and tests the processor
python examples/process_fertility_example.py
python examples/process_survival_example.py
python examples/process_migration_example.py
```

---

## Data Citation

If using this system for publication, cite data sources appropriately:

**SEER**:
> Surveillance, Epidemiology, and End Results (SEER) Program (www.seer.cancer.gov)
> SEER*Stat Database: Mortality - All COD, Aggregated With State, Total U.S.
> (1969-2021) <Katrina/Rita Population Adjustment>, National Cancer Institute,
> DCCPS, Surveillance Research Program, released December 2023.
> Underlying mortality data provided by NCHS (www.cdc.gov/nchs).

**IRS Migration**:
> Internal Revenue Service, Statistics of Income Division,
> County-to-County Migration Data, [Years Used],
> Retrieved from https://www.irs.gov/statistics/soi-tax-stats-migration-data

**Census/ACS**:
> U.S. Census Bureau, American Community Survey [Year] [1-Year/5-Year] Estimates,
> Table [Table ID], Retrieved from data.census.gov

**BigQuery Public Data**:
> Google Cloud Public Datasets, U.S. Census Bureau datasets,
> Accessed via BigQuery (bigquery-public-data.census_bureau_usa)

---

## Data Privacy & Ethics

**Important considerations**:

1. **Small cell suppression**: When dealing with small geographies or race categories, apply cell suppression rules (typically suppress cells <10 or <20)

2. **Aggregate data only**: This system is designed for aggregate population projections, not individual-level data

3. **Licensing**: Ensure compliance with data use agreements:
   - SEER: Free for research but requires registration
   - IRS: Public domain
   - Census/ACS: Public domain

4. **Race/ethnicity categories**: The 6-category system balances detail with sample size stability. Document any category combinations clearly.

---

## Questions or Issues?

**Can't access SEER data?**
- Try CDC WONDER or public NVSS files first
- Consider using national rates with state scaling
- Contact SEER help desk: SEERStat@imsweb.com

**IRS migration files confusing?**
- See examples in `examples/process_migration_example.py`
- Migration processor handles the complex distribution calculations

**Missing geographic reference data?**
- The system can auto-generate basic reference files
- Or query BigQuery for TIGER/Line data

**Need help?**
- Check ADRs in `docs/adr/` for design decisions
- Review example scripts in `examples/`
- See processor READMEs in each module directory

---

## Updates

**Last updated**: December 2025

**Data vintage recommendations** (as of Dec 2025):
- Fertility: 2018-2022 (5-year average)
- Mortality: 2020 life tables (pre-COVID or adjusted)
- Migration: 2018-2022 (5-year average, excludes COVID anomaly years)
- Base population: 2024 Population Estimates or 2020 Census

These recommendations will change over time as new data becomes available.
