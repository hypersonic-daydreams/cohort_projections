# Population Data Source Notes

## Overview

This directory contains population data files used to construct base populations and validate
county-level projections in the North Dakota cohort projection system. The primary workflow is:

1. **Base population totals** come from the Census PEP Vintage 2025 stcoreview file
   (`stcoreview_v2025_nd_parsed.parquet`) and the derived county population file
   (`nd_county_population.csv`).
2. **Age-sex-race distributions** are built from Census Bureau full-count estimates
   (`cc-est2024-alldata-38.csv` for 5-year age groups, `sc-est2024-alldata6.csv` for
   single-year-of-age) via the processing script `scripts/data/build_race_distribution_from_census.py`.
3. The distribution is applied to county population totals in the base population loader
   (`cohort_projections/data/load/base_population_loader.py`) to produce the starting
   population for each projection.

**Key ADRs**: ADR-044 (Census full-count race distribution), ADR-047 (county-specific
distributions), ADR-048 (single-year-of-age from SC-EST)

---

## Census County Characteristics Estimates (cc-est2024-alldata) -- Added 2026-02-18

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `cc-est2024-alldata-38.csv` | Census V2024 county-level population by age group x sex x race x Hispanic origin, North Dakota only | 6,042 | 2026-02-18 |

**Source:** Census Bureau Population Estimates Program, County Characteristics Resident Population Estimates, Vintage 2024
**URL:** `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata-38.csv`
**Layout:** `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/CC-EST2024-ALLDATA.pdf`

**Content:** Full-count demographic analysis-based estimates (not sample-based) for all 53 ND
counties. Covers April 1, 2020 through July 1, 2024. Provides population counts by 18 five-year
age groups (0-4 through 85+) x sex x race/Hispanic origin. The project uses YEAR=6 (July 1, 2024
estimate, the most recent available).

**Key column definitions:**

| Column | Description |
|--------|-------------|
| SUMLEV | Summary level (050 = county) |
| STATE | State FIPS code (38 = North Dakota) |
| COUNTY | County FIPS code (001-105) |
| STNAME | State name |
| CTYNAME | County name |
| YEAR | Year code: 1=Apr 2020 Census, 2=Apr 2020 base, 3=Jul 2021, 4=Jul 2022, 5=Jul 2023, 6=Jul 2024 |
| AGEGRP | Age group code: 0=total, 1=0-4, 2=5-9, ..., 18=85+ |
| TOT_POP | Total population |
| NHWA_MALE/FEMALE | Non-Hispanic White alone, male/female |
| NHBA_MALE/FEMALE | Non-Hispanic Black alone, male/female |
| NHIA_MALE/FEMALE | Non-Hispanic AIAN alone, male/female |
| NHAA_MALE/FEMALE | Non-Hispanic Asian alone, male/female |
| NHNA_MALE/FEMALE | Non-Hispanic NHPI alone, male/female |
| NHTOM_MALE/FEMALE | Non-Hispanic Two or More Races, male/female |
| H_MALE/H_FEMALE | Hispanic (any race), male/female |

**Race mapping to project categories (per ADR-007):**

| Census Columns | Project Category |
|---------------|-----------------|
| NHWA_MALE/FEMALE | white_nonhispanic |
| NHBA_MALE/FEMALE | black_nonhispanic |
| NHIA_MALE/FEMALE | aian_nonhispanic |
| NHAA + NHNA (summed) | asian_nonhispanic (Asian + NHPI combined) |
| NHTOM_MALE/FEMALE | multiracial_nonhispanic |
| H_MALE/H_FEMALE | hispanic |

**Processing scripts:**
- `scripts/data/build_race_distribution_from_census.py` -- Reads this file, filters to YEAR=6 and AGEGRP>0, sums across counties for statewide distribution, and also builds per-county distributions (ADR-047)
- Also used as population denominator in `scripts/data/build_nd_fertility_rates.py` for computing ND-specific ASFR

**Outputs produced:**
- `data/raw/population/nd_age_sex_race_distribution.csv` (statewide, 216 rows)
- `data/processed/county_age_sex_race_distributions.parquet` (per-county, 11,448 rows)

**Validation:**

| Metric | Value | Status |
|--------|-------|--------|
| Counties | 53 | All ND counties present |
| Age groups per county | 18 (0-4 through 85+) | Complete |
| YEAR codes | 1-6 | Complete (Apr 2020 through Jul 2024) |
| Total rows | 6,042 (53 counties x 6 years x 19 AGEGRP including total) | Expected |
| State total population (YEAR=6) | 796,568 | Matches NST-EST2024 |

---

## Census State Characteristics Single-Year Estimates (SC-EST2024-ALLDATA6) -- Added 2026-02-18

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `sc-est2024-alldata6.csv` | Census V2024 state-level population by single year of age x sex x race x Hispanic origin, all states | 236,844 | 2026-02-18 |

**Source:** Census Bureau Population Estimates Program, Annual State Resident Population Estimates for 6 Race Groups by Age, Sex, and Hispanic Origin, Vintage 2024
**URL:** `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/sc-est2024-alldata6.csv`
**Layout:** `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/SC-EST2024-ALLDATA6.pdf`

**Content:** State-level single-year-of-age (0-85) population estimates by sex, race (6 groups),
and Hispanic origin for all 50 states + DC. For the project, filtered to STATE=38 (North Dakota),
yielding 4,644 rows. The 85+ open-ended age group is distributed to single years 85-90 using
exponential decay with a survival factor of 0.7 per year.

**Key column definitions:**

| Column | Description |
|--------|-------------|
| SUMLEV | Summary level (040 = state) |
| STATE | State FIPS code (38 = North Dakota) |
| NAME | State name |
| SEX | 0=total, 1=male, 2=female |
| ORIGIN | 0=total, 1=Not Hispanic, 2=Hispanic |
| RACE | 1=White, 2=Black, 3=AIAN, 4=Asian, 5=NHPI, 6=Two or More |
| AGE | Single year of age (0-85; 85 = 85+) |
| ESTIMATESBASE2020 | April 1, 2020 estimates base |
| POPESTIMATE2020-2024 | July 1 population estimates for each year |

**Processing script:** `scripts/data/build_race_distribution_from_census.py` (function `build_single_year_statewide_distribution`)
- Filters to STATE=38, SEX in [1,2], ORIGIN in [1,2] (excludes totals)
- Maps (ORIGIN, RACE) pairs to project race categories (NHPI combined with Asian)
- Uses POPESTIMATE2024 as the population column
- Expands 85+ to ages 85-90 via exponential decay
- Aggregates Hispanic races (all RACE values within ORIGIN=2 sum to "hispanic")

**Output produced:** `data/raw/population/nd_age_sex_race_distribution_single_year.csv` (1,092 rows)

---

## Statewide Age-Sex-Race Distribution (Generated) -- Added 2026-02-18

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `nd_age_sex_race_distribution.csv` | Statewide proportional distribution by 5-year age group x sex x race | 216 | Generated 2026-02-18 |

**Generated by:** `scripts/data/build_race_distribution_from_census.py` from `cc-est2024-alldata-38.csv`
**ADR:** ADR-044 (Census Full-Count Race Distribution)

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| age_group | string | 5-year age group (0-4, 5-9, ..., 85+) |
| sex | string | male, female |
| race_ethnicity | string | white_nonhispanic, black_nonhispanic, aian_nonhispanic, asian_nonhispanic, multiracial_nonhispanic, hispanic |
| estimated_count | float | Population count from Census full-count estimates |
| proportion | float | Fraction of state total population (sums to 1.0) |

**Dimensions:** 18 age groups x 2 sexes x 6 races = 216 rows

**Validation:**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total rows | 216 | 216 | Exact |
| Populated cells | 216 / 216 | 216 / 216 | All cells non-zero |
| Proportion sum | 1.00000000 | 1.0 | Within 1e-8 |
| Sex ratio | 105.5 males per 100 females | 95-115 | Reasonable |
| Black females 15-49 | ~7,600 | >0 | All 7 age groups populated |
| State total population | 796,568 | Matches NST-EST2024 | Consistent |

**Usage:** Used as the statewide distribution for state-level projections and as the blending
anchor for small counties (population < 5,000) in the county-specific distribution system (ADR-047).

---

## Single-Year-of-Age Statewide Distribution (Generated) -- Added 2026-02-18

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `nd_age_sex_race_distribution_single_year.csv` | Statewide proportional distribution by single year of age x sex x race | 1,092 | Generated 2026-02-18 |

**Generated by:** `scripts/data/build_race_distribution_from_census.py` from `sc-est2024-alldata6.csv`
**ADR:** ADR-048 (Single-Year-of-Age Base Population)

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| age | int | Single year of age (0-90; 90 = open-ended 90+) |
| sex | string | male, female |
| race_ethnicity | string | white_nonhispanic, black_nonhispanic, aian_nonhispanic, asian_nonhispanic, multiracial_nonhispanic, hispanic |
| estimated_count | float | Population count (85-90 are derived via exponential decay from 85+ total) |
| proportion | float | Fraction of state total population (sums to 1.0) |

**Dimensions:** 91 ages x 2 sexes x 6 races = 1,092 rows

**Config reference:** `base_population.single_year_distribution` in `config/projection_config.yaml`

**Usage:** Primary distribution file when `base_population.age_resolution: "single_year"` is set
(the current default). Loaded by `base_population_loader.py` to construct the starting population
for each projection run. Eliminates step-function artifacts that occurred when 5-year groups were
uniformly split to single years.

---

## Census PEP Vintage 2025 Stcoreview (Pre-Release) -- Added 2026-02-17

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `stcoreview_v2025_nd_parsed.parquet` | Parsed Census PEP Vintage 2025 State/County Review file for ND | 13,284 | 2026-02-17 |

**Source:** Census Bureau Population Estimates Program, Vintage 2025, State and County Review File
for North Dakota. This is an advance/pre-release internal product sent to state demographers before
the official public county-level release.
**Original file:** `stcoreview_v2025_ND.xlsx` (downloaded from Census Bureau distribution)

**Processing script:** `scripts/data/ingest_stcoreview.py`
- Reads the Excel file (sheet "in"), which contains 54 rows (1 state + 53 counties) x 249 columns
- Parses column names using regex to extract variable, age group, and period
- Converts to long format (tidy data)
- Handles '.' missing values

**Column definitions (parsed output):**

| Column | Type | Description |
|--------|------|-------------|
| geoid | string | 5-digit FIPS code (e.g., 38000 = state, 38001 = Adams) |
| state_fips | string | State FIPS (38) |
| county_fips | string | 3-digit county code (000 = state total) |
| county_name | string | Geographic name |
| is_state_total | bool | True for the state summary row |
| variable | string | Demographic variable (see below) |
| age_group | string | total, 0-17, 18-64, or 65+ |
| period | string | census, base, or year (2020-2025) |
| value | float | The estimate value (nullable) |
| year | Int64 | Numeric year (null for census/base periods) |

**Variables available:**

| Variable | Description |
|----------|-------------|
| Respop | Resident population |
| HHpop | Household population |
| GQpop | Group quarters population |
| Births | Annual births |
| Deaths | Annual deaths |
| Dommig | Domestic migration (count) |
| Dommigrate | Domestic migration rate |
| Intlmig | International migration |
| Residual | Residual change |
| Natrake | Natural change (births - deaths) |
| Popturning18 | Population turning 18 |
| Popturning65 | Population turning 65 |

**Usage:** Primary source for Vintage 2025 county population totals and components of change.
The `Respop` variable for period 2025 provides the base year population used in projections.
Components (Births, Deaths, Dommig, Intlmig) are used for calibration and validation.

**Important note:** This is pre-release data. Re-verify against the official public release when
published. Document source as "Census Bureau PEP, Vintage 2025, State and County Review File for
North Dakota (advance/pre-release)."

---

## ND County Population Summary -- Added 2026-02-17

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `nd_county_population.csv` | County-level population totals and components for 2024-2025 | 53 | Derived 2026-02-17 |

**Derived from:** `stcoreview_v2025_nd_parsed.parquet` (Vintage 2025 data)

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| county_fips | string | 5-digit FIPS code (38001-38105) |
| county_name | string | County name |
| population_2024 | int | Revised July 1, 2024 population (Vintage 2025 revision) |
| births_2024 | int | Births, July 2023 - June 2024 |
| deaths_2024 | int | Deaths, July 2023 - June 2024 |
| net_migration_2024 | int | Net migration, July 2023 - June 2024 |
| population_2025 | int | July 1, 2025 population estimate |
| births_2025 | int | Births, July 2024 - June 2025 |
| deaths_2025 | int | Deaths, July 2024 - June 2025 |
| net_migration_2025 | int | Net migration, July 2024 - June 2025 |

**Rows:** 53 (all ND counties; no state total row)

**Usage:** Loaded by `base_population_loader.py` (function `load_county_population()`) to provide
the base year population totals for each county. The `population_2025` column is the primary
population total that gets allocated across age-sex-race cells using the distribution files.

**Config reference:** The base population loader uses `pop_col = "population_2025"` to select the
column for the projection base year.

---

## Census PEP County Estimates (co-est2024-alldata) -- Added pre-2026-02

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `co-est2024-alldata.csv` | Census V2024 county-level population estimates and components of change, all U.S. counties | 3,195 | Pre-2026-02 (file timestamp 2025-06-17) |

**Source:** Census Bureau Population Estimates Program, County Population Totals, Vintage 2024
**URL:** `https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html`

**Content:** Annual population estimates (2020-2024) with full demographic components (births,
deaths, domestic migration, international migration, residual) for all U.S. counties. Contains
54 ND rows (53 counties + 1 state total where COUNTY=0).

**Key columns:** SUMLEV, REGION, DIVISION, STATE, COUNTY, STNAME, CTYNAME, ESTIMATESBASE2020,
POPESTIMATE2020-2024, BIRTHS2020-2024, DEATHS2020-2024, NATURALCHG2020-2024,
INTERNATIONALMIG2020-2024, DOMESTICMIG2020-2024, NETMIG2020-2024, RESIDUAL2020-2024,
plus rate columns (RBIRTH, RDEATH, etc.)

**Usage:** Used for validation of county-level estimates and comparison with Vintage 2025 data.
Also consumed by the PEP county migration extraction scripts
(`scripts/data_processing/extract_pep_county_migration.py`).

**Note:** This is Vintage 2024 data. For the base year 2025 population, the project uses the
Vintage 2025 stcoreview data instead. This file is retained for reference, validation, and
historical comparison.

---

## Census PEP State Estimates (NST-EST2024-ALLDATA) -- Added pre-2026-02

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `NST-EST2024-ALLDATA.csv` | Census V2024 state-level population estimates and components of change, all states | 66 | Pre-2026-02 (file timestamp 2025-06-17) |

**Source:** Census Bureau Population Estimates Program, State Population Totals, Vintage 2024
**URL:** `https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html`

**Content:** Annual population estimates (2020-2024) with full demographic components for all
50 states, DC, and regional/national aggregates. Contains summary levels 010 (nation),
020 (region), 030 (division), and 040 (state).

**Key columns:** SUMLEV, REGION, DIVISION, STATE, NAME, ESTIMATESBASE2020, POPESTIMATE2020-2024,
BIRTHS2020-2024, DEATHS2020-2024, NATURALCHG2020-2024, INTERNATIONALMIG2020-2024,
DOMESTICMIG2020-2024, NETMIG2020-2024, RESIDUAL2020-2024, plus rate columns

**ND-specific:** North Dakota (NAME="North Dakota", STATE=38) POPESTIMATE2024 = 796,568

**Usage:** State-level totals used for validation of county sums and for state-level demographic
component analysis. The ND POPESTIMATE2024 value (796,568) is cross-checked against the
cc-est2024-alldata and county population file sums.

---

## ACS PUMS Person Records (Legacy) -- Added pre-2026-02

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `pums_person.parquet` | ACS Public Use Microdata Sample, ND person records | 1,000 | Pre-2026-02 (file timestamp 2025-10-17) |

**Source:** Census Bureau American Community Survey, Public Use Microdata Sample (PUMS)
**URL:** `https://www.census.gov/programs-surveys/acs/microdata.html`
**Fetched from:** Sibling repository `maps/data/raw/pums_person.parquet` via `scripts/fetch_data.py`

**Content:** ACS PUMS person-level microdata for North Dakota (STATE=38). Contains 287 columns
including demographic, economic, and housing characteristics. Key demographic columns: AGEP (age),
SEX, RAC1P (race), HISP (Hispanic origin), PWGTP (person weight).

**Status: LEGACY / SUPERSEDED.** This file was originally used for race allocation in the
Census+PUMS hybrid base population approach (ADR-041). It has been superseded by the Census
full-count data (`cc-est2024-alldata-38.csv`) per ADR-044. The PUMS 1% sample was
catastrophically insufficient for small race groups: zero Black females at reproductive ages,
45% of Hispanics concentrated in a single age-sex cell, and only 115 of 216 possible cells
populated.

**Retained for:** Reference and potential future analysis (e.g., migration pattern research).
Not used in the current production pipeline.

---

## Historical Notes

### PUMS to Census Full-Count Transition (ADR-044, 2026-02-18)

Prior to ADR-044, the project used a Census+PUMS hybrid approach (ADR-041) for the age-sex-race
distribution. The Census API provided age-sex totals, and ACS PUMS microdata provided race
proportions within each age-sex cell. The PUMS 1% sample (~12,277 records for ND) was insufficient
for race cross-tabulation of small groups:

| Group | PUMS Observations | Populated Cells (of 36) | Critical Defect |
|-------|-------------------|------------------------|-----------------|
| Black non-Hispanic | ~44 | 7 | Zero females at reproductive ages |
| Hispanic | ~163 | 11 | 45% in single cell (F 10-14) |
| Asian/PI | ~139 | 8 | Sparse across most age groups |
| Two or more races | ~268 | 17 | Gaps in older age groups |

The switch to Census full-count data (`cc-est2024-alldata-38.csv`) populated all 216 cells,
restored Black females at reproductive ages (~7,600), and distributed Hispanic population
realistically across age groups.

### Statewide to County-Specific Distributions (ADR-047, 2026-02-18)

Prior to ADR-047, a single statewide distribution was applied to all 53 counties. This meant
Sioux County (78% AIAN in reality) received only ~4.8% AIAN allocation (the statewide average).
The median county misallocation was 17.6%, with reservation counties reaching 46-76%.

ADR-047 introduced county-specific distributions from the same `cc-est2024-alldata-38.csv` source,
with population-weighted blending for counties below 5,000 population to prevent zero-cell
artifacts.

### Uniform Splitting to Single-Year-of-Age (ADR-048, 2026-02-18)

Prior to ADR-048, the 216-row 5-year age group distribution was split to single years by dividing
each group's proportion equally across 5 years. This created staircase artifacts with ~4.4% jumps
at group boundaries that propagated through all projection years.

ADR-048 introduced `sc-est2024-alldata6.csv` as the source for genuine single-year-of-age
estimates, producing a smooth 1,092-row distribution. For county-level single-year expansion,
Sprague osculatory interpolation is used (configured via `base_population.county_race_interpolation`
in `projection_config.yaml`).

### Vintage 2024 to Vintage 2025 Transition (2026-02-17)

Prior to 2026-02-17, the project used Vintage 2024 population totals from `co-est2024-alldata.csv`.
The incorporation of the pre-release Vintage 2025 stcoreview file provided an additional year of
data (2025 estimates) and revised 2024 figures. The `nd_county_population.csv` was updated with
both 2024 (revised) and 2025 columns.

**Note:** The age-sex-race distributions still use Vintage 2024 data (`cc-est2024-alldata-38.csv`
and `sc-est2024-alldata6.csv`) because the county characteristics file for Vintage 2025 has not
yet been publicly released. The race composition changes less than 1% per year, so the vintage
mismatch introduces negligible error. Update when `cc-est2025-alldata` is released (expected
mid-2026).

---

## Data Quality Notes

1. **Vintage mismatch:** V2024 race distributions applied to V2025 population totals. Negligible
   error (< 1% racial composition change per year).

2. **Pre-release data:** The stcoreview Vintage 2025 file is an advance product. Values should be
   re-verified against the official public release when available.

3. **Terminal age group:** SC-EST data has 85 as the maximum age (representing 85+). The processing
   script distributes this to ages 85-90 using exponential decay (survival factor 0.7). Age 90
   represents the open-ended 90+ group in the projection engine.

4. **NHPI combined with Asian:** Native Hawaiian/Pacific Islander populations are combined with
   Asian populations per the project's 6-category race scheme (ADR-007). In ND, NHPI is a very
   small group.

5. **Small county blending:** For counties below 5,000 population, distributions are blended with
   the statewide distribution using `alpha = min(county_pop / 5000, 1.0)`. This prevents zero-cell
   artifacts while preserving the county's dominant demographic patterns.

---

## References

1. Census Bureau CC-EST2024-ALLDATA: `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/`
2. Census Bureau SC-EST2024-ALLDATA6: `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/state/asrh/`
3. Census Bureau CO-EST2024-ALLDATA: `https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html`
4. Census Bureau NST-EST2024-ALLDATA: `https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html`
5. Census Bureau ACS PUMS: `https://www.census.gov/programs-surveys/acs/microdata.html`
6. ADR-007: Race/Ethnicity Categorization
7. ADR-041: Census+PUMS Hybrid Base Population (superseded by ADR-044)
8. ADR-044: Census Full-Count Race Distribution
9. ADR-047: County-Specific Age-Sex-Race Distributions
10. ADR-048: Single-Year-of-Age Base Population from Census SC-EST Data

---

## Related Documentation

- Processing script: `scripts/data/build_race_distribution_from_census.py`
- Ingestion script: `scripts/data/ingest_stcoreview.py`
- Base population loader: `cohort_projections/data/load/base_population_loader.py`
- Config: `config/projection_config.yaml` (base_population section)
- Vintage 2025 analysis: `docs/reviews/2026-02-17-vintage-2025-census-data-analysis.md`
- Sanity check review: `docs/reviews/2026-02-18-projection-output-sanity-check.md`
