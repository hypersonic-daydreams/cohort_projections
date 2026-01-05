# DHS Naturalizations Data Manifest

**Download Date:** 2025-12-28
**Additional Download Date:** 2026-01-04 (NATZ2004%20Supp_0.zip; NATZ%202012%20Supp_0.zip)
**Source:** U.S. Department of Homeland Security, Office of Homeland Security Statistics (OHSS)

## Downloaded Files

### Refugee Flow Reports (PDFs)

Refugee flow reports are stored in `data/raw/immigration/dhs_refugees_asylees/` to keep DHS product families separated.

### Naturalization Data (Excel)

**Current Years (FY 2021-2023):**

| File | Fiscal Year | Size | Source URL |
|------|-------------|------|-----------|
| `naturalizations_fy2023.xlsx` | FY 2023 | 159 KB | https://ohss.dhs.gov/system/files/2025-07/2025_0725_plcy_yearbook_naturalizations_fy2023.xlsx |
| `naturalizations_fy2022.xlsx` | FY 2022 | 178 KB | https://ohss.dhs.gov/sites/default/files/2023-12/2023_0818_plcy_yearbook_naturalizations_fy2022.xlsx_0_0.xlsx |
| `naturalizations_fy2021.zip` | FY 2021 | 84 KB | https://ohss.dhs.gov/sites/default/files/2023-12/2022_0624_plcy_naturalizations_fy2021_tables_1.zip |

**Historical Yearbook Data (FY 2000-2020):**

Located in `historical_downloads/` subdirectory. Downloaded from DHS Yearbook of Immigration Statistics.

| Year Range   | Files                              | Contains                                     |
|--------------|------------------------------------|--------------------------------------------- |
| FY 2000      | `natz_fy2000.zip`                  | Tables 46-57 (State data in Table 51)        |
| FY 2001      | `natz_fy2001.zip`                  | Tables 45-56 (State data in Table 50)        |
| FY 2002      | `natz_fy2002.zip`, `*_supp.zip`    | Tables 34-37, Supp Tables                    |
| FY 2003      | `natz_fy2003.zip`, `*_supp.zip`    | Tables 31-34, Supp Tables                    |
| FY 2004      | `natz_fy2004.zip`, `*_supp.zip`    | Tables 31-34, Supp Tables                    |
| FY 2005-2020 | `natz_fyYYYY.zip`, `*_supp.zip`    | Table 22 (state), Tables 20-24, Supp Tables  |

Each zip file contains Excel files with naturalization tables. Table numbers vary by year but include:

- State-level multi-year naturalization counts (Table 22 equivalent)
- Supplemental Table 1: State x Country of Birth cross-tabulation

**Additional archive note (2026-01-04):**
- `NATZ2004%20Supp_0.zip` added to `historical_downloads/` as an alternate download of the FY2004 supplemental tables (duplicate content retained for provenance).
- `NATZ%202012%20Supp_0.zip` added to `historical_downloads/` as an alternate download of FY2012 supplemental tables.

## Extracted Data

### Naturalization Data

**Output files:**

- `../../analysis/dhs_naturalizations_by_state.parquet` - State-level naturalizations (FY 2014-2023)
- `../../analysis/dhs_naturalizations_by_state_country.parquet` - Detailed state x country of birth
- `../../analysis/dhs_naturalizations_by_state_historical.parquet` - **NEW** Full historical state-level data (FY 1986-2023)
- `../../analysis/dhs_naturalizations_by_state_country_historical.parquet` - **NEW** Historical state x country data

**Historical Data Coverage (FY 1986-2023):**

The historical parquet file contains 2,366 records across 77 states/territories and 38 fiscal years (1986-2023).

Data extracted from DHS Yearbook Excel files across all available years:

1. **State-Level Multi-Year Tables:**
   - FY 2000: Table 51 (1986-2000)
   - FY 2001: Table 50 (1986-2001)
   - FY 2002: Table 36 (1986-2002)
   - FY 2003-2004: Table 33 (1986-2003/2004)
   - FY 2005-2023: Table 22 (rolling 10-year windows)

2. **Supplemental Table 1: Persons Naturalized by State and Country of Birth**
   - Available for FY 2002-2020 and FY 2023
   - Cross-tabulation of state of residence by country of birth

**Key Findings - Naturalizations (FY 2023):**

| Rank | State        | Naturalizations |
|------|--------------|-----------------|
| 1    | California   | 154,520         |
| 2    | Texas        | 100,290         |
| 3    | Florida      | 94,210          |
| 4    | New York     | 91,480          |
| 5    | New Jersey   | 38,820          |
| ...  | ...          | ...             |
| 40   | North Dakota | 1,780           |

**North Dakota Historical Trend (FY 1986-2023):**

| Period    | Avg. Annual | Peak Year | Peak Count |
|-----------|-------------|-----------|------------|
| 1986-1995 | 160         | 1988      | 212        |
| 1996-2005 | 175         | 2005      | 203        |
| 2006-2015 | 423         | 2013      | 532        |
| 2016-2023 | 956         | 2023      | 1,780      |

North Dakota naturalizations have grown 10x from the 1980s average to the 2020s, reflecting
increased immigration to the state.

## Data Notes

### Naturalization Data Notes
- Includes all persons who took the oath of citizenship during the fiscal year
- State reflects residence at time of naturalization
- Country of birth may differ from nationality/citizenship

### Per Capita Calculations
- State population estimates from U.S. Census Bureau (2022 NST-EST2022-POP)
- Per capita rates calculated as: (count / population) * 100,000

## Processing Scripts

**`process_dhs_data.py`** - Main processing script that:

1. Extracts data tables from refugee flow PDF reports (stored in `data/raw/immigration/dhs_refugees_asylees/`)
2. Reads and processes Excel naturalization files (FY 2021-2023)
3. Merges with state population data
4. Calculates per-capita rates
5. Outputs parquet files for analysis

**`process_historical_naturalizations.py`** - Historical data processing script that:

1. Processes all yearbook files in `historical_downloads/` (FY 2000-2020)
2. Handles varying table structures across years (Table 22/33/36/50/51)
3. Extracts multi-year state-level naturalization data
4. Processes Supplemental Table 1 (state x country) where available
5. Combines with FY 2021-2023 data
6. Outputs consolidated historical parquet files

## Related Documentation

- OHSS Main Page: https://ohss.dhs.gov/topics/immigration
- Yearbook of Immigration Statistics: https://ohss.dhs.gov/topics/immigration/yearbook

## Data Quality Notes

1. **Consistency**: Numbers in this manifest may differ slightly from DHS published totals due to rounding
2. **Timeliness**: Data represents final counts as of report publication dates
