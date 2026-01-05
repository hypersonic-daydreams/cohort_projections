# Census Bureau Foreign-Born Population Data (ACS Table B05006)

## Overview

This directory contains American Community Survey (ACS) 5-Year Estimates from the Census Bureau's Table B05006: "Place of Birth for the Foreign-Born Population in the United States".

## Data Source

- **Table**: B05006
- **Survey**: American Community Survey (ACS) 5-Year Estimates
- **Source**: U.S. Census Bureau Data API
- **API Endpoint**: `https://api.census.gov/data/{year}/acs/acs5?get=group(B05006)&for=state:*`
- **Access Date**: 2024-12-28
- **Geographic Level**: State-level for all 50 states, DC, and Puerto Rico (52 total)

## Years Covered

- 2015 (ACS 5-Year: 2011-2015)
- 2016 (ACS 5-Year: 2012-2016)
- 2017 (ACS 5-Year: 2013-2017)
- 2018 (ACS 5-Year: 2014-2018)
- 2019 (ACS 5-Year: 2015-2019)
- 2020 (ACS 5-Year: 2016-2020)
- 2021 (ACS 5-Year: 2017-2021)
- 2022 (ACS 5-Year: 2018-2022)
- 2023 (ACS 5-Year: 2019-2023)

## Files in This Directory

### Raw Data Files

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `b05006_states_2015.csv` | 2015 5-year estimates by state | 52 | 327 |
| `b05006_states_2016.csv` | 2016 5-year estimates by state | 52 | 327 |
| `b05006_states_2017.csv` | 2017 5-year estimates by state | 52 | 327 |
| `b05006_states_2018.csv` | 2018 5-year estimates by state | 52 | 327 |
| `b05006_states_2019.csv` | 2019 5-year estimates by state | 52 | 339 |
| `b05006_states_2020.csv` | 2020 5-year estimates by state | 52 | 339 |
| `b05006_states_2021.csv` | 2021 5-year estimates by state | 52 | 339 |
| `b05006_states_2022.csv` | 2022 5-year estimates by state | 52 | 359 |
| `b05006_states_2023.csv` | 2023 5-year estimates by state | 52 | 359 |
| `b05006_states_all_years.csv` | Combined all years | 468 | 360 |
| `b05006_variable_labels.json` | Variable labels and metadata | - | - |

### Scripts

| File | Description |
|------|-------------|
| `download_b05006.py` | Script to download data from Census API |
| `process_b05006.py` | Script to transform data to long format and calculate ND share |

## Data Structure

### Raw Format (Wide)

Each row represents one state/year combination. Columns include:
- `NAME`: State name
- `state`: State FIPS code
- `GEO_ID`: Geographic identifier
- `B05006_###E`: Estimate for variable ### (foreign-born population count)
- `B05006_###M`: Margin of error for variable ###
- `year`: Survey year

### Variable Hierarchy

B05006 provides a hierarchical breakdown of foreign-born population by place of birth:

1. **Total** (B05006_001E)
2. **Regions**: Europe, Asia, Africa, Oceania, Americas
3. **Sub-regions**: Northern Europe, Eastern Asia, etc.
4. **Countries**: Germany, China, Mexico, etc.
5. **Details**: Some countries have sub-country detail (e.g., China broken down by Hong Kong, Taiwan)

### Key Regions and Sub-regions

- **Europe**: Northern, Western, Southern, Eastern Europe
- **Asia**: Eastern, South Central, South Eastern, Western Asia
- **Africa**: Eastern, Middle, Northern, Southern, Western Africa
- **Oceania**: Australia/New Zealand, Fiji, Pacific Islands
- **Americas**: Northern America, Latin America (Caribbean, Central America, South America)

## Processed Output Files

The processing script creates two parquet files in the analysis directory:

### `acs_foreign_born_by_state_origin.parquet`

Long-format data with all states and all origins:

| Column | Description |
|--------|-------------|
| `year` | Survey year (2015-2023) |
| `state_fips` | 2-digit state FIPS code |
| `state_name` | State name |
| `variable` | Original Census variable code |
| `region` | World region (Europe, Asia, etc.) |
| `sub_region` | Sub-region (Northern Europe, etc.) |
| `country` | Country of birth |
| `detail` | Sub-country detail (if any) |
| `level` | Hierarchy level (total, region, sub_region, country, detail) |
| `foreign_born_pop` | Foreign-born population estimate |
| `margin_of_error` | Margin of error for estimate |

### `acs_foreign_born_nd_share.parquet`

North Dakota-specific analysis with national comparison:

| Column | Description |
|--------|-------------|
| All columns from above, plus: | |
| `nd_foreign_born` | North Dakota foreign-born population |
| `nd_moe` | North Dakota margin of error |
| `national_foreign_born` | National foreign-born population |
| `national_moe` | National margin of error |
| `nd_share_of_national` | ND share as fraction of national total |

## Notes on Table Changes Over Time

The B05006 table has evolved slightly over the years:
- 2015-2018: 327 columns (fewer country breakdowns)
- 2019-2021: 339 columns (additional country codes added)
- 2022-2023: 359 columns (more detailed breakdowns)

Some countries may not be available in all years if the table structure changed.

## North Dakota Key Findings (2023)

Top origins for foreign-born population in North Dakota:
1. Kenya: ~12,000 (0.45% of national Kenyan-born)
2. Central America: ~8,500
3. Australia: ~5,700 (0.56% of national Australian-born)
4. Somalia: ~3,600 (0.47% of national Somali-born)
5. Cambodia: ~3,500

North Dakota has disproportionately high shares of:
- Australian-born (0.56% of national)
- Kenyan-born (0.45% of national)
- Somali-born (0.47% of national)

This is notable given North Dakota's ~0.2% share of the total US population.

## Data Quality Notes

- Estimates are from 5-year pooled samples, providing greater precision for small populations
- Margins of error are at the 90% confidence level
- Some cells may show 0 or null for very small populations
- Sub-county detail (e.g., Hong Kong within China) may have higher relative MOE

## References

- Census Bureau B05006 Documentation: https://data.census.gov/table/ACSDT5Y2023.B05006
- ACS Technical Documentation: https://www.census.gov/programs-surveys/acs/technical-documentation.html
- Foreign-Born Population Topic: https://www.census.gov/topics/population/foreign-born.html
