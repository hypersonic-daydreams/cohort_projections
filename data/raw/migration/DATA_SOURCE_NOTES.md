# IRS SOI Migration Data - Source Documentation

## Overview

This directory contains IRS Statistics of Income (SOI) county-to-county migration data
for use in the North Dakota cohort projection system. The data tracks tax filer address
changes between tax years to estimate domestic migration flows.

## Data Source

- **Source**: IRS Statistics of Income (SOI) Division
- **Official URL**: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- **Download Date**: 2025-12-28

## Files Downloaded

| File | Year | Type | Size | Records |
|------|------|------|------|---------|
| countyinflow1819.csv | 2018-2019 | Inflows | 4.1 MB | ~56,000 |
| countyoutflow1819.csv | 2018-2019 | Outflows | 4.1 MB | ~56,000 |
| countyinflow1920.csv | 2019-2020 | Inflows | 4.2 MB | ~56,000 |
| countyoutflow1920.csv | 2019-2020 | Outflows | 4.2 MB | ~56,000 |
| countyinflow2021.csv | 2020-2021 | Inflows | 4.2 MB | ~56,000 |
| countyoutflow2021.csv | 2020-2021 | Outflows | 4.2 MB | ~56,000 |
| countyinflow2122.csv | 2021-2022 | Inflows | 4.3 MB | ~56,000 |
| countyoutflow2122.csv | 2021-2022 | Outflows | 4.3 MB | ~56,000 |
| irs_migration_documentation_2122.pdf | 2021-2022 | Documentation | 427 KB | - |

## Column Definitions

### Inflow Files (countyinflow*.csv)
People moving INTO a destination county:

| Column | Description |
|--------|-------------|
| y2_statefips | Destination state FIPS code (where people moved TO) |
| y2_countyfips | Destination county FIPS code |
| y1_statefips | Origin state FIPS code (where people moved FROM) |
| y1_countyfips | Origin county FIPS code |
| y1_state | Origin state abbreviation |
| y1_countyname | Origin county name or flow category |
| n1 | Number of returns (approximate households) |
| n2 | Number of exemptions (approximate persons) |
| agi | Adjusted gross income (thousands of dollars) |

### Outflow Files (countyoutflow*.csv)
People moving OUT OF an origin county:

| Column | Description |
|--------|-------------|
| y1_statefips | Origin state FIPS code (where people moved FROM) |
| y1_countyfips | Origin county FIPS code |
| y2_statefips | Destination state FIPS code (where people moved TO) |
| y2_countyfips | Destination county FIPS code |
| y2_state | Destination state abbreviation |
| y2_countyname | Destination county name or flow category |
| n1 | Number of returns (approximate households) |
| n2 | Number of exemptions (approximate persons) |
| agi | Adjusted gross income (thousands of dollars) |

## Special FIPS Codes

The data uses special codes for aggregations and suppressed values:

| State FIPS | County FIPS | Meaning |
|------------|-------------|---------|
| 96 | 0 | Total Migration (US and Foreign) |
| 97 | 0 | Total Migration (US only) |
| 97 | 1 | Total Same-State Migration |
| 97 | 3 | Total Different-State Migration |
| 98 | 0 | Total Migration (Foreign only) |
| 58 | 0 | Other flows - Same State (aggregated small flows) |
| 59 | 0 | Other flows - Different State (aggregated) |
| 59 | 1 | Other flows - Northeast |
| 59 | 3 | Other flows - Midwest |
| 59 | 5 | Other flows - South |
| 59 | 7 | Other flows - West |
| XX | XX (same) | Non-migrants (stayed in county) |

**Note**: A value of -1 indicates suppressed data (typically fewer than 10 returns for privacy).

## North Dakota (State FIPS 38)

To filter for North Dakota migration flows:

### Inflows to North Dakota
```python
import pandas as pd
df = pd.read_csv('countyinflow2122.csv')
nd_inflows = df[df['y2_statefips'] == 38]
```

### Outflows from North Dakota
```python
df = pd.read_csv('countyoutflow2122.csv')
nd_outflows = df[df['y1_statefips'] == 38]
```

### North Dakota County FIPS Codes
ND has 53 counties with odd-numbered FIPS codes (001, 003, 005, ... 105).

Example: Cass County (Fargo) = 38017, Burleigh County (Bismarck) = 38015

## Data Limitations

1. **Tax Filer Coverage**: Only ~70% of population files tax returns
   - Excludes children without income
   - Excludes non-filers (low income, elderly)
   - Excludes undocumented immigrants

2. **No Demographic Detail**: Data is aggregate only
   - No age breakdown
   - No sex breakdown
   - No race/ethnicity breakdown
   - Must use distribution algorithms (see ADR-003)

3. **Privacy Suppression**: Flows with <10 returns are shown as -1

4. **Timing**: Reflects address changes between tax filing years
   - 2021-2022 data reflects moves from 2021 to 2022 filing addresses

5. **Exemptions vs Persons**: The n2 (exemptions) column approximates persons
   but may differ from actual household size after 2018 tax law changes

## Processing Notes

For the cohort projection system:

1. **Use n2 (exemptions)** as the proxy for persons migrating
2. **Calculate net migration**: inflows - outflows per county
3. **Distribute by age/sex/race** using patterns in ADR-003
4. **Average multiple years** (2018-2022) to reduce volatility
5. **Handle suppressed values** (-1): set to 0 or use regional totals

## Download URLs

For future updates, download from:

- 2021-2022: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2021-2022
- 2020-2021: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2020-2021
- 2019-2020: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2019-2020
- 2018-2019: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2018-2019

Direct CSV links (example for 2021-2022):
- Inflows: https://www.irs.gov/pub/irs-soi/countyinflow2122.csv
- Outflows: https://www.irs.gov/pub/irs-soi/countyoutflow2122.csv

## References

- IRS SOI Migration Data Main Page: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- IRS Migration Data Documentation (PDF): irs_migration_documentation_2122.pdf
- ADR-003: Migration Rate Processing Methodology (../../../docs/adr/003-migration-rate-processing.md)
- Data Sources Manifest: ../../../config/data_sources.yaml
