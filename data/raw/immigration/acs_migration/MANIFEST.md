# ACS Mobility Data (B07007) - Moved From Abroad

## Overview

This directory contains American Community Survey (ACS) 5-Year Estimates from Census Table B07007: "Geographical Mobility in the Past Year by Citizenship Status". The series is used as a proxy for inflows from abroad (people who moved from outside the U.S. in the prior year), by state and citizenship.

## Data Source

- **Table**: B07007
- **Survey**: ACS 5-Year Estimates
- **Source**: U.S. Census Bureau Data API
- **API Endpoint**: `https://api.census.gov/data/{year}/acs/acs5?get=group(B07007)&for=state:*`
- **Access Date**: 2026-01-04
- **Geographic Level**: State-level (50 states + DC + Puerto Rico)

## Years Covered

- 2010-2023 (ACS 5-year estimates)
- **Note:** 2009 group is not available via the API (HTTP 404).

## Files in This Directory

### Raw Data Files

| File | Description |
|------|-------------|
| `b07007_states_{year}.csv` | State-level B07007 group data for year `{year}` |
| `b07007_states_all_years.csv` | Combined all years |
| `b07007_variable_labels_{year}.json` | Variable labels for year `{year}` |
| `b07007_variable_labels.json` | Labels from most recent year (default) |

## Data Structure

B07007 provides mobility categories by citizenship status. The moved-from-abroad subseries uses the following estimates (with matching MOE variables):

- `B07007_026E`: Total moved from abroad
- `B07007_027E`: Moved from abroad: Native
- `B07007_028E`: Moved from abroad: Foreign born
- `B07007_029E`: Moved from abroad: Foreign born: Naturalized U.S. citizen
- `B07007_030E`: Moved from abroad: Foreign born: Not a U.S. citizen

MOE fields use the same numeric codes with `M` suffix (e.g., `B07007_026M`).

## Processing

- **Download Script**: `sdc_2024_replication/data_immigration_policy/scripts/download_b07007.py`
- **Processing Script**: `sdc_2024_replication/data_immigration_policy/scripts/process_b07007.py`
- **Outputs**: `data/processed/immigration/analysis/acs_moved_from_abroad_by_state.parquet`

## Data Quality Notes

- ACS 5-year estimates are rolling averages; "2023" represents 2019-2023.
- Margins of error are at the 90% confidence level.
- Small states can have higher relative MOE; interpret year-to-year changes cautiously.

## References

- ACS Table B07007: https://data.census.gov/table/ACSDT5Y2023.B07007
- ACS Technical Documentation: https://www.census.gov/programs-surveys/acs/technical-documentation.html
