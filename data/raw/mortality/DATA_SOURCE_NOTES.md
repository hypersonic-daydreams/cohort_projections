# CDC Life Table Data Source Notes

## Overview

This directory contains life table data from the CDC National Center for Health Statistics (NCHS)
for use in the cohort projections mortality module. National U.S. life tables are used as proxies
for North Dakota, as state-specific life tables by race/ethnicity are not available.

## Data Source

**Source**: CDC National Center for Health Statistics (NCHS)
**Publication**: National Vital Statistics Reports (NVSR)
**Data Years**: 2022 and 2023

### Primary URLs

- Life Tables Homepage: https://www.cdc.gov/nchs/products/life_tables.htm
- 2023 Data (NVSR 74-06): https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/74-06/
- 2022 Data (NVSR 74-02): https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/74-02/
- PDF Reports: https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-06.pdf (2023)

## Downloaded Files

### 2023 Life Tables (Most Recent - Recommended)

| File | Description |
|------|-------------|
| cdc_lifetable_2023_table01.xlsx | Total population |
| cdc_lifetable_2023_table02.xlsx | Males (all races) |
| cdc_lifetable_2023_table03.xlsx | Females (all races) |
| cdc_lifetable_2023_table04.xlsx | Hispanic (both sexes) |
| cdc_lifetable_2023_table05.xlsx | Hispanic males |
| cdc_lifetable_2023_table06.xlsx | Hispanic females |
| cdc_lifetable_2023_table07.xlsx | American Indian/Alaska Native, non-Hispanic (both sexes) |
| cdc_lifetable_2023_table08.xlsx | AIAN, non-Hispanic males |
| cdc_lifetable_2023_table09.xlsx | AIAN, non-Hispanic females |
| cdc_lifetable_2023_table10.xlsx | Asian, non-Hispanic (both sexes) |
| cdc_lifetable_2023_table11.xlsx | Asian, non-Hispanic males |
| cdc_lifetable_2023_table12.xlsx | Asian, non-Hispanic females |
| cdc_lifetable_2023_table13.xlsx | Black, non-Hispanic (both sexes) |
| cdc_lifetable_2023_table14.xlsx | Black, non-Hispanic males |
| cdc_lifetable_2023_table15.xlsx | Black, non-Hispanic females |
| cdc_lifetable_2023_table16.xlsx | White, non-Hispanic (both sexes) |
| cdc_lifetable_2023_table17.xlsx | White, non-Hispanic males |
| cdc_lifetable_2023_table18.xlsx | White, non-Hispanic females |

### Combined Processed File

**cdc_lifetables_2023_combined.csv** - Unified CSV combining all 2023 sex-specific life tables

## Column Definitions

| Column | Description | Unit |
|--------|-------------|------|
| age | Age in completed years | Integer (0-99) |
| sex | Sex category | Male, Female |
| race_ethnicity | Race/ethnicity group | AIAN, Asian, Black, Hispanic, White, Total |
| qx | Probability of dying between ages x and x+1 | Probability (0-1) |
| lx | Number surviving to age x | Out of 100,000 radix |
| dx | Number dying between ages x and x+1 | Count |
| Lx | Person-years lived between ages x and x+1 | Person-years |
| Tx | Total person-years lived above age x | Person-years |
| ex | Life expectancy at age x | Years |
| survival_rate | Probability of surviving to next age (1 - qx) | Probability (0-1) |

## Race/Ethnicity Categories

The CDC provides life tables for the following groups:

| CDC Category | Project Mapping | Notes |
|--------------|-----------------|-------|
| White, non-Hispanic | White | Largest group in ND |
| Black, non-Hispanic | Black | Small population in ND |
| Hispanic | Hispanic | Growing population in ND |
| Asian, non-Hispanic | Asian | Includes Pacific Islander in projections |
| American Indian/Alaska Native, non-Hispanic | AIAN | Significant population in ND |
| Total population | Total | For validation/comparison |

### Missing Categories

The following categories from the project's race/ethnicity schema are NOT available in CDC life tables:

- **Native Hawaiian/Pacific Islander (NHPI)**: Use Asian rates as proxy (combined in earlier CDC data)
- **Two or More Races**: Use Total population rates as proxy

## Usage in Projection Engine

### Key Column for Projections: `qx` (probability of dying)

The survival rate used in the cohort component method is:
```
survival_rate = 1 - qx
```

### Age Range

- Standard ages: 0-99 (single year of age)
- Age 100+: Use age 99 rate or apply open-ended age group formula from ADR-002

### Integration with mortality.py

The survival rates should be formatted as:
```python
survival_rates = pd.DataFrame({
    'age': [...],           # 0-90+
    'sex': [...],           # Male, Female
    'race': [...],          # Match config race categories
    'survival_rate': [...]  # 1 - qx
})
```

## Data Quality Notes

1. **Life Expectancy at Birth (e0) - 2023**:
   - White Female: ~81.5 years
   - White Male: ~76.5 years
   - Black Female: ~79.5 years
   - Black Male: ~71.5 years
   - Hispanic Female: ~84.5 years
   - Hispanic Male: ~79.0 years
   - AIAN Female: ~73.5 years
   - AIAN Male: ~68.5 years
   - Asian Female: ~87.5 years
   - Asian Male: ~83.5 years

2. **Validation**: Compare calculated life expectancy against published CDC values

3. **Mortality Improvement**: Base year is 2023; apply improvement factor per ADR-002

## Alternative Data Sources (Not Used)

### SEER Life Tables
- URL: https://seer.cancer.gov/expsurvival/
- Note: Requires SEER*Stat software; life tables distributed with software
- Advantage: Includes more granular race/ethnicity and geography options
- Disadvantage: Not directly downloadable as CSV

### State-Level Life Tables
- CDC publishes state life tables periodically
- 2020 and 2022 state tables available but limited race/ethnicity detail
- URL: https://www.cdc.gov/nchs/data/nvsr/nvsr71/nvsr71-02.pdf (2020)

## Preprocessing Requirements

For use in the projection engine, the combined CSV should be processed to:

1. Filter to sex-specific tables only (exclude "Both sexes" tables)
2. Map race_ethnicity to project standard categories
3. Calculate survival_rate = 1 - qx
4. Handle ages 90+ as open-ended group per ADR-002
5. Create missing race categories (NHPI, Two or More) using proxies

## Citation

National Center for Health Statistics. United States Life Tables, 2023.
National Vital Statistics Reports; vol 74 no 6. Hyattsville, MD:
National Center for Health Statistics. 2025. Available from:
https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-06.pdf

## Download Date

2025-12-28

## Related Documentation

- ADR-002: Survival Rate Processing Methodology (`/docs/adr/002-survival-rate-processing.md`)
- Mortality Module: `/cohort_projections/core/mortality.py`
