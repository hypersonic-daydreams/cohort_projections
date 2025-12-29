# North Dakota Immigrant Profile Data - MANIFEST

## Overview

This directory contains data on North Dakota's immigrant population compiled from multiple authoritative sources. The data supports demographic analysis and population projections by providing context on immigration patterns, labor force participation, and population characteristics.

## Data Sources

### 1. Migration Policy Institute (MPI)
- **URL**: https://www.migrationpolicy.org/data/state-profiles/state/demographics/ND
- **Original Data Source**: U.S. Census Bureau American Community Survey (ACS)
- **Data Vintage**: 2019-2023 pooled ACS estimates (pooled due to ND's smaller sample size)
- **Key Data**:
  - Foreign-born population counts and percentages
  - Region of birth breakdown
  - Growth statistics (123% increase 2010-2022)
  - Top countries of origin

### 2. American Immigration Council (AIC)
- **URL**: https://www.americanimmigrationcouncil.org/research/immigrants-in-north-dakota
- **Interactive Map**: https://map.americanimmigrationcouncil.org/locations/north-dakota
- **PDF Fact Sheet**: https://www.americanimmigrationcouncil.org/wp-content/uploads/2025/01/immigrants_in_north_dakota.pdf
- **Key Data**:
  - Labor force participation by industry
  - Business ownership statistics
  - Educational attainment
  - Citizenship status
  - International migration trends (2022-2023)

### 3. National Immigration Forum / Forum Together
- **URL**: https://forumtogether.org/article/map-to-success-north-dakota/
- **Key Data**:
  - Employment by industry (workers counts)
  - Business ownership
  - Educational attainment

### 4. Census Reporter
- **URL**: https://censusreporter.org/profiles/04000US38-north-dakota/
- **Original Data Source**: U.S. Census Bureau ACS
- **Key Data**:
  - Foreign-born population by region of birth
  - Comparison to national averages

### 5. Data USA
- **URL**: https://datausa.io/profile/geo/north-dakota
- **Original Data Source**: U.S. Census Bureau ACS
- **Key Data**:
  - Top countries of origin with population estimates
  - Year-over-year trends (2022-2023)

### 6. USAFacts
- **URL**: https://usafacts.org/answers/how-many-immigrants-are-in-the-us/state/north-dakota/
- **Original Data Source**: U.S. Census Bureau ACS
- **Key Data**:
  - Time series data (2014-2024)
  - Jobs held by immigrants

## Download Date
2025-12-28

## Methodology

### Data Collection
1. Web scraping and API queries where available
2. Manual extraction from web pages that block automated access
3. Cross-validation between multiple sources for key statistics

### Data Processing
1. Raw data saved as JSON files in `data/raw/immigration/nd_immigrant_profile/`
2. Python script (`scripts/process_nd_immigrant_data.py`) consolidates data
3. Cleaned and standardized data saved as Parquet files in `data/processed/immigration/analysis/`

**Note:** Data has been relocated to project-level directories:
- Raw data: `data/raw/immigration/`
- Processed data: `data/processed/immigration/`

### Column Naming Conventions
- All column names use lowercase with underscores
- State FIPS code included for joining with other datasets
- Data year and source documented for each record

### Data Quality Notes
- Sample sizes for ND are small, requiring pooled 5-year ACS estimates
- Margins of error can be substantial (see countries_of_origin table)
- Some statistics vary between sources due to different ACS vintages
- 2024 estimates are projections/preliminary data

## Output Files

### Raw Source Files (`data/raw/immigration/nd_immigrant_profile/`)
| File | Description |
|------|-------------|
| `migration_policy_institute_nd_demographics.json` | MPI state profile data |
| `american_immigration_council_nd.json` | AIC fact sheet data |
| `forum_together_nd_immigrants.json` | Forum Together profile data |
| `census_reporter_nd_foreign_born.json` | Census Reporter profile |
| `data_usa_nd_foreign_born.json` | Data USA statistics |
| `usafacts_nd_immigrants.json` | USAFacts time series |

### Processed Analysis Files (`data/processed/immigration/analysis/`)
| File | Description |
|------|-------------|
| `nd_immigrant_characteristics.parquet` | Consolidated characteristics and summary statistics |
| `nd_foreign_born_time_series.parquet` | Historical foreign-born population (2010-2024) |
| `nd_countries_of_origin.parquet` | Top countries of origin with percentages |
| `nd_region_of_birth.parquet` | Foreign-born by world region |
| `nd_employment_by_industry.parquet` | Immigrant workers by industry sector |

## Key Statistics Summary

### Population (2023)
- Foreign-born population: ~31,000-33,000 (3.97%-4.27% of state)
- US comparison: 14.28% (ND is about 1/4 of national rate)
- Native-born with immigrant parent: 5% of population

### Growth Trends
- 2010-2022 growth: 123% (largest relative increase of any state)
- 2010 baseline: 16,000 (2.5%)
- 2022 peak: 36,023 (4.9%)

### Top Countries of Origin
1. Philippines (8%)
2. Bhutan (8%)
3. Nepal (8%)
4. Canada (6%)
5. Liberia (6%)

### Region of Birth
1. Africa: 34.24% (notably high compared to US average)
2. Asia: 30.11%
3. Latin America: 16.05%
4. Europe: 10.05%
5. Northern America: 8.43%
6. Oceania: 1.11%

### Labor Force
- Immigrants as % of labor force: 6%
- In production occupations: 13%
- In manufacturing: 11%
- Top industry: Health Care & Social Assistance (6,245 workers)

### Business Ownership
- Immigrant business owners: 1,056
- Share of self-employed: 3%

### Education (Foreign-Born Adults)
- High school or less: 41%
- Some college/associates: 24%
- Bachelor's or higher: 35%
- Brain waste (underemployment): 24%

### Citizenship Status
- Naturalized citizens: 45%
- Undocumented: 23%

## Comparison to US Averages

| Metric | North Dakota | United States |
|--------|-------------|---------------|
| Foreign-born % of population | 4.0% | 14.28% |
| Ratio | ND is about 1/4 of US rate | - |

## Notes for Population Projections

1. **Unique immigration patterns**: ND has high representation from Africa (34%) and refugee resettlement (Bhutan, Nepal, Liberia) compared to typical US patterns dominated by Latin America
2. **Rapid growth**: 123% increase 2010-2022 is significant for projection assumptions
3. **Labor market drivers**: Very low unemployment (1.9%) and high job openings drive immigration
4. **Healthcare pipeline**: Significant Filipino nurse recruitment through formal programs
5. **Growing diaspora communities**: Somali, Haitian, Liberian, Filipino, Indian, Rwandan, Sudanese populations expanding

## Related Documentation

- See also: `../docs/methodology_comparison_sdc_2024.md` for how immigration factors into SDC projections
- SDC 2024 reference materials in `../data/raw/nd_sdc_2024_projections/`
