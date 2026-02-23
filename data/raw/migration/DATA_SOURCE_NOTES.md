# Migration Data Sources

## Primary Source: Census PEP County Components of Change (ADR-035)

**As of ADR-035 (2026-02-03), Census PEP components of change replaced IRS SOI
county-to-county flows as the primary migration data source.** PEP provides
comprehensive migration (domestic + international) for all 53 ND counties across
a 26-year time series (2000-2025). The averaging methodology follows BEBR
multi-period trimmed averaging with Census Bureau-style convergence interpolation
(ADR-036).

---

## Census PEP Data

### Source

- **Program**: U.S. Census Bureau, Population Estimates Program (PEP)
- **Methodology**: Residual method (net migration = population change - births + deaths)
- **Coverage**: All 53 North Dakota counties, 2000-2025
- **Vintages Used**:
  - co-est2009-alldata (2000-2009)
  - co-est2019-alldata (2010-2019)
  - stcoreview-v2025 (2020-2025, Vintage 2025 pre-release)
- **Official URL**: https://www2.census.gov/programs-surveys/popest/datasets/
- **PEP Archive**: `~/workspace/shared-data/census/popest/` (per ADR-034)

### Processed Files

| File | Location | Rows | Description |
|------|----------|------|-------------|
| pep_county_components_2000_2025.parquet | `data/processed/` | 1,378 | Primary: 53 counties x 26 years, Vintage 2025 |
| pep_county_components_2000_2025.csv | `data/processed/` | 1,378 | CSV copy of the above |
| pep_county_components_2000_2024.parquet | `data/processed/` | 1,325 | Prior version: 53 counties x 25 years |
| pep_county_components_2000_2024.csv | `data/processed/` | 1,325 | CSV copy of the above |

### Column Definitions

| Column | Type | Description |
|--------|------|-------------|
| state_fips | string | State FIPS code ("38" for North Dakota) |
| county_fips | string | County FIPS code (3-digit, e.g., "001") |
| state_name | string | "North Dakota" |
| county_name | string | County name (e.g., "Adams County") |
| year | int | Calendar year (2000-2025) |
| netmig | int | Net migration (domestic + international combined) |
| intl_mig | int | International migration component |
| domestic_mig | int | Domestic migration component |
| residual | int | Statistical residual from PEP estimation |
| dataset_id | string | Source PEP vintage dataset identifier |
| estimate_type | string | "postcensal" |
| revision_status | string | Revision status of the estimate |
| uncertainty_level | string | Data quality indicator (e.g., "moderate") |
| geoid | string | Full 5-digit FIPS (state + county, e.g., "38001") |
| is_preferred_estimate | bool | Whether this is the preferred estimate for overlap years |

### Processing Scripts

| Script | Purpose |
|--------|---------|
| `scripts/data_processing/extract_pep_county_migration_with_metadata.py` | Extracts county components from PEP archive with metadata tracking |
| `scripts/data_processing/extract_pep_county_migration.py` | Simpler extraction variant |
| `scripts/data/ingest_stcoreview.py` | Ingests Vintage 2025 stcoreview Excel file |
| `scripts/data_processing/analyze_pep_regimes.py` | Migration regime analysis by county |
| `scripts/data_processing/process_pep_rates.py` | Processes PEP data into migration rates |

### Downstream Processing

PEP net migration totals are further processed into age/sex-specific migration rates through:

1. **Residual migration calculation** (`cohort_projections/data/process/residual_migration.py`): Computes age/sex-distributed net migration rates from PEP components and survival data
2. **Convergence interpolation** (`cohort_projections/data/process/convergence_interpolation.py`): Implements Census Bureau-style time-varying convergence from recent rates toward long-term averages, with age-aware rate caps (ADR-043)
3. **BEBR multi-period averaging** (ADR-036): Trimmed average of 4 overlapping base periods (short: 2019-2024, medium: 2014-2024, long: 2005-2024, full: 2000-2024)

### Configuration

In `config/projection_config.yaml`:
```yaml
rates:
  migration:
    domestic:
      method: "PEP_components"
      source: "Census_PEP"
      averaging_method: "BEBR_multiperiod"
    international:
      method: "PEP_included"
      allocation: "proportional"
```

Primary PEP input path: `pipeline.data_processing.migration.pep_input: "data/processed/pep_county_components_2000_2025.parquet"`

### State-Level Net Migration Summary (from PEP)

| Period | Avg Net Migration/Year | Regime |
|--------|----------------------|--------|
| 2000-2004 | -2,278 | Pre-Bakken (decline) |
| 2005-2009 | -765 | Pre-Bakken (stabilizing) |
| 2010-2015 | +9,692 | Bakken boom |
| 2016-2021 | -2,435 | Bust + COVID |
| 2022-2025 | +2,720 | Recovery |
| 2000-2025 | +1,311 | Full period |

### Key Advantages Over IRS Data

1. **Comprehensive**: Captures both domestic and international migration (IRS misses ~1,100-1,200 international migrants/year)
2. **Temporal depth**: 26 years of history vs. 4 years for IRS, enabling regime analysis and robust averaging
3. **Current**: Includes 2020-2025 Vintage 2025 data showing recovery period
4. **Methodological alignment**: Matches SDC 2024 approach and standard demographic practice
5. **Decomposition**: Separates domestic vs. international components, enabling ADR-039 international-only migration factor

### Data Limitations

1. **Residual method**: Net migration calculated as population change minus natural increase; includes measurement error
2. **No age/sex detail**: PEP components are county totals only; age/sex distribution requires allocation algorithms
3. **No origin-destination**: Net migration only; no information about where migrants come from or go to
4. **Vintage differences**: Methodology changes between Census decades may introduce discontinuities at 2010 and 2020 boundaries
5. **Residual column**: Non-zero residuals indicate estimation artifacts; should be monitored but not included in migration totals

---

## Historical: IRS SOI Data (Superseded by ADR-035)

> **Note**: As of 2026-02-03, IRS SOI data is no longer the primary migration source for projections.
> It was replaced by Census PEP county components per ADR-035. The IRS data files remain in this
> directory for reference and potential future use (e.g., directional flow analysis). See ADR-035
> section "Deferred to Future Work" for potential blending of PEP totals with IRS directional patterns.

### Overview

This directory contains IRS Statistics of Income (SOI) county-to-county migration data
for use in the North Dakota cohort projection system. The data tracks tax filer address
changes between tax years to estimate domestic migration flows.

### Data Source

- **Source**: IRS Statistics of Income (SOI) Division
- **Official URL**: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- **Download Date**: 2025-12-28

### Files Downloaded

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
| nd_migration_processed.csv | 2019-2022 | Processed ND extract | 8 KB | 212 |

### Column Definitions

#### Inflow Files (countyinflow*.csv)
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

#### Outflow Files (countyoutflow*.csv)
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

#### nd_migration_processed.csv
Pre-processed ND extract (53 counties x 4 years = 212 rows):

| Column | Type | Description |
|--------|------|-------------|
| county_fips | string | 5-digit FIPS code (e.g., "38001") |
| county_name | string | County name |
| year | int | Calendar year (2019-2022) |
| inflow_n2 | int | Inflow exemptions (approximate persons) |
| outflow_n2 | int | Outflow exemptions (approximate persons) |
| net_migration | int | Net migration (inflow - outflow) |
| inflow_domestic | int | Domestic inflow |
| outflow_domestic | int | Domestic outflow |

### Special FIPS Codes

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

### IRS Data Limitations

1. **Tax Filer Coverage**: Only ~70% of population files tax returns
   - Excludes children without income
   - Excludes non-filers (low income, elderly)
   - Excludes undocumented immigrants

2. **No International Migration**: Only captures domestic address changes (~1,100-1,200 international migrants/year missing for ND)

3. **No Demographic Detail**: Data is aggregate only
   - No age breakdown
   - No sex breakdown
   - No race/ethnicity breakdown
   - Must use distribution algorithms (see ADR-003)

4. **Privacy Suppression**: Flows with <10 returns are shown as -1

5. **Timing**: Reflects address changes between tax filing years
   - 2021-2022 data reflects moves from 2021 to 2022 filing addresses

6. **Exemptions vs Persons**: The n2 (exemptions) column approximates persons
   but may differ from actual household size after 2018 tax law changes

7. **Period Selection Bias**: Available IRS period (2019-2022) captured average net migration of -987/year, representing the worst recent conditions (COVID + Bakken bust aftermath). Full PEP period (2000-2025) shows +1,311/year average.

### Download URLs (for reference)

- 2021-2022: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2021-2022
- 2020-2021: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2020-2021
- 2019-2020: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2019-2020
- 2018-2019: https://www.irs.gov/statistics/soi-tax-stats-migration-data-2018-2019

---

## Historical Notes

Prior to ADR-035 (2026-02-03), the project used IRS SOI county-to-county migration
flows (2019-2022) as the sole migration data source. The IRS data was domestic-only,
covering 4 years of a period dominated by COVID disruptions and Bakken bust aftermath.
This produced an average net migration assumption of -987 people/year, resulting in
baseline projections showing ND declining to ~755,000 by 2045.

The switch to Census PEP components of change (ADR-035) added:
- International migration (~1,100-1,200/year, previously missing entirely)
- 26 years of historical context (2000-2025) instead of 4 years
- Recent recovery data (2022-2025) showing return to positive net migration

Combined with BEBR multi-period averaging (ADR-036) and Census Bureau-style convergence
interpolation, the updated methodology produces baseline projections with modest net
in-migration, aligning with the 2022-2025 recovery trend and long-term historical average.
The total impact of the data source switch was estimated at ~74,000-80,000 people by 2045
(approximately 10% of the ND population).

The IRS data files remain in this directory for potential future use in directional
flow analysis (blending PEP net totals with IRS origin-destination patterns).

---

## References

### ADRs
- **ADR-035**: Census PEP Components of Change for Migration Inputs (`docs/governance/adrs/035-migration-data-source-census-pep.md`)
- **ADR-036**: Migration Averaging Methodology (`docs/governance/adrs/036-migration-averaging-methodology.md`)
- **ADR-034**: Census PEP Data Archive (`docs/governance/adrs/034-census-pep-data-archive.md`)
- **ADR-039**: International-Only Migration Factor (`docs/governance/adrs/039-international-only-migration-factor.md`)
- **ADR-043**: Migration Rate Cap (`docs/governance/adrs/043-migration-rate-cap.md`)
- **ADR-003**: Migration Rate Processing Methodology (`docs/governance/adrs/003-migration-rate-processing.md`)

### External Sources
- Census PEP Methodology: https://www.census.gov/programs-surveys/popest/technical-documentation/methodology.html
- Census PEP Datasets: https://www2.census.gov/programs-surveys/popest/datasets/
- IRS SOI Migration Data: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- IRS Migration Documentation (PDF): `irs_migration_documentation_2122.pdf` (in this directory)
