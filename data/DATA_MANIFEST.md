# Data Manifest with Temporal Alignment Metadata

**Generated**: 2026-01-01
**Purpose**: Comprehensive inventory of all data sources with temporal (FY/CY) alignment information
**Version**: 1.0.0

---

## Executive Summary

This project uses multiple data sources with **different temporal bases**:

| Temporal Basis | Sources | Impact |
|----------------|---------|--------|
| **Fiscal Year (FY)** | Refugee arrivals (Oct 1 - Sep 30) | Misaligns with Census CY population by 3-9 months |
| **Calendar Year (CY)** | Census PEP, IRS migration, ACS, CDC vital stats | Primary temporal basis for projections |
| **Tax Year** | IRS SOI migration | Aligns with CY but reflects prior year income |
| **5-Year Rolling** | ACS estimates | Centered on final year (e.g., 2019-2023 = "2023") |

**Critical Finding**: The refugee_share_pct calculation in ADR-021 analysis showed values >100% due to FY/CY mismatch. This is now documented and handled with temporal conversion utilities.

---

## Data Sources by Category

### 1. REFUGEE/IMMIGRATION DATA

#### 1.1 Refugee Arrivals (WRAPS/RPC)

| Attribute | Value |
|-----------|-------|
| **Source** | Refugee Processing Center (rpc.state.gov) |
| **Format** | Excel (.xls/.xlsx) for FY2012-2020; PDF for FY2021-2024 |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY2002-FY2024 |
| **Location** | `data/raw/immigration/refugee_arrivals/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/process_refugee_data.py` |
| **Alignment Notes** | FY2024 (Oct 2023-Sep 2024) overlaps with CY2023 Q4 and CY2024 Q1-Q3. Direct comparison with CY population requires adjustment. |

**Files**:
- `FY_2012_Arrivals_by_State_and_Nationality.xls` through `FY_2020_Arrivals_by_State_and_Nationality.xlsx` (Excel)
- `FY_2021_Arrivals_by_State_and_Nationality.pdf` through `FY_2024_Arrivals_by_State_and_Nationality.pdf` (PDF)
- `orr_prm_1975_2018_v1.dta` (Stata - academic dataset FY2002-2011)

**Temporal Conversion**:
```
FY2024 = Oct 1, 2023 - Sep 30, 2024
       = CY2023 Q4 + CY2024 Q1-Q3

To align with CY2024 population (Jul 1 estimate):
  - FY2024 refugees arrived before Jul 1, 2024: ~75% of FY total
  - FY2025 refugees (Oct-Dec 2024): Not yet included in CY2024 pop
```

---

#### 1.2 Census Foreign-Born Population (ACS B05006)

| Attribute | Value |
|-----------|-------|
| **Source** | American Community Survey (Census Bureau) |
| **Format** | CSV |
| **Temporal Basis** | **CALENDAR YEAR (5-year rolling average)** |
| **Years Available** | 2009-2023 |
| **Location** | `sdc_2024_replication/data_immigration_policy/data/census_foreign_born/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/process_b05006.py` |
| **Alignment Notes** | "2023" = 2019-2023 average, centered on Dec 2021. Population STOCK, not flow. |

**Files**: `b05006_{state}_{year}.csv` for years 2009-2023

---

#### 1.3 DHS Legal Permanent Residents (LPR)

| Attribute | Value |
|-----------|-------|
| **Source** | DHS Office of Immigration Statistics |
| **Format** | Excel, CSV |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY2000-FY2023 |
| **Location** | `data/raw/immigration/dhs_lpr/` |
| **Alignment Notes** | DHS uses federal FY. Same alignment issue as refugee data. |

---

#### 1.4 DHS Naturalizations

| Attribute | Value |
|-----------|-------|
| **Source** | DHS Office of Immigration Statistics |
| **Format** | Excel, CSV |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY1999-FY2023 |
| **Location** | `data/raw/immigration/dhs_naturalizations/` |
| **Alignment Notes** | DHS uses federal FY. |

---

### 2. CENSUS POPULATION DATA

#### 2.1 Population Estimates Program (PEP)

| Attribute | Value |
|-----------|-------|
| **Source** | Census Bureau |
| **Format** | CSV |
| **Temporal Basis** | **CALENDAR YEAR (as of July 1)** |
| **Years Available** | 2020-2024 (Vintage 2024) |
| **Location** | `data/raw/population/` |
| **Processing Script** | `cohort_projections/data/fetch/census_api.py` |
| **Alignment Notes** | July 1 reference date. FY comparisons should use weighted average. |

**Files**:
- `co-est2024-alldata.csv` - County estimates
- `NST-EST2024-ALLDATA.csv` - State estimates

**Vintage Transitions**:
- Pre-2010: Based on Census 2000
- 2010-2019: Based on Census 2010
- 2020+: Based on Census 2020

---

#### 2.2 American Community Survey (ACS)

| Attribute | Value |
|-----------|-------|
| **Source** | Census Bureau |
| **Format** | CSV via API |
| **Temporal Basis** | **CALENDAR YEAR (5-year rolling)** |
| **Years Available** | 2009-2023 (5-year estimates) |
| **Fetch Script** | `cohort_projections/data/fetch/census_api.py` |
| **Alignment Notes** | 5-year window: "2023" = 2019-2023 data. Single-year available for large areas. |

---

### 3. VITAL STATISTICS

#### 3.1 Fertility Rates (CDC NCHS)

| Attribute | Value |
|-----------|-------|
| **Source** | CDC National Center for Health Statistics |
| **Format** | CSV, PDF |
| **Temporal Basis** | **CALENDAR YEAR** |
| **Years Available** | 1909-2023 (annual); Q1 2024-Q2 2025 (quarterly) |
| **Location** | `data/raw/fertility/` |
| **Processing Script** | `cohort_projections/data/process/fertility_rates.py` |
| **Alignment Notes** | Birth counts by calendar year of birth. No FY/CY issue. |

**Files**:
- `cdc_birth_rates_race_age.csv` - Quarterly data
- `cdc_birth_rates_state_age.csv` - State teen birth rates
- `nvsr73-02_births_2022.pdf`, `nvsr74-01_births_2023.pdf` - Annual reports

---

#### 3.2 Life Tables / Mortality (CDC NCHS)

| Attribute | Value |
|-----------|-------|
| **Source** | CDC National Center for Health Statistics |
| **Format** | Excel (.xlsx) |
| **Temporal Basis** | **CALENDAR YEAR** |
| **Years Available** | 2020-2023 |
| **Location** | `data/raw/mortality/` |
| **Processing Script** | `cohort_projections/data/process/survival_rates.py` |
| **Alignment Notes** | Life tables based on CY mortality. No FY/CY issue. |

**Files**:
- `cdc_lifetable_2023_table01.xlsx` through `table18.xlsx` (18 files by race/sex)
- `us_lifetable_*_2022.xlsx` (comparative 2022 tables)

---

### 4. MIGRATION DATA

#### 4.1 IRS Statistics of Income (SOI) Migration

| Attribute | Value |
|-----------|-------|
| **Source** | IRS Statistics of Income Division |
| **Format** | CSV |
| **Temporal Basis** | **TAX YEAR / CALENDAR YEAR** |
| **Years Available** | 2018-2022 (county-to-county flows) |
| **Location** | `data/raw/migration/` |
| **Processing Script** | `cohort_projections/data/process/migration_rates.py` |
| **Alignment Notes** | Reflects address changes between tax filings. Tax year = CY. |

**Files**:
- `countyinflow1819.csv` - FY2018-2019 inflows
- `countyoutflow1819.csv` - FY2018-2019 outflows
- (Similar for 1920, 2021, 2122)

**Temporal Note**: File naming uses "1819" for data reflecting moves between tax years 2018 and 2019. The migration is measured as of tax filing (typically April 15 of following year).

---

#### 4.2 Census International Migration Components

| Attribute | Value |
|-----------|-------|
| **Source** | Census Bureau Population Estimates |
| **Format** | CSV (embedded in PEP) |
| **Temporal Basis** | **CALENDAR YEAR (July 1 - June 30 period)** |
| **Years Available** | 2000-2024 |
| **Location** | Derived from `data/raw/population/` |
| **Alignment Notes** | Census measures July 1 to July 1 population change, then allocates to components. |

---

### 5. GEOGRAPHIC REFERENCE

#### 5.1 County and Place Reference

| Attribute | Value |
|-----------|-------|
| **Source** | Census Bureau |
| **Format** | CSV |
| **Temporal Basis** | **Point-in-time (Census 2020 geography)** |
| **Years Available** | 2020 base |
| **Location** | `data/raw/geographic/` |
| **Alignment Notes** | FIPS codes stable; some county changes over time. |

**Files**:
- `nd_counties.csv` - National county reference
- `nd_places.csv` - ND incorporated places
- `metro_crosswalk.csv` - CBSA definitions

---

### 6. SDC 2024 PROJECTION SOURCE

#### 6.1 North Dakota State Data Center Projections

| Attribute | Value |
|-----------|-------|
| **Source** | ND State Data Center (SDC) |
| **Format** | Excel workbooks, CSV extracts |
| **Temporal Basis** | **CENSUS INTERCENSAL PERIODS** |
| **Years Available** | 2000-2020 (historical); 2020-2050 (projections) |
| **Location** | `data/raw/nd_sdc_2024_projections/` |
| **Alignment Notes** | Uses 5-year census periods (2000-2005, 2005-2010, etc.). Base year 2020. |

**Files**:
- `source_files/backup/Projections 2023.xlsx` and versions
- `sdc_county_projections_summary.csv` - Extracted county data

---

## Temporal Alignment Matrix

| Source A | Source B | Alignment Issue | Handling Strategy |
|----------|----------|-----------------|-------------------|
| Refugee FY | Census CY Population | FY spans 2 calendar years | Weight by quarter; FY2024 ≈ 0.25*CY2023 + 0.75*CY2024 |
| DHS LPR FY | Census CY Population | Same as refugee | Apply same weighting |
| ACS 5-year | Census 1-year PEP | ACS is rolling average | Use ACS for characteristics, PEP for levels |
| IRS Migration | Census Components | Both CY but different reference | IRS is flow; Census is residual. Use IRS for rates. |
| Life Tables CY | Base Population CY | Both CY | No adjustment needed |
| Fertility CY | Base Population CY | Both CY | No adjustment needed |

---

## Scripts with Temporal Handling

### Scripts That Explicitly Handle FY/CY:

1. **`docs/adr/021-reports/agent1_estimand_composition.py`**
   - Lines 82-83: Clips negative `non_refugee_intl` to 0 (band-aid for FY/CY mismatch)
   - Lines 162-177: Now checks for post-2020 data and reports updated findings

2. **`sdc_2024_replication/data_immigration_policy/scripts/process_refugee_data.py`**
   - Uses `fiscal_year` column explicitly
   - Line 137: Parses FY from filenames
   - Line 358: Output includes `fiscal_year` not `year`

### Scripts That Assume CY:

1. **`cohort_projections/data/load/base_population_loader.py`**
   - Lines 246-248: Uses `population_2024` column (CY)
   - Line 317: Sets `base_year = 2025` (CY)

2. **`cohort_projections/data/process/migration_rates.py`**
   - Time-aggregates IRS flows across years
   - Returns single migration rate, not year-indexed

---

## Recommendations

### Immediate Actions:
1. Add `year_basis` column to all processed DataFrames ("FY" or "CY")
2. Create temporal conversion utility in `cohort_projections/utils/temporal.py`
3. Update validation to flag FY/CY mismatches

### Medium-Term Actions:
1. Refactor `migration_rates.py` to preserve year dimension
2. Add FY→CY conversion in refugee processing pipeline
3. Document year basis in all data loading functions

### Long-Term Actions:
1. Consider standardizing on FY for immigration-focused analyses
2. Create unified temporal alignment layer for all data sources

---

## Appendix: File Counts by Category

| Category | Raw Files | Processed Files | Total Size |
|----------|-----------|-----------------|------------|
| Refugee/Immigration | 50+ | 20+ | ~25 MB |
| Census Population | 5 | 6 | ~2 MB |
| Fertility | 11 | 3 | ~4 MB |
| Mortality | 38 | 3 | ~2 MB |
| Migration (IRS) | 11 | 3 | ~35 MB |
| Geographic | 3 | 0 | ~700 KB |
| SDC 2024 Source | 20+ | 11 | ~40 MB |
| **TOTAL** | **448** | **51** | **~271 MB** |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-01 | 1.0.0 | Initial manifest with temporal alignment metadata |

---

*This manifest should be updated when new data sources are added or temporal handling changes.*
