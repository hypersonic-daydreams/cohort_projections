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
| **Format** | Excel (.xls/.xlsx) for FY2012-2020; PDF for FY2021-2024 (mixed text/encoded) |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY2002-FY2024 |
| **Location** | `data/raw/immigration/refugee_arrivals/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/process_refugee_data.py` |
| **Alignment Notes** | FY2024 (Oct 2023-Sep 2024) overlaps with CY2023 Q4 and CY2024 Q1-Q3. Direct comparison with CY population requires adjustment. |

**Files**:
- `FY_2012_Arrivals_by_State_and_Nationality.xls` through `FY_2020_Arrivals_by_State_and_Nationality.xlsx` (Excel)
- `FY 2021 Arrivals by State and Nationality as of 30 Sep 2021.pdf` (PDF)
- `FY 2022 Arrivals by State and Nationality as of 30 Sep 2022.pdf` (PDF; image-only after page 1)
- `FY 2023 Refugee Arrivals by State and Nationality as of 30 Sep 2023.pdf` (PDF)
- `FY 2024 Arrivals by State and Nationality as of 30 Oct 2024_updated.pdf` (PDF; encoded text)
- Legacy copies: `FY_2021_Arrivals_by_State_and_Nationality.pdf` through `FY_2024_Arrivals_by_State_and_Nationality.pdf`
- `orr_prm_1975_2018_v1.dta` (Stata - academic dataset FY2002-2011)
- `PRM_Refugee_Admissions_Report_Nov_2025.xlsx` (national-level monthly admissions, no state breakdown)

**Archive URLs (downloaded 2026-01-04):**
- https://www.rpc.state.gov/documents/FY%202021%20Arrivals%20by%20State%20and%20Nationality%20as%20of%2030%20Sep%202021.pdf
- https://www.rpc.state.gov/documents/FY%202022%20Arrivals%20by%20State%20and%20Nationality%20as%20of%2030%20Sep%202022.pdf
- https://www.rpc.state.gov/documents/FY%202023%20Refugee%20Arrivals%20by%20State%20and%20Nationality%20as%20of%2030%20Sep%202023.pdf
- https://www.rpc.state.gov/documents/FY%202024%20Arrivals%20by%20State%20and%20Nationality%20as%20of%2030%20Oct%202024_updated.pdf

**Processed Output Notes:**
- `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet` includes extracted state×nationality panels for FY2021–FY2024 from RPC PDFs (FY2022 uses OCR tesseract/Acrobat; FY2024 uses OCR). State coverage remains partial but improved (FY2021=46, FY2022=49, FY2023=48, FY2024=50); remaining gaps reflect omissions in source PDFs and are left missing for downstream handling.
- Monthly and PEP-year aligned outputs (FY2021–FY2024): `refugee_arrivals_by_state_nationality_monthly.parquet`, `refugee_arrivals_by_state_nationality_pep_year.parquet`, and `refugee_fy_month_to_pep_year_crosswalk.csv`.
- OCR-enhanced PDF inputs are stored in `data/interim/immigration/ocr/` and used for FY2022/FY2024 extraction (including `FY2022_Arrivals_by_State_and_Nationality_ocr_acrobat.pdf`).

**Temporal Conversion**:
```
FY2024 = Oct 1, 2023 - Sep 30, 2024
       = CY2023 Q4 + CY2024 Q1-Q3

To align with CY2024 population (Jul 1 estimate):
  - FY2024 refugees arrived before Jul 1, 2024: ~75% of FY total
  - FY2025 refugees (Oct-Dec 2024): Not yet included in CY2024 pop
```

---

#### 1.1b Amerasian & SIV Arrivals (RPC)

| Attribute | Value |
|-----------|-------|
| **Source** | Refugee Processing Center (rpc.state.gov) |
| **Format** | PDF |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY2021-FY2024 (additional years available in archives) |
| **Location** | `data/raw/immigration/refugee_arrivals/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/process_siv_amerasian_data.py` |
| **Alignment Notes** | Same FY-to-PEP alignment logic as refugee arrivals. |

**Files**:
- `FY2021 Amerasian & SIV Arrivals by Nationality and State.pdf`
- `FY 2022 Amerasian & SIV Arrivals by Nationality and State.pdf`
- `FY 2023 Amerasian & SIV Arrivals by Nationality and State.pdf`
- `FY 2024 Amerasian & SIV Arrivals by Nationality and State_updated_2025_01_14.pdf`

**Processed Outputs:**
- `data/processed/immigration/analysis/amerasian_siv_arrivals_by_state_nationality.parquet`
- `data/processed/immigration/analysis/amerasian_siv_arrivals_by_state_nationality_monthly.parquet`
- `data/processed/immigration/analysis/amerasian_siv_arrivals_by_state_nationality_pep_year.parquet`

**Archive URLs (downloaded 2026-01-04):**
- https://www.rpc.state.gov/documents/FY2021%20Amerasian%20%26%20SIV%20Arrivals%20by%20Nationality%20and%20State.pdf
- https://www.rpc.state.gov/documents/FY%202022%20Amerasian%20%26%20SIV%20Arrivals%20by%20Nationality%20and%20State.pdf
- https://www.rpc.state.gov/documents/FY%202023%20Amerasian%20%26%20SIV%20Arrivals%20by%20Nationality%20and%20State.pdf
- https://www.rpc.state.gov/documents/FY%202024%20Amerasian%20%26%20SIV%20Arrivals%20by%20Nationality%20and%20State_updated_2025_01_14.pdf

---

#### 1.2 Census Foreign-Born Population (ACS B05006)

| Attribute | Value |
|-----------|-------|
| **Source** | American Community Survey (Census Bureau) |
| **Format** | CSV |
| **Temporal Basis** | **CALENDAR YEAR (5-year rolling average)** |
| **Years Available** | 2009-2023 |
| **Location** | `data/raw/immigration/census_foreign_born/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/process_b05006.py` |
| **Alignment Notes** | "2023" = 2019-2023 average, centered on Dec 2021. Population STOCK, not flow. |

**Files**: `b05006_states_{year}.csv` for years 2009-2023

---

#### 1.2b ACS Mobility (B07007) - Moved From Abroad

| Attribute | Value |
|-----------|-------|
| **Source** | American Community Survey (Census Bureau) |
| **Format** | CSV |
| **Temporal Basis** | **CALENDAR YEAR (5-year rolling average)** |
| **Years Available** | 2010-2023 (B07007 group; 2009 not available) |
| **Location** | `data/raw/immigration/acs_migration/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/process_b07007.py` |
| **Alignment Notes** | "2023" = 2019-2023 average. Flow proxy (moved-from-abroad), not a direct net migration measure. |

**Files**: `b07007_states_{year}.csv` for years 2010-2023, plus `b07007_states_all_years.csv` and year-specific label JSON files.

**Processed Outputs:**
- `data/processed/immigration/analysis/acs_moved_from_abroad_by_state.parquet`
- `data/processed/immigration/analysis/acs_moved_from_abroad_by_state.csv`
- `data/processed/immigration/analysis/acs_moved_from_abroad_by_state_validation.md`

---

#### 1.3 DHS Legal Permanent Residents (LPR)

| Attribute | Value |
|-----------|-------|
| **Source** | DHS Office of Immigration Statistics |
| **Format** | Excel, CSV |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY2000-FY2023 yearbook tables (FY2012 state totals sourced from yearbook PDF) |
| **Location** | `data/raw/immigration/dhs_lpr/` |
| **Alignment Notes** | DHS uses federal FY. Same alignment issue as refugee data. |

**Notes:** Pre-2007 yearbook ZIPs require manual browser download (CDN blocks programmatic access). State-level time series now covers FY2000-FY2023; FY2012 state totals were extracted from `data/raw/immigration/dhs_lpr/yearbook_pdfs/Yearbook_Immigration_Statistics_2012.pdf` due to missing main ZIP.

---

#### 1.4 DHS Naturalizations

| Attribute | Value |
|-----------|-------|
| **Source** | DHS Office of Immigration Statistics |
| **Format** | Excel, CSV |
| **Temporal Basis** | **FISCAL YEAR (Oct 1 - Sep 30)** |
| **Years Available** | FY1999-FY2023 (plus supplemental FY2012 ZIP staged) |
| **Location** | `data/raw/immigration/dhs_naturalizations/` |
| **Alignment Notes** | DHS uses federal FY. |

---

#### 1.5 Census PEP Components of Change (Migration)

| Attribute | Value |
|-----------|-------|
| **Source** | Census Population Estimates Program (PEP) |
| **Format** | CSV |
| **Temporal Basis** | **CALENDAR YEAR (as of July 1)** |
| **Years Available** | 2000-2024 (combined vintages) |
| **Raw Location** | `data/raw/immigration/census_population_estimates/` |
| **Processing Script** | `sdc_2024_replication/data_immigration_policy/scripts/combine_census_vintages.py` |
| **Alignment Notes** | Net international and domestic migration components; used for regime-aware modeling. |

**Processed Outputs:**
- `data/processed/immigration/state_migration_components_2000_2024.csv`
- `data/processed/immigration/state_migration_components_2000_2024_with_regime.csv` (adds `regime_pre_2010`, `regime_post_2010`, `regime_post_2020`, `regime_period`)
- `data/processed/immigration/state_migration_decade_summary.csv`

---

#### 1.6 DHS Refugees and Asylees (Yearbook Tables + Flow Reports)

| Attribute | Value |
|-----------|-------|
| **Source** | DHS Office of Immigration Statistics |
| **Format** | ZIP (tables), PDF (annual flow reports) |
| **Temporal Basis** | **FISCAL YEAR** |
| **Years Available** | FY2000-FY2006 and FY2012 (yearbook tables), FY2019-FY2023 (flow reports) |
| **Location** | `data/raw/immigration/dhs_refugees_asylees/` |
| **Alignment Notes** | Stored for provenance; no current processing pipeline in v0.8.6. |

---

#### 1.7 DHS Nonimmigrant Admissions (Yearbook Tables)

| Attribute | Value |
|-----------|-------|
| **Source** | DHS Office of Immigration Statistics |
| **Format** | ZIP |
| **Temporal Basis** | **FISCAL YEAR** |
| **Years Available** | FY2004 (supplementary tables only), FY2012 (yearbook + supplemental tables) |
| **Location** | `data/raw/immigration/dhs_nonimmigrants/` |
| **Alignment Notes** | Stored for provenance; not used in current analyses. |

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
