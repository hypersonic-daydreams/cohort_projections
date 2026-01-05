# Refugee Resettlement Data Manifest

**Download Date:** 2025-12-28 (updated with historical data)
**Purpose:** Track refugee arrivals by state and nationality for North Dakota population analysis

---

## Data Sources

### 1. ORR/PRM Academic Dataset (FY 2002-2011 Historical Data)

**Source URL:** https://www.refugeeresettlementdata.com/data.html

**Provider:** Academic research dataset compiled by Dreher, A., Langlotz, S., Matzat, J. and Parsons, C. (2020)

**Citation:** Dreher, A., Langlotz, S., Matzat, J. and Parsons, C. (2020). Immigration, Political Ideologies and the Polarization of American Politics. CESifo WP 8789.

**Data Description:** This dataset merges digitized Office of Refugee Resettlement (ORR) records with publicly available information from the Bureau for Population, Refugees and Migration (PRM). It covers the geo-locations of the universe of refugees that entered the U.S. between 1975 and 2018.

**Downloaded File:**
| File | Coverage | Size | Description |
|------|----------|------|-------------|
| orr_prm_1975_2018_v1.dta | FY 1975-2018 | 20MB | Stata file with city-level geocoded refugee resettlement data |

**Note:** We use this dataset for FY 2002-2011 only, as WRAPS Excel files are available for FY 2012+. The academic dataset totals may differ slightly from official WRAPS data due to different data collection methodologies (e.g., inclusion of Amerasians and SIVs).

---

### 2. WRAPS - Refugee Processing Center (FY 2012-2020)

**Source URL:** https://www.rpc.state.gov/admissions-and-arrivals/ (also https://www.wrapsnet.org/admissions-and-arrivals/)

**Provider:** U.S. Department of State, Bureau of Population, Refugees, and Migration, Office of Admissions - Refugee Processing Center

**Data Description:** Refugee arrivals by placement state and nationality. These reports provide information on the total number of refugees admitted through the U.S. Refugee Admissions Program (USRAP) for each fiscal year, broken down by state of resettlement, primary applicant nationality, and month of admission.

#### Downloaded Files - Excel Format (FY 2012-2020)

| File | Fiscal Year | Size | Records | States |
|------|-------------|------|---------|--------|
| FY_2012_Arrivals_by_State_and_Nationality.xls | 2012 | 81KB | 811 | 49 |
| FY_2013_Arrivals_by_State_and_Nationality.xls | 2013 | 84KB | 848 | 50 |
| FY_2014_Arrivals_by_State_and_Nationality.xls | 2014 | 84KB | 850 | 49 |
| FY_2015_Arrivals_by_State_and_Nationality.xls | 2015 | 94KB | 934 | 49 |
| FY_2016_Arrivals_by_State_and_Nationality.xls | 2016 | 90KB | 1,023 | 49 |
| FY_2017_Arrivals_by_State_and_Nationality.xls | 2017 | 90KB | 988 | 50 |
| FY_2018_Arrivals_by_State_and_Nationality.xls | 2018 | 65KB | 730 | 49 |
| FY_2019_Arrivals_by_State_and_Nationality.xlsx | 2019 | 22KB | 754 | 49 |
| FY_2020_Arrivals_by_State_and_Nationality.xlsx | 2020 | 27KB | 632 | 47 |

#### Downloaded Files - PDF Format (FY 2021-2024)

These files are in PDF format (Tableau exports). FY2021/FY2023 are parseable with PDF text extraction; FY2022 is image-only after page 1; FY2024 uses encoded text that does not round-trip cleanly.

| File | Fiscal Year | Size |
|------|-------------|------|
| FY_2021_Arrivals_by_State_and_Nationality.pdf | 2021 | 350KB |
| FY_2022_Arrivals_by_State_and_Nationality.pdf | 2022 | 2.3MB |
| FY_2023_Arrivals_by_State_and_Nationality.pdf | 2023 | 507KB |
| FY_2024_Arrivals_by_State_and_Nationality.pdf | 2024 | 2.6MB |

#### RPC Archive “As Of” PDFs (FY 2021-2024)

These archive files are the primary sources used for FY2021–FY2024 extraction.
OCR-enhanced copies are stored under `data/interim/immigration/ocr/` for extraction workflows.
- `FY2022_Arrivals_by_State_and_Nationality_ocr_acrobat.pdf`
- `FY2022_Arrivals_by_State_and_Nationality_ocr_tesseract.pdf`
- `FY2022_Arrivals_by_State_and_Nationality_ocr.pdf`
- `FY2024_Arrivals_by_State_and_Nationality_ocr.pdf`

| File | Fiscal Year | Notes |
|------|-------------|-------|
| FY 2021 Arrivals by State and Nationality as of 30 Sep 2021.pdf | 2021 | Parsed (partial panel; 46 states) |
| FY 2022 Arrivals by State and Nationality as of 30 Sep 2022.pdf | 2022 | OCR extraction yields 49-state panel; Hawaii/Wyoming omitted in source and left missing |
| FY 2023 Refugee Arrivals by State and Nationality as of 30 Sep 2023.pdf | 2023 | Parsed (partial panel; 48 states) |
| FY 2024 Arrivals by State and Nationality as of 30 Oct 2024_updated.pdf | 2024 | Encoded text; OCR extraction used (partial panel; 50 states) |

#### Additional File - National Admissions Report

| File | Description | Size |
|------|-------------|------|
| PRM_Refugee_Admissions_Report_Nov_2025.xlsx | Refugee admissions by nationality and month, FY 2001-2026 (national level, no state breakdown) | 228KB |

---

### 3. American Immigration Council FOIA Data

**Source URL:** https://www.americanimmigrationcouncil.org/foia-request/refugee-resettlement-data/

**Data Description:** The American Immigration Council obtained individual-level refugee resettlement data (October 2017 - December 2024) through a Freedom of Information Act (FOIA) request. The data includes:
- Nationality
- Gender
- Age at admission
- Highest level of education
- Native language
- English proficiency (oral and written)
- State and city of resettlement

**Access Status:** NOT DIRECTLY DOWNLOADABLE

The full dataset (six spreadsheets) is not publicly available for direct download. The Council created a visualization tool at https://data.americanimmigrationcouncil.org/en/refugee-resettlement-us/ but the underlying data requires contacting: research@immcouncil.org

**Note:** The visualization tool suppresses data for localities with fewer than 50 resettlements to protect refugee privacy.

---

## Processing Details

### Script: `process_refugee_data.py`

Processes the academic dataset (FY 2002-2011), WRAPS Excel files (FY 2012-2020), and RPC archive PDFs (FY 2021-2024 where possible) to create a unified dataset.

**Output Location:** `../../analysis/refugee_arrivals_by_state_nationality.parquet`

**Output Columns:**
| Column | Type | Description |
|--------|------|-------------|
| fiscal_year | int64 | Federal fiscal year (Oct 1 - Sep 30) |
| state | string | U.S. state name |
| nationality | string | Country of origin ("Total" for state aggregates) |
| arrivals | int64 | Number of refugee arrivals |
| data_source | string | Source of the data for that record |
| national_total | float64 | Total arrivals for nationality nationwide (null for "Total" rows) |
| state_share_of_nationality | float64 | State's share of national arrivals for that nationality |

---

## Summary Statistics (FY 2002-2020)

### National Totals
| Fiscal Year | Total Arrivals | Data Source |
|-------------|----------------|-------------|
| 2002 | 30,198 | Academic Dataset |
| 2003 | 32,174 | Academic Dataset |
| 2004 | 56,359 | Academic Dataset |
| 2005 | 51,927 | Academic Dataset |
| 2006 | 39,540 | Academic Dataset |
| 2007 | 47,725 | Academic Dataset |
| 2008 | 64,717 | Academic Dataset |
| 2009 | 79,934 | Academic Dataset |
| 2010 | 71,354 | Academic Dataset |
| 2011 | 51,457 | Academic Dataset |
| 2012 | 58,238 | WRAPS |
| 2013 | 69,926 | WRAPS |
| 2014 | 69,987 | WRAPS |
| 2015 | 69,933 | WRAPS |
| 2016 | 84,994 | WRAPS |
| 2017 | 53,716 | WRAPS |
| 2018 | 22,491 | WRAPS |
| 2019 | 30,000 | WRAPS |
| 2020 | 11,814 | WRAPS |

### North Dakota Arrivals
| Fiscal Year | Arrivals | % of National |
|-------------|----------|---------------|
| 2002 | 61 | 0.20% |
| 2003 | 117 | 0.36% |
| 2004 | 232 | 0.41% |
| 2005 | 245 | 0.47% |
| 2006 | 154 | 0.39% |
| 2007 | 222 | 0.47% |
| 2008 | 463 | 0.72% |
| 2009 | 503 | 0.63% |
| 2010 | 441 | 0.62% |
| 2011 | 402 | 0.78% |
| 2012 | 555 | 0.95% |
| 2013 | 456 | 0.65% |
| 2014 | 582 | 0.83% |
| 2015 | 497 | 0.71% |
| 2016 | 540 | 0.64% |
| 2017 | 420 | 0.78% |
| 2018 | 162 | 0.72% |
| 2019 | 127 | 0.42% |
| 2020 | 47 | 0.40% |

### Top Nationalities in North Dakota (FY 2002-2020)
| Nationality | Total Arrivals |
|-------------|----------------|
| Bhutan | 2,566 |
| Somalia | 1,117 |
| Iraq | 929 |
| Democratic Republic of the Congo | 484 |
| Liberia | 236 |
| Sudan | 222 |
| Burundi | 167 |
| Yugoslavia | 68 |
| Eritrea | 63 |
| Afghanistan | 56 |

---

## Data Quality Notes

1. **Data Source Transition:** The dataset combines two sources at FY 2012. The academic dataset (FY 2002-2011) may have slightly higher totals than official WRAPS data due to different methodologies.

2. **Historical Reconciliation:** WRAPS notes that "Historical monthly arrivals are subject to change due to reconciliation."

3. **State Coverage:** Not all 50 states receive refugees every year. Some states (like Wyoming, Montana) have limited or no resettlement activity.

4. **FY 2021-2024 Gap:** RPC archive PDFs are partially parsed. FY2024 now uses OCR extraction (partial state coverage); FY2022 remains image-only after page 1 with ND-only totals retained as placeholders until alternative sources are acquired.
   - Alternative data from DHS OHSS (https://ohss.dhs.gov/topics/immigration/refugees)
   - Request data from American Immigration Council
5. **COVID-19 Impact:** FY 2020 shows dramatically reduced arrivals (11,814 vs typical 50,000-85,000) due to pandemic travel restrictions.

6. **Post-9/11 Impact:** FY 2002 shows dramatically reduced arrivals (~30,000) due to enhanced security screening following September 11, 2001.

7. **Nationality Standardization:** Some nationality names have been standardized across data sources (e.g., "Dem. Rep. Congo" -> "Democratic Republic of the Congo", "Yugoslavia" for pre-breakup countries).

---

## Alternative Data Sources

For additional refugee data or to fill gaps:

1. **DHS OHSS Yearbook of Immigration Statistics**
   - URL: https://ohss.dhs.gov/topics/immigration/yearbook
   - Contains annual refugee flow reports with detailed statistics

2. **Migration Policy Institute**
   - URL: https://www.migrationpolicy.org/programs/data-hub/charts/us-refugee-resettlement
   - Provides historical ceilings and admissions data

3. **UNHCR Resettlement Data Finder**
   - URL: https://rsq.unhcr.org/en/
   - Global refugee resettlement statistics

4. **RefugeeResettlementData.com** (Academic Dataset)
   - URL: https://www.refugeeresettlementdata.com/data.html
   - City-level geocoded data 1975-2018

---

## Usage Notes

To regenerate the processed data:
```bash
cd /home/nigel/cohort_projections/sdc_2024_replication/data_immigration_policy/source/refugee_data
python process_refugee_data.py
```

The output parquet file is suitable for:
- Analysis of refugee distribution by state and nationality
- Calculating North Dakota's share of specific nationality groups
- Trend analysis of refugee resettlement patterns
- Integration with population projection models
- Historical analysis of refugee policy impacts (post-9/11, pre-/post-Obama administration, etc.)
