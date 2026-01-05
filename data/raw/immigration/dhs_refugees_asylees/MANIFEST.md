# DHS Refugees and Asylees - Data Manifest

**Download Date:** 2026-01-04 (yearbook tables, including FY2012) | 2023-12-27 to 2024-11-08 (flow reports)
**Source:** U.S. Department of Homeland Security, Office of Homeland Security Statistics (OHSS)

## Data Sources

### Yearbook Refugees and Asylees Tables (ZIP)

Yearbook table ZIPs from the DHS Yearbook of Immigration Statistics, FY 2000-2006.

| File | Fiscal Year | Notes |
|------|-------------|-------|
| `RFA2000.zip` | 2000 | Refugees and Asylees tables |
| `RFA2001.zip` | 2001 | Refugees and Asylees tables |
| `RFA2002.zip` | 2002 | Refugees and Asylees tables |
| `RFA2003.zip` | 2003 | Refugees and Asylees tables |
| `RFA%202004.zip` | 2004 | Refugees and Asylees tables (URL-encoded filename) |
| `RFA2005_0.zip` | 2005 | Refugees and Asylees tables (parted file as downloaded) |
| `RFA2006_0.zip` | 2006 | Refugees and Asylees tables (parted file as downloaded) |
| `RFA%202012_0.zip` | 2012 | Refugees and Asylees tables (URL-encoded filename) |

### Refugee Annual Flow Reports (PDF)

| File | Fiscal Year | Source URL |
|------|-------------|------------|
| `refugee_flow_report_2023.pdf` | FY 2023 | https://ohss.dhs.gov/sites/default/files/2024-11/2024_1108_ohss_refugee_annual_flow_report_2023.pdf |
| `refugee_flow_report_2022.pdf` | FY 2022 | https://ohss.dhs.gov/sites/default/files/2024-03/2023_0818_plcy_refugees_and_asylees_fy2022_v2_0.pdf |
| `refugee_flow_report_2021.pdf` | FY 2021 | https://ohss.dhs.gov/sites/default/files/2023-12/2022_0920_plcy_refugees_and_asylees_fy2021_v2.pdf |
| `refugee_flow_report_2020.pdf` | FY 2020 | https://ohss.dhs.gov/sites/default/files/2023-12/2022_0308_plcy_refugee_and_asylee_fy2020v2.pdf |
| `refugee_flow_report_2019.pdf` | FY 2019 | https://ohss.dhs.gov/sites/default/files/2023-12/refugee_and_asylee_2019.pdf |

## Notes

- These files are raw DHS/OHSS publications stored for provenance and possible future extraction.
- No processing pipeline is currently wired for these tables in v0.8.6.

## Extracted Data (from Flow Reports)

**Output file:** `data/processed/immigration/analysis/dhs_refugee_admissions.parquet`

Data extracted from PDF tables (FY 2019-2023 flow reports):

1. **Table 5: Refugee Arrivals by State of Residence (FY 2021-2023)**
   - Top 10 states by refugee arrivals
   - Total and "Other" category for remaining states

2. **Table 3: Refugee Arrivals by Country of Nationality (FY 2021-2023)**
   - Top 10 countries of nationality
   - "All other countries" category

3. **Table 1: Proposed and Actual Refugee Admissions by Region (FY 2021-2023)**
   - Ceilings and actual admissions by world region

## Data Notes

- PDF flow reports publish top-10 state tables; smaller states are aggregated into "Other".
- State-level figures reflect initial resettlement location (secondary migration not captured).
- Tables are rounded to the nearest 10 for disclosure control.
