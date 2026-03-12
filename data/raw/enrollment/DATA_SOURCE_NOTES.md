# Enrollment Data Source Notes

## NDUS Annual Enrollment Reports

### Source
North Dakota University System (NDUS), Office of Institutional Research.
Annual Enrollment Reports, academic years 2001-2002 through 2024-2025.

### Files
- `ndus_annual_enrollment_reports_2001-2025.zip` — Original archive of 22 PDF reports
- Individual PDF files: `{YYYY-YYYY} Annual Enrollment Report.pdf` (22 files)
- `ndus_institution_reference.csv` — Institution-to-county FIPS mapping with 2024-2025 enrollment

### Coverage
- **Years**: AY 2001-2002 through AY 2024-2025 (22 reports; gaps at 2005-2006, 2011-2012, 2012-2013)
- **Institutions**: 11 NDUS campuses across 3 tiers (Community College, Regional University, Research University)
- **Metrics**: Degree credit headcount, non-degree credit, non-credit, by delivery mode (on-campus face-to-face, off-campus, distance education)

### Key Tables
- **Table 1 (all years)**: Annual Enrollment Summary — total degree credit, non-degree credit, non-credit headcount per institution
- **Table 2 (2018+ format)**: Degree Credit Headcount by All Delivery Modes — on-campus face-to-face, off-campus face-to-face, hyflex/hybrid, distance ed
- **Older format**: "All Delivery Methods" table — on-campus, off-campus, IVN, internet columns

### Notes on Delivery Mode Changes
- **Pre-2019**: Distance education categories were "IVN" (Interactive Video Network) and "Internet"
- **2019+**: SBHE Policy 404.1 realigned NDUS delivery mode categories with federal IPEDS definitions
- **2020-2021**: COVID-19 shifted most courses to Hyflex/remote delivery; on-campus counts are not representative of typical in-person presence
- **On-campus face-to-face** is the most relevant metric for population projection purposes, as it approximates students physically present in the county

### Institution-to-County Mapping
See `ndus_institution_reference.csv` for the complete mapping including FIPS codes.

### Private Institutions (Non-NDUS)

Two private institutions are included in the institution reference file with estimated on-campus enrollment:

- **University of Mary** (Bismarck, Burleigh County) — ~4,003 total enrollment (2025), estimated ~2,000 on-campus
- **University of Jamestown** (Jamestown, Stutsman County) — ~1,376 total enrollment (2024-2025), estimated ~900 on-campus

On-campus figures for private institutions are estimates based on full-time enrollment data, as private institutions do not report in the NDUS delivery mode format.

### Use in Projection Pipeline
This data supports the college county identification for migration rate smoothing (ADR-049, ADR-061).
The enrollment-to-population ratio determines which counties have large enough student populations
to distort IRS-based residual migration rates. Counties with high ratios should be included in the
`college_counties` config list for college-age smoothing.

### License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
