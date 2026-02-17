# ADR-038: Multi-Workbook Export Format for Population Projections

## Status
Accepted

## Date
2026-02-17

## Last Reviewed
2026-02-17

## Scope
Export format and delivery structure for population projection results intended for the North Dakota Department of Health & Human Services and other stakeholders.

## Context

Comparison with the SDC 2024 County Population Projections workbook (`data/raw/nd_sdc_2024_projections/County_Population_Projections_2023.xlsx`) revealed significant gaps in our provisional export format:

| Attribute | SDC 2024 Workbook | Our Provisional Workbook |
|---|---|---|
| Sheets | 55 (1 county list + 1 blank + 53 county sheets) | 9 (summary only) |
| County detail | 18 age groups × Male/Female/Both × 9 time points (324 cells/county) | Total population at 5 key years only |
| Sex breakdown | Full per-county breakdown | Not shown |
| Regional aggregation | Separate workbook with 8 planning regions | Not included |
| Historical data | 2010, 2015, 2020 alongside projections | Projections only (2025+) |

The Department of Health & Human Services (DHHS) requires county-level age-sex detail for:
- Service planning (child welfare, WIC, senior services, workforce)
- Caseload projections by age eligibility
- Dependency ratio calculations
- Population pyramid visualizations

A single workbook containing all scenarios and all counties (~193 sheets) would be unwieldy on state-issued hardware and difficult to navigate.

## Decision

Adopt a multi-workbook export format:

### 1. Summary Workbook (~10 sheets)
Cross-scenario comparison for leadership review. Retains current structure with state totals, scenario comparison, county rankings, and age overview.

**File**: `nd_population_projections_provisional_{datestamp}.xlsx`

### 2. Detail Workbooks — One Per Scenario (~63 sheets each)
Full age-sex distributions for state, regions, and all counties.

**Files**:
- `nd_projections_baseline_detail_{datestamp}.xlsx`
- `nd_projections_restricted_growth_detail_{datestamp}.xlsx`
- `nd_projections_high_growth_detail_{datestamp}.xlsx`

### Detail Workbook Structure

Each per-scenario detail workbook contains:

| Sheet | Content |
|---|---|
| Table of Contents | Hyperlinked navigation with population summary per geography |
| North Dakota | State-level age-sex detail |
| Reg 1 - Williston through Reg 8 - Dickinson | Regional aggregations (8 sheets) |
| Adams through Williams | All 53 counties, alphabetical (53 sheets) |

### Per-Geography Sheet Layout

Each geography sheet includes three sections (Male, Female, Both Sexes), each containing:
- 18 five-year age groups (0-4 through 85+) at 5-year intervals (2025, 2030, 2035, 2040, 2045)
- Row totals per sex
- Change and % Change columns (2045 vs 2025)

Below the population tables:
- Dependency ratios: Youth (0-14 / 15-64), Aged (65+ / 15-64), Total
- Hyperlink back to Table of Contents

### Planning Region Definitions

Eight regions following SDC/Human Service Zone conventions:

| Region | Name | Counties |
|---|---|---|
| 1 | Williston | Divide, McKenzie, Williams |
| 2 | Minot | Bottineau, Burke, McHenry, Mountrail, Pierce, Renville, Ward |
| 3 | Devils Lake | Benson, Cavalier, Eddy, Ramsey, Rolette, Towner |
| 4 | Grand Forks | Grand Forks, Nelson, Pembina, Walsh |
| 5 | Fargo | Cass, Ransom, Richland, Sargent, Steele, Traill |
| 6 | Jamestown | Barnes, Dickey, Foster, Griggs, LaMoure, Logan, McIntosh, Stutsman, Wells |
| 7 | Bismarck | Burleigh, Emmons, Grant, Kidder, McLean, Mercer, Morton, Oliver, Sheridan, Sioux |
| 8 | Dickinson | Adams, Billings, Bowman, Dunn, Golden Valley, Hettinger, Slope, Stark |

East Four (Regions 3-6), West Four (Regions 1, 2, 7, 8) groupings available for future use.

### Data Resolution

The workbooks display 5-year age groups at 5-year intervals for readability, matching the SDC format. Underlying parquet files retain full resolution:
- Single-year ages (0-90)
- 6 race/ethnicity categories
- Annual time steps (2025-2045)

Advanced users needing race breakdowns or annual granularity should use the parquet files directly.

## Alternatives Considered

### 1. Single monolithic workbook (~193 sheets)
All scenarios and counties in one file. Rejected: tab navigation breaks down above ~50 sheets; slow to open on state-issued laptops; difficult to share.

### 2. CSV/flat file only
Just provide CSV exports. Rejected: DHHS analysts expect Excel workbooks matching SDC format; no built-in navigation or formatting.

### 3. One sheet per county with all scenarios side-by-side
Compact but too wide (18 age groups × 5 years × 3 scenarios = 270+ columns per sex). Doesn't match established SDC format.

### 4. One sheet per county containing all years (annual)
Would show all 21 years instead of 5 key years. Rejected: 21 columns per sex section is too wide for printing; 5-year intervals match SDC convention and are sufficient for planning purposes.

## Consequences

### Positive
- Matches and exceeds SDC 2024 format — familiar to state analysts
- Adds regional aggregations and dependency ratios not in SDC workbook
- Manageable file sizes (~200-400 KB each, opens quickly)
- Per-scenario workbooks are self-contained for distribution
- Hyperlink navigation for easy sheet access
- Race/ethnicity detail available in underlying parquet data

### Negative
- 4 files instead of 1 (mitigated by consistent naming convention)
- No historical data (2010/2015/2020) in workbooks — available in separate reference data

## Cross-References
- [ADR-037](037-cbo-grounded-scenario-methodology.md): CBO-Grounded Scenario Methodology (defines the 3 scenarios)
- SDC 2024 Reference: `data/raw/nd_sdc_2024_projections/County_Population_Projections_2023.xlsx`
- Detail builder: `scripts/exports/build_detail_workbooks.py`
- Summary builder: `scripts/exports/build_provisional_workbook.py`
- Projection data: `data/projections/{scenario}/county/*.parquet`
