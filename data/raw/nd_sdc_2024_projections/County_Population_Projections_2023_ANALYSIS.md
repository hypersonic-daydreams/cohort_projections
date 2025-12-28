# County Population Projections 2023 Excel File Analysis

**File:** `County_Population_Projections_2023.xlsx`
**Source:** North Dakota State Data Center
**Analysis Date:** 2025-12-28

---

## File Structure

### Sheets Overview

| Sheet | Description |
|-------|-------------|
| Sheet1 | Empty |
| County List | Reference table with region assignments (8 regions, 53 counties) |
| CO1 - CO53 | Individual county projection data |

### County Sheet Structure (CO1-CO53)

Each county sheet contains 70 rows and 40 columns with:

**Rows:**
- Rows 0-1: County name header
- Rows 2-23: Male population by age group (18 groups: 0-4 through 85+)
- Row 23: Male total
- Rows 25-46: Female population by age group
- Row 46: Female total
- Rows 48-69: Both sexes population by age group
- Row 69: Both sexes total

**Columns:**
- Col 1: Age group identifier (1-18)
- Cols 2-10: Population values for years 2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050
- Cols 13-39: Population pyramid visualization data (negative males, positive females, scaled)

### Age Groups

18 five-year age groups:
1. 0-4
2. 5-9
3. 10-14
4. 15-19
5. 20-24
6. 25-29
7. 30-34
8. 35-39
9. 40-44
10. 45-49
11. 50-54
12. 55-59
13. 60-64
14. 65-69
15. 70-74
16. 75-79
17. 80-84
18. 85+

---

## Formulas and Data Sources

The spreadsheet contains formulas that reference external workbooks:

```
=SUM('[1]Census 2010'!$E3)      # Census 2010 data
=SUM([1]Senthetic_2015_2!$E3)   # Synthetic 2015 estimates
=SUM('[1]Census 2020'!$E3)      # Census 2020 data
=SUM('[1]2025 Pro'!$E3)         # 2025 projection
=SUM('[1]2030 Pro'!$E3)         # 2030 projection
=SUM('[1]2035 Pro'!$E3)         # 2035 projection
=SUM('[1]2040 Pro'!$E3)         # 2040 projection
=SUM('[1]2045 Pro'!$E3)         # 2045 projection
=SUM('[1]2050 Pro'!$E3)         # 2050 projection
```

**Key Observations:**
- Historical data (2010, 2015, 2020) are whole numbers from Census/PEP
- Projected data (2025-2050) have many decimal places (computed values)
- The actual projection calculations are in external linked workbooks
- This file is a summary/compilation sheet

---

## Regional Groupings

The counties are organized into 8 planning regions:

| Region | Counties |
|--------|----------|
| 1 | Divide, McKenzie, Williams |
| 2 | Bottineau, Burke, McHenry, Mountrail, Pierce, Renville, Ward |
| 3 | Benson, Cavalier, Eddy, Ramsey, Rolette, Towner |
| 4 | Grand Forks, Nelson, Pembina, Walsh |
| 5 | Cass, Ransom, Richland, Sargent, Steele, Traill |
| 6 | Barnes, Dickey, Foster, Griggs, LaMoure, Logan, McIntosh, Stutsman, Wells |
| 7 | Burleigh, Emmons, Grant, Kidder, McLean, Mercer, Morton, Oliver, Sheridan, Sioux |
| 8 | Adams, Billings, Bowman, Dunn, Golden Valley, Hettinger, Slope, Stark |

---

## State Population Totals

| Year | Population | Change from 2020 |
|------|------------|------------------|
| 2010 | 672,591 | -13.7% |
| 2015 | 749,517 | -3.8% |
| 2020 | 779,094 | -- |
| 2025 | 796,989 | +2.3% |
| 2030 | 831,543 | +6.7% |
| 2035 | 865,397 | +11.1% |
| 2040 | 890,424 | +14.3% |
| 2045 | 925,101 | +18.7% |
| 2050 | 957,194 | +22.9% |

These totals exactly match the published SDC 2024 projections, confirming this is the source data.

---

## County Growth Patterns (2020-2050)

### Fastest Growing Counties

| County | 2020 Pop | 2050 Pop | Growth % |
|--------|----------|----------|----------|
| Stark | 33,646 | 52,510 | +56.1% |
| Cass | 184,525 | 272,878 | +47.9% |
| McKenzie | 14,704 | 21,633 | +47.1% |
| Dunn | 4,095 | 5,976 | +45.9% |
| Williams | 40,950 | 56,047 | +36.9% |
| Divide | 2,195 | 2,917 | +32.9% |
| Burke | 2,201 | 2,900 | +31.7% |
| Burleigh | 98,458 | 128,663 | +30.7% |
| Billings | 945 | 1,229 | +30.1% |
| Morton | 33,291 | 41,359 | +24.2% |

### Fastest Declining Counties

| County | 2020 Pop | 2050 Pop | Growth % |
|--------|----------|----------|----------|
| Towner | 2,162 | 1,805 | -16.5% |
| Emmons | 3,301 | 2,751 | -16.7% |
| Steele | 1,798 | 1,488 | -17.3% |
| Barnes | 2,200 | 1,805 | -17.9% |
| Grant | 2,301 | 1,884 | -18.1% |
| McIntosh | 2,530 | 2,004 | -20.8% |
| Sheridan | 1,265 | 985 | -22.1% |
| Wells | 3,982 | 3,093 | -22.3% |
| Nelson | 3,015 | 2,229 | -26.1% |
| Pembina | 6,844 | 5,030 | -26.5% |

### Urban vs Rural Summary

| Type | Counties | Avg Growth |
|------|----------|------------|
| Urban | 7 (Cass, Burleigh, Grand Forks, Ward, Stark, Williams, Morton) | +32.9% |
| Rural | 46 | -3.6% |

---

## Usefulness for Our Project

### Valuable Data

1. **County-level age-sex distributions** - Complete age-sex breakdowns by county for comparison
2. **Regional groupings** - 8-region classification for regional aggregation analysis
3. **Historical baselines** - 2010, 2015, 2020 populations by age-sex-county
4. **Projection assumptions validated** - Our state totals can be compared at county level

### Methodology Insights

1. **18 age groups** - SDC uses 5-year groups ending at 85+; our project uses single-year ages to 90+
2. **External projection models** - The actual cohort-component calculations are in linked workbooks we don't have
3. **Synthetic 2015** - SDC created synthetic 2015 estimates (not direct Census data)
4. **Population pyramid format** - Data formatted for visualization (scaled, signed by sex)

### Comparison Opportunities

1. **Age structure comparison** - Our 2025-2045 projections can be grouped to 5-year age groups for direct comparison
2. **County-level validation** - Compare our county totals to SDC by county
3. **Growth pattern analysis** - Are the same counties growing/declining in both projections?
4. **Demographic composition** - SDC has no race data; our race-specific projections are unique

---

## Data Quality Notes

- All 53 counties present with complete data
- Formula references indicate structured projection methodology
- 2010/2020 Census years have integer values (actual counts)
- 2015 labeled as "Synthetic" (estimated, not Census)
- Projection years have high decimal precision (computed)
- Population pyramid columns are derived (negative males = left side)

---

## Recommendations

1. **Extract for comparison analysis** - Export county totals by year for systematic comparison
2. **Age group aggregation** - Group our projections to 5-year intervals for SDC comparison
3. **County growth correlation** - Analyze whether county-level divergence follows urban/rural patterns
4. **Document missing source** - Note that full projection model (linked workbooks) is not available
