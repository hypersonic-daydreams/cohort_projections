# Public Download Specification

Internal repo record. Do not include this file in the marketing handoff packet.
It defines the public Excel/CSV production shape for the data team after
CF-001 is resolved and final numbers are produced.

Status: draft specification for next-week final-number production.

## Decision

The public release should provide one baseline-only public path:

1. One consolidated Excel workbook.
2. One consolidated CSV.

Both files exclude city/place projections. Public geography levels are state,
region, and county. The public scenario label is `baseline` with the display
name Baseline (CBO-Adjusted); `recent_trend_continuation`,
`restricted_growth`, and `high_growth` remain internal or legacy sensitivities
and are not part of the public package.

## Excel Workbook

Working filename:

`nd_population_projections_public_2026.xlsx`

Required sheets:

| Sheet | Purpose | Notes |
|-------|---------|-------|
| `README` | Plain-language file description, release date, contacts, methodology pointer, caveats | Include ADR-042 caveats and baseline wording. |
| `State Key Years` | Statewide key-year values for the public baseline path | Use 2025, 2030, 2035, 2040, 2045, 2050, 2055. |
| `State Annual` | Annual state totals and broad demographics | One row per baseline-year. |
| `Region Annual` | Annual region totals | One row per baseline-region-year. |
| `County Annual` | Annual county totals | One row per baseline-county-year. |
| `County Key Years` | Compact county table for public users | Key years only, baseline-only. |
| `State Age Groups` | Broad state age-group composition | Include under 18, 18-64, 65+, and 85+. |
| `Data Dictionary` | Column definitions and baseline description | Match CSV schema. |

## CSV File

Working filename:

`nd_population_projections_public_2026.csv`

Shape:

- Tidy format.
- One row per baseline, geography, and year.
- Scenario: `baseline` only.
- Geographies: 1 state, 8 regions, 53 counties.
- Years: 2025-2055 inclusive.
- Expected row count after final production: `1,922`.

Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `scenario` | string | Scenario key: `baseline` only; public label is Baseline (CBO-Adjusted) |
| `geography_level` | string | `state`, `region`, or `county` |
| `geography_fips` | string | State or county FIPS; region code for region rows |
| `geography_name` | string | Display name |
| `region_id` | string | Region ID for region/county rows; blank for state |
| `region_name` | string | Region display name for region/county rows; blank for state |
| `year` | integer | Projection year |
| `total_population` | number | Projected total population |
| `population_under_18` | number | Projected population under age 18 |
| `population_working_age_18_64` | number | Projected population age 18-64 |
| `population_65_plus` | number | Projected population age 65 and older |
| `population_85_plus` | number | Projected population age 85 and older |
| `male_population` | number | Projected male population |
| `female_population` | number | Projected female population |
| `sex_ratio` | number | Males per 100 females |
| `youth_dependency_ratio` | number | Youth dependency ratio |
| `elderly_dependency_ratio` | number | Elderly dependency ratio |
| `total_dependency_ratio` | number | Total dependency ratio |

## Region Definitions

Use the ADR-038 region definitions:

| Region ID | Region Name |
|-----------|-------------|
| `R1` | Williston |
| `R2` | Minot |
| `R3` | Devils Lake |
| `R4` | Grand Forks |
| `R5` | Fargo |
| `R6` | Jamestown |
| `R7` | Bismarck |
| `R8` | Dickinson |

## Packaging Rules

- Produce final files from a clean staging directory.
- Do not package city/place rows.
- Do not package `recent_trend_continuation`, `restricted_growth`, or
  `high_growth` rows or tabs.
- Do not ship existing March 1 ZIP packages as public downloads.
- Check that no stale `2025_2045` files are used as public input.
- Keep exact values in downloads; PDF values can be rounded for readability.

## Final-Data Dependencies

- CF-001 benchmark decision must be resolved.
- ADR-065 baseline adjustments must be reflected in the public outputs.
- If CF-001 is promoted, production projections must be rerun before final public files are built.
- Public files should record the production run date, method/config identifiers, and source export report.
