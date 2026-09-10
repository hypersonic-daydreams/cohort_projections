# 2026-02-28 Place-to-County Mapping Strategy Note (PP3-S03)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Reviewer** | Codex (GPT-5) |
| **Scope** | PP3-S03 mapping strategy and boundary/vintage handling rules for place share-trending |
| **Status** | Complete (rules defined) |
| **Related ADR** | ADR-033 |

---

## 1. Problem Statement

Place totals files (`sub-est00int`, `sub-est2020int`, `sub-est2024`) do not provide county assignment for ND places (`COUNTY=000`), but ADR-033 requires city shares relative to county totals. A deterministic and versioned place->county mapping policy is required before model implementation.

## 2. Authoritative Mapping Standard

Phase 1 mapping will use **Census 2020-vintage place/county geography** as the canonical reference frame for the first release.

Required artifact:
- `data/processed/geographic/place_county_crosswalk_2020.csv`

Required columns:
- `state_fips` (2-digit)
- `place_fips` (7-digit GEOID: state+place)
- `county_fips` (5-digit GEOID: state+county)
- `assignment_type` (`single_county`, `multi_county_primary`)
- `area_share` (0-1 for assigned county, for auditability)
- `source_vintage` (`2020`)
- `source_method` (e.g., `census_relationship_file` or `tiger_overlay`)

## 3. Assignment Rules

1. If a place intersects one county only, assign that county (`single_county`).
2. If a place intersects multiple counties, assign the county with the largest 2020 area share as primary (`multi_county_primary`).
3. Keep secondary-county overlaps in a supplemental audit table for QA (`place_county_crosswalk_2020_multicounty_detail.csv`), but Phase 1 share-trending uses the primary assignment only.
4. Mapping keys must be unique on `place_fips` in the primary crosswalk.

## 4. Boundary/Vintage Handling Rules

1. **Freeze projection geography to 2020 vintage** for Phase 1 backtests and forward runs.
2. Use historical place totals for trend estimation, but map all records to the fixed 2020 place universe.
3. Mark dissolved/inactive places as `historical_only` and exclude from projection output universe.
   - Current known cases from readiness check: Bantry city (`04740`), Churchs Ferry city (`14140`).
4. Maintain a small alias table for name-only changes so time-series joins remain stable on FIPS.
   - Current known case: `12060` renamed from `Canton City city` to `Canton City (Hensel) city`.

## 5. QA Rules for Mapping Artifact

Before PP3-S04 modeling begins, enforce:

1. Every active place in the projection universe has exactly one `county_fips` in the primary crosswalk.
2. No null/invalid FIPS in `place_fips` or `county_fips`.
3. `area_share` in `(0, 1]` for all rows.
4. Multi-county places are explicitly flagged in `assignment_type` and present in the supplemental detail table.
5. Spot-check county assignments for major places (Fargo, Bismarck, Grand Forks, Minot, West Fargo, Williston).

## 6. Implementation Dependency

PP3-S03 is scoped/defined, but implementation of the mapping file itself remains a pre-modeling dependency for PP3-S04. No projection methodology change is executed in this step.

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
