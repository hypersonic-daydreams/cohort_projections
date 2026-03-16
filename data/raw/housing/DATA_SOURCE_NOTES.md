# Housing Data Source Notes

## Overview

This directory contains housing unit and household size data for North Dakota places, used by the housing-unit method (HU method) as a complementary short-term cross-check for share-trending place projections.

**Key ADR**: ADR-060 (Housing-Unit Method for Place Projections)

---

## ACS Place-Level Housing Units -- Added 2026-03-01

| File | Description | Rows | Download Date |
|------|-------------|------|---------------|
| `nd_place_housing_units.csv` | ACS 5-year estimate housing units and average household size for all ND places, 2009-2023 vintages | ~3,700 | 2026-03-01 |

**Source:** U.S. Census Bureau, American Community Survey (ACS) 5-Year Estimates
**API:** `https://api.census.gov/data/{year}/acs/acs5`
**Tables:** B25001 (Total Housing Units), B25010 (Average Household Size)
**Geography:** Place level, State FIPS 38 (North Dakota)

**Content:** Place-level housing unit counts and average household size (persons per household) for all Census-designated places in North Dakota. Data spans 15 ACS 5-year vintages: 2009 through 2023, representing the 2005-2009 through 2019-2023 estimate periods. One row per place per vintage year.

**Column definitions:**

| Column | Type | Description |
|--------|------|-------------|
| place_fips | string | 7-digit place FIPS code (state + place, e.g., 3800100 = Abercrombie) |
| place_name | string | Place name from Census API (e.g., "Abercrombie city, North Dakota") |
| year | int | ACS end-year vintage (e.g., 2023 = 2019-2023 estimates) |
| housing_units | float | Total housing units (Table B25001_001E); null if unavailable |
| avg_hh_size | float | Average household size (Table B25010_001E); null if unavailable |

**Fetch script:** `scripts/data/fetch_census_housing_data.py`

**Notes:**
- Census API sentinel values (e.g., -666666666) are converted to null
- The script retries up to 3 times per vintage with a 5-second delay on failure
- Some smaller places may have null values in earlier vintages where ACS estimates were suppressed

**Usage:** Consumed by `cohort_projections/data/process/place_housing_unit_projection.py` for the housing-unit method, which projects population as Housing Units x Persons Per Household (HU x PPH).

---

## References

1. Census Bureau ACS 5-Year API: `https://www.census.gov/data/developers/data-sets/acs-5year.html`
2. Table B25001 (Housing Units): `https://data.census.gov/table/ACSDT5Y2023.B25001`
3. Table B25010 (Average Household Size): `https://data.census.gov/table/ACSDT5Y2023.B25010`
4. ADR-060: Housing-Unit Method for Place Projections

---

## Related Documentation

- Fetch script: `scripts/data/fetch_census_housing_data.py`
- HU projection module: `cohort_projections/data/process/place_housing_unit_projection.py`
- Config: `config/projection_config.yaml` (housing_unit section)
