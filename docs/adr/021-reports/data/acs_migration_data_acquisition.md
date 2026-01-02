# ACS Migration Data Acquisition Report

**ADR-021: External Analysis Integration**
**Date:** 2026-01-01
**Purpose:** Assess ACS data availability for analyzing secondary migration of foreign-born populations to North Dakota

---

## Executive Summary

This report assesses the availability of American Community Survey (ACS) data for analyzing secondary migration of foreign-born populations to North Dakota. The external AI analysis suggests that ND receives foreign-born population through domestic redistribution (secondary migration) rather than solely through direct international arrival.

**Key Findings:**
1. **Current project data**: ACS foreign-born origin data (B05006) available 2009-2023, but this is population stock, not migration flow
2. **Missing data**: State-to-state migration flows filtered by foreign-born/nativity status
3. **Data available externally**: ACS PUMS with MIGSP + NATIVITY variables (through 2024)
4. **Recommendation**: Use ACS PUMS microdata for custom tabulation of foreign-born secondary migration

---

## 1. Current ACS Data in Project

### 1.1 Existing Files

| File | Description | Years | Coverage |
|------|-------------|-------|----------|
| `acs_foreign_born_by_state_origin.parquet` | B05006 - Foreign-born by country of origin, all states | 2009-2023 | 128,596 rows |
| `acs_foreign_born_nd_share.parquet` | ND's share of national foreign-born by origin | 2009-2023 | 2,473 rows |

**Location:** `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/immigration/analysis/`

### 1.2 What This Data Contains

The existing ACS data is from table **B05006** (Place of Birth for the Foreign-Born Population). This provides:
- Foreign-born population counts by state and country of origin
- Hierarchical breakdown: Region > Sub-region > Country > Detail
- ND's share of national foreign-born for each origin

**Columns:** year, state_fips, state_name, variable, region, sub_region, country, detail, level, foreign_born_pop, margin_of_error

### 1.3 What This Data DOES NOT Contain

The B05006 data is a **population stock** measure, not a **migration flow** measure:
- Does NOT indicate where foreign-born residents moved FROM within the US
- Does NOT distinguish between:
  - Direct international arrivals (arrived from abroad in past year)
  - Secondary migrants (moved from another US state in past year)
  - Long-term residents (lived in current state for years)

**This is the key gap for secondary migration analysis.**

---

## 2. External Data Availability

### 2.1 ACS PUMS (Public Use Microdata Sample)

**Best option for secondary migration analysis**

The ACS PUMS contains individual-level records with migration and nativity variables that can be combined for custom tabulations.

#### Key Variables for Secondary Migration

| Variable | Description | Use |
|----------|-------------|-----|
| `NATIVITY` | Native or foreign-born | Filter to foreign-born population |
| `MIG` | Mobility status (lived here 1 year ago) | Identify movers |
| `MIGSP` | Migration state code (residence 1 year ago) | Identify origin state |
| `ST` | Current state of residence | Identify destination (ND = 38) |
| `POBP` | Place of birth (detailed) | Country of origin |
| `YOEP` | Year of entry to US | Distinguish recent vs long-term immigrants |

#### Available Years

| Dataset | Years | Notes |
|---------|-------|-------|
| ACS 1-Year PUMS | 2005-2024 | Most current; 2024 released Dec 2024 |
| ACS 5-Year PUMS | 2009-2023 | More stable for small areas like ND |

**Source:** [Census PUMS Documentation](https://www.census.gov/programs-surveys/acs/microdata.html)

#### Access Methods

1. **Direct Download (FTP)**
   - URL: `https://www2.census.gov/programs-surveys/acs/data/pums/`
   - Format: CSV, compressed by state or national
   - Size: Several GB per year for national files

2. **IPUMS USA**
   - URL: `https://usa.ipums.org/usa/`
   - Advantage: Harmonized variables across years, easier data extraction
   - Requires (free) account registration

3. **data.census.gov Microdata Access Tool**
   - Interactive subsetting of PUMS
   - Can download custom extracts

### 2.2 Pre-Tabulated Migration Flows (ACS Flows API)

The Census Bureau provides pre-tabulated state-to-state migration flows, but with **limited nativity filtering**.

#### Available Data

| Endpoint | Years | Foreign-Born Filter? |
|----------|-------|---------------------|
| `/data/{year}/acs/flows` | 2010-2020 (5-year) | **LIMITED** - only POB characteristic in some years |

**API Base URL:** `https://api.census.gov/data/2020/acs/flows`

**Limitation:** The flows API does not have a direct NATIVITY filter. The POB (Place of Birth) characteristic is available in some years but may not provide the foreign-born vs native-born breakdown needed.

**Recommendation:** Use PUMS microdata for full control over filtering.

### 2.3 ACS Detailed Tables with Nativity x Mobility

Several ACS detailed tables cross-tabulate geographic mobility with citizenship/nativity:

| Table | Title | Available via API? |
|-------|-------|-------------------|
| B07007 | Geographic Mobility by Citizenship Status (Current Residence) | Yes |
| B07407 | Geographic Mobility by Citizenship Status (Residence 1 Year Ago) | Yes |

**API Example:**
```
https://api.census.gov/data/2023/acs/acs1?get=NAME,B07007_001E,B07007_002E,B07007_003E&for=state:38
```

**Limitation:** These tables provide aggregate counts by mobility type (same house, different house same county, different county same state, different state, abroad) crossed with citizenship status. They do NOT provide origin-state-to-destination-state flows.

**Use case:** Can quantify total foreign-born in-migration to ND from other states (aggregate), but not identify source states.

---

## 3. Analysis Strategy for Secondary Migration

### 3.1 Recommended Approach: ACS PUMS

To analyze foreign-born secondary migration to ND:

```
Methodology:
1. Download ACS 1-Year PUMS for relevant years (2019-2024)
2. Filter to:
   - ST = 38 (current residence North Dakota)
   - NATIVITY = 2 (foreign-born)
   - MIG = 1 (different house 1 year ago) AND MIGSP != 38 (not from ND)
3. Exclude direct international arrivals:
   - MIGSP != 999 (not abroad 1 year ago)
4. Tabulate by MIGSP (origin state) to get secondary migration sources
5. Apply person weights (PWGTP) for population estimates
6. Calculate replicate-weight standard errors for margin of error
```

### 3.2 Alternative: Use Detailed Tables for Aggregate Validation

Use table B07407 to get aggregate counts for validation:

| Mobility Type (for Foreign-born living in ND) | Variable |
|----------------------------------------------|----------|
| Total foreign-born | B07407 (filter by citizenship status) |
| Same house | Check for non-movers |
| Different state | Secondary migrants |
| Abroad | Direct international arrivals |

### 3.3 Data Quality Considerations for ND

**Small Population Challenge:**
- ND has ~23,000 foreign-born population (2023)
- ACS 1-year samples ~1% of population
- Expected sample size: ~230 foreign-born respondents
- State-specific migration flows will have HIGH variance

**Recommendations:**
1. Use 5-year ACS for more stable estimates
2. Aggregate source states into regions (Midwest, West, etc.)
3. Pool multiple years for trend analysis
4. Report confidence intervals, not just point estimates

---

## 4. Specific Variables Needed

### 4.1 PUMS Variables for Secondary Migration Analysis

| Variable | Type | Values of Interest |
|----------|------|-------------------|
| NATIVITY | Person | 2 = Foreign-born |
| MIG | Person | 1 = Yes, moved from different house |
| MIGSP | Person | 01-56 = US states/territories; 999 = Abroad |
| ST | Person/HH | 38 = North Dakota |
| PWGTP | Weight | Person weight for population estimation |
| PWGTP1-80 | Weights | Replicate weights for variance estimation |
| POBP | Person | Country of birth (for origin analysis) |
| YOEP | Person | Year of entry (for recent vs established) |

### 4.2 Detailed Table Variables

| Table | Variables | Use |
|-------|-----------|-----|
| B07007 | B07007_001E through B07007_XXX | Mobility by citizenship, current residence |
| B07407 | B07407_001E through B07407_XXX | Mobility by citizenship, prior residence |
| B05006 | B05006_001E through B05006_XXX | Foreign-born by origin (already in project) |

---

## 5. Data Format and Access Summary

### 5.1 Recommended: PUMS via IPUMS

| Attribute | Value |
|-----------|-------|
| Source | IPUMS USA (usa.ipums.org) |
| Format | CSV or fixed-width, with codebook |
| Years needed | 2019-2024 (1-year) or 2019-2023 (5-year) |
| Geographic filter | ST=38 (ND) or download national for comparison |
| Variables | NATIVITY, MIG, MIGSP, ST, PWGTP, POBP, YOEP |
| Effort | Low-Medium (requires free registration, custom extract) |

### 5.2 Alternative: Direct Census FTP

| Attribute | Value |
|-----------|-------|
| Source | census.gov FTP |
| URL | https://www2.census.gov/programs-surveys/acs/data/pums/ |
| Format | CSV.gz |
| Size | ~500MB-1GB per national file |
| Effort | Medium (larger download, need data dictionary) |

### 5.3 Detailed Tables via Census API

| Attribute | Value |
|-----------|-------|
| Source | Census Data API |
| Base URL | https://api.census.gov/data/{year}/acs/acs1 |
| Tables | B07007, B07407 |
| Geography | state:38 |
| Years | 2010-2023 (1-year), 2009-2023 (5-year) |
| Effort | Low (can use existing census_api.py module) |

---

## 6. Recommendations

### 6.1 Short-term (Aggregate Validation)

1. **Use Census API for B07007/B07407** to get aggregate mobility by citizenship for ND
2. Compare "moved from different state" counts for foreign-born vs native-born
3. This provides a quick validation of the secondary migration hypothesis

### 6.2 Medium-term (Full Analysis)

1. **Acquire ACS PUMS via IPUMS** for 2019-2024
2. Create custom tabulation of foreign-born secondary migration to ND
3. Identify source states/regions for ND's foreign-born in-migrants
4. Compare direct international arrivals vs secondary domestic migration

### 6.3 Integration with Existing Infrastructure

The project already has:
- `CensusDataFetcher` class in `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/data/fetch/census_api.py`
- ACS processing pipeline in `process_b05006.py`

**Extension needed:**
- Add PUMS download/processing capability
- Add B07007/B07407 table fetching to Census API module
- Create secondary migration analysis script

---

## 7. Data Acquisition Checklist

| Item | Status | Notes |
|------|--------|-------|
| ACS B05006 (foreign-born by origin) | COMPLETE | 2009-2023, in project |
| ACS B07007/B07407 (mobility by citizenship) | NOT STARTED | Available via API, aggregate only |
| ACS PUMS (MIGSP + NATIVITY) | NOT STARTED | Requires IPUMS registration or FTP download |
| State-to-state flows by nativity | BLOCKED | Not available pre-tabulated; requires PUMS |

---

## Sources

- [Census ACS PUMS Documentation](https://www.census.gov/programs-surveys/acs/microdata.html)
- [Census ACS Migration Flows](https://www.census.gov/data/developers/data-sets/acs-migration-flows.html)
- [Census Foreign-Born Tables by Subject](https://www.census.gov/topics/population/foreign-born/guidance/acs-guidance/acs-by-subject.html)
- [Census State-to-State Migration Flows](https://www.census.gov/topics/population/migration/guidance/state-to-state-migration-flows.html)
- [Census Reporter B07007 Table](https://censusreporter.org/tables/B07007/)
- [IPUMS USA](https://usa.ipums.org/usa/)
- [2023 State-to-State Migration Flows Release](https://www.census.gov/newsroom/press-releases/2024/state-to-state-migration-flows.html)

---

*Report generated: 2026-01-01*
*ADR-021: External Analysis Integration - Data Acquisition Phase*
