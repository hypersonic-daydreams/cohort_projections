---
**ARCHIVED:** 2025-12-31
**Reason:** Session-specific data findings; implementation complete
**Original Location:** /BIGQUERY_DATA_SUMMARY.md
**Superseded By:** DEVELOPMENT_TRACKER.md (for ongoing status)
---

# BigQuery Data Availability Summary

## ✅ BigQuery Connection: SUCCESS

- **Project**: `antigravity-sandbox`
- **Credentials**: Configured and working
- **Access**: Can query Census and SDOH public datasets

---

## 📊 Available Data in BigQuery

### 1. Census Bureau USA (`bigquery-public-data.census_bureau_usa`)
**Status**: ✓ Accessible

**Tables Found**:
- `population_by_zip_2010`
  - Columns: `geo_id`, `zipcode`, `population`, `minimum_age`, `maximum_age`, `gender`
  - **Use for**: Base population by ZIP, age groups, gender
  - **Limitation**: Not by race/ethnicity, age groups are broad

### 2. Census Bureau ACS (`bigquery-public-data.census_bureau_acs`)
**Status**: ✓ Accessible

**Potential tables** (need to explore column names):
- County-level demographic data
- Blockgroup-level data
- **Use for**: Detailed demographics, migration/mobility data

### 3. SDOH CDC WONDER Natality (`bigquery-public-data.sdoh_cdc_wonder_natality`)
**Status**: ✓ Accessible

**Table**: `county_natality`
- Columns: `Year`, `County_of_Residence`, `County_of_Residence_FIPS`, `Births`, `Ave_Age_of_Mother`, etc.
- **Use for**: Birth counts by county and year
- **Limitation**: NOT age-specific fertility rates by race

---

## ❌ What BigQuery DOES NOT Provide

### Critical Data Still Needed:

1. **Age-Specific Fertility Rates (ASFR)**
   - Need: Rates by single-year age (15-49), race/ethnicity
   - BigQuery has: Total births by county, average mother age
   - **Must obtain from**: SEER or NVSS downloads

2. **Life Tables / Survival Rates**
   - Need: Survival probabilities by age, sex, race
   - BigQuery has: None
   - **Must obtain from**: SEER, CDC, or SSA downloads

3. **Age-Specific Migration Rates**
   - Need: Net migration by age, sex, race cohorts
   - BigQuery has: Aggregate migration (possibly in ACS)
   - **Must obtain from**: IRS county flows + allocation algorithms

---

## 📥 Data Sources Decision

### Recommendation:

**Option 1: SEER + Direct Downloads (Recommended)**
- ✅ Most accurate for demographic projections
- ✅ Standard methodology used by state agencies
- ✅ Follows project documentation
- ❌ Requires manual download of data files

### Option 2: Census API (Already Implemented)**
- ✅ `census_api.py` already fetches population data
- ✅ Automated data refresh
- ❌ Doesn't provide fertility/mortality rates

### Option 3: Mixed Approach**
- Use BigQuery for exploratory analysis
- Use Census API for base population
- Use SEER downloads for demographic rates

---

## 🎯 Next Steps

### Immediate: Implement Data Processors

Since demographic rates aren't available in BigQuery, proceed with implementing processors for downloaded data files:

### Step 1: Implement `fertility_rates.py`
**Input**: Downloaded SEER age-specific fertility rate files
**Output**: Processed rates for projection engine
**Approach**: Launch sub-agent to implement processor

### Step 2: Implement `survival_rates.py`
**Input**: SEER life tables
**Output**: Survival rates by cohort
**Approach**: Launch sub-agent after fertility processor

### Step 3: Implement `migration_rates.py`
**Input**: IRS county flow data + allocation algorithms
**Output**: Age-specific migration by cohort
**Approach**: Launch sub-agent after survival processor

---

## 📋 BigQuery Utility: What to Use It For

Even though BigQuery doesn't have the demographic rates we need, it's still useful for:

1. **Validation**: Compare projection results to Census population estimates
2. **Base Population**: Alternative source for starting population
3. **Geographic Data**: FIPS codes, county names, boundaries
4. **Births Validation**: Compare projected births to actual natality data
5. **Migration Trends**: Aggregate migration for validation

---

## 🚀 Recommendation: Proceed with SEER Data Processors

**Action Plan**:
1. ✅ BigQuery is set up and working
2. → Implement `fertility_rates.py` using sub-agent (assumes SEER download)
3. → Implement `survival_rates.py` using sub-agent
4. → Implement `migration_rates.py` using sub-agent
5. → Create data download scripts for SEER/IRS data
6. → Integrate BigQuery for validation and supplementary data

**Ready to launch sub-agent for fertility_rates.py processor?**
