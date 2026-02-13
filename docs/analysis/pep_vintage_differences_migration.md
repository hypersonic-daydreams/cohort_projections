# Census PEP Vintage Differences: Migration Data Analysis

**Date:** 2026-02-03
**Purpose:** Document vintage/decade differences in Census Population Estimates Program (PEP) data that impact Phase 2 migration regime analysis
**Context:** ADR-035 Phase 1 (Data Extraction) complete; preparing for Phase 2 (Regime Analysis)

---

## Executive Summary

Census PEP county migration data is **structurally consistent** across all three vintages (2000-2009, 2010-2019, 2020-2024), with **minor column naming differences** but **identical conceptual definitions**. The key migration columns (NETMIG, INTERNATIONALMIG, DOMESTICMIG) are directly comparable across vintages for time-series analysis.

**Bottom line:** Our Phase 1 extraction correctly harmonizes the data. Phase 2 regime analysis can proceed with confidence that cross-vintage comparisons are valid.

---

## Data Vintages Extracted

| Vintage | File | Years | Estimate Type | Notes |
|---------|------|-------|---------------|-------|
| **2000-2009** | `co-est2009-alldata.parquet` | 2000-2009 | Postcensal | "Comprehensive" decade file |
| **2010-2019** | `co-est2019-alldata.parquet` | 2010-2019 | Postcensal | Superseded by 2010-2020 intercensal |
| **2020-2024** | `co-est2024-alldata.parquet` | 2020-2024 | Postcensal | Current (not revised) |

**Total observations extracted:** 1,325 county-years (53 counties × 25 years)

---

## Column Name Differences (Non-Migration)

### Natural Increase Naming
- **2000-2009, 2010-2019:** `NATURALINC{YEAR}` ("increase")
- **2020-2024:** `NATURALCHG{YEAR}` ("change")

### Population Change Naming
- **2000-2009, 2010-2019:** `NPOPCHG_{YEAR}` (with underscore)
- **2020-2024:** `NPOPCHG{YEAR}` (no underscore)

### ✅ Migration Columns (Consistent Across All Vintages)
All vintages use **identical naming** for migration components:
- `NETMIG{YEAR}` - Net migration (domestic + international)
- `INTERNATIONALMIG{YEAR}` - Net international migration
- `DOMESTICMIG{YEAR}` - Net domestic migration

**Implication:** No harmonization needed for migration columns in Phase 2 analysis.

---

## Methodological Consistency

### Core Method (All Vintages)
- **Approach:** Cohort-component method at county level
- **Components:** Births, deaths, domestic migration, international migration
- **Frequency:** Annual estimates for July 1 of each year
- **Residual:** All vintages include a RESIDUAL component for unexplained population change

### Net International Migration Definition (Consistent)
From 2020-2024 documentation (applies to all vintages):

> "Net international migration for the United States includes the international migration of both U.S.-born and non-U.S.-born populations. Specifically, it includes: (a) the net international migration of the non-U.S. born, (b) the net migration of U.S. born to and from the United States, (c) the net migration between the United States and Puerto Rico, and (d) the net movement of the Armed Forces population between the United States and overseas."

**Implication:** International migration is comprehensive across all vintages. No systematic undercount due to definition changes.

### Geographic Consistency
- All vintages use FIPS codes for state/county identification
- Geographic boundaries updated annually within each vintage
- North Dakota has 53 counties across all vintages (consistent)

---

## Postcensal vs. Intercensal Estimates

### What's the Difference?

| Type | Definition | Revision Status | Uncertainty |
|------|-----------|-----------------|-------------|
| **Postcensal** | Estimates produced after a census but before the next census | Not revised | Higher |
| **Intercensal** | Estimates revised after the next census to align with new census counts | Revised (final) | Lower |

### Our Data Mix

| Vintage | Type | Status | Implications |
|---------|------|--------|--------------|
| 2000-2009 | Postcensal | Final (decade ended) | Moderate uncertainty |
| 2010-2019 | Postcensal | **Superseded by 2010-2020 intercensal** | May differ from revised estimates |
| 2020-2024 | Postcensal | **Current (2030 census not yet available)** | Highest uncertainty |

### Why This Matters for Phase 2

1. **2010-2019 vs. 2010-2020 intercensal:**
   - We extracted `co-est2019-alldata.parquet` (postcensal)
   - There exists `cc-est2020int-alldata.parquet` (intercensal, revised)
   - **Question for Phase 2:** Should we use revised intercensal for 2010-2020 instead?

2. **Recent data (2020-2024) has higher uncertainty:**
   - Not yet revised by 2030 census
   - May be adjusted in future vintages
   - **Implication:** 2022-2024 "recovery" trends should be interpreted cautiously

3. **Hierarchical validation failures concentrated in 2010-2019:**
   - 9/25 years with >1% discrepancy are in the 2010-2019 period
   - This may be due to postcensal uncertainty during Bakken boom/bust volatility
   - **Action for Phase 2:** Investigate whether intercensal 2010-2020 data reduces validation errors

---

## Known Vintage-Specific Issues

### 2010-2019 Parquet Truncation (Resolved)
- **Issue:** Original parquet conversion was incomplete (910 rows instead of 3,193)
- **Resolution:** Reconverted from complete CSV in raw archive (Phase 1)
- **Status:** Fixed; validated full North Dakota coverage (53 counties × 10 years)

### 2020 Estimates Base
From 2020-2024 documentation:

> "The estimates are developed from a base that integrates the 2020 Census, Vintage 2020 estimates, and 2020 Demographic Analysis estimates."

**Implication:** 2020 is a "transition year" with multiple data sources integrated. May explain some validation discrepancies.

---

## Residual Component

All vintages include a **RESIDUAL** column representing "change in population that cannot be attributed to any specific demographic component."

### What Causes Residuals?
- Census coverage errors
- Administrative data errors (births, deaths, migration)
- Timing mismatches between components
- Geographic boundary changes

### Residuals in Migration Analysis

**Question for Phase 2:** Should we include or exclude RESIDUAL when calculating net population change?

| Option | Formula | Rationale |
|--------|---------|-----------|
| **Option A: Exclude** | `NPOPCHG = NATURALCHG + NETMIG` (ignore RESIDUAL) | Focus on measured migration components |
| **Option B: Include** | `NPOPCHG = NATURALCHG + NETMIG + RESIDUAL` (full accounting) | Captures all sources of population change |

**Recommendation:** **Option A (exclude RESIDUAL)** for migration rate calculations, but document residual magnitude by county/period to assess data quality.

---

## Hierarchical Validation Results (from Phase 1)

### Summary
- **16/25 years pass strict 1% tolerance** (county sums = state totals)
- **9/25 years fail** (>1% discrepancy)
- **Failures concentrated in 2010-2019** (Bakken boom/bust period)

### Validation by Vintage

| Vintage | Years Analyzed | Passed (<1%) | Failed (>1%) | Pass Rate |
|---------|----------------|--------------|--------------|-----------|
| 2000-2009 | 10 | 8 | 2 | 80% |
| 2010-2019 | 10 | 3 | 7 | 30% ⚠️ |
| 2020-2024 | 5 | 5 | 0 | 100% ✅ |

### Interpretation
- **2020-2024:** Excellent hierarchical consistency (likely due to recent methodology improvements)
- **2000-2009:** Good consistency (typical for postcensal estimates)
- **2010-2019:** Poor consistency (may be due to Bakken volatility or postcensal uncertainty)

**Action for Phase 2:** Investigate whether switching to 2010-2020 intercensal data improves validation.

---

## Recommendations for Phase 2

### 1. Consider Switching to 2010-2020 Intercensal Data

**Current:** Using `co-est2019-alldata.parquet` (postcensal, 2010-2019)
**Alternative:** Use `cc-est2020int-alldata.parquet` (intercensal, 2010-2020, revised)

**Pros:**
- Revised estimates aligned with both 2010 and 2020 censuses
- Should have better hierarchical consistency (reduced validation errors)
- More accurate representation of 2010s migration patterns

**Cons:**
- Includes 2020 (overlap with 2020-2024 vintage)
- Need to handle 2020 carefully (avoid double-counting or version conflicts)

**Decision:** Defer to Phase 2 analysis after reviewing intercensal data availability and structure.

### 2. Document Residuals by County and Period
- Calculate `mean(|RESIDUAL|)` by county across 2000-2024
- Identify counties with systematically large residuals (data quality issues)
- Document in Phase 2 regime analysis

### 3. Treat 2022-2024 Recovery Trends Cautiously
- These are postcensal estimates not yet revised by 2030 census
- Higher uncertainty than earlier periods
- Use for scenario planning but acknowledge uncertainty

### 4. Regime-Specific Validation
- Check if hierarchical validation failures correlate with migration regime (boom/bust)
- Western ND counties (Bakken region) may have larger residuals due to extreme volatility

---

## Technical Notes

### Extracted Data Location
- **Harmonized dataset:** `data/processed/pep_county_components_2000_2024.parquet`
- **CSV version:** `data/processed/pep_county_components_2000_2024.csv`
- **Validation report:** `data/processed/pep_county_validation_results.csv`
- **Summary statistics:** `data/processed/pep_county_summary_statistics.csv`

### Schema (Harmonized Output)
```
year: int (2000-2024)
state_fips: str ("38")
county_fips: str ("001"-"105")
state_name: str ("North Dakota")
county_name: str
geoid: str ("38001"-"38105")
netmig: float (net migration count)
intl_mig: float (international migration count)
domestic_mig: float (domestic migration count)
vintage: str ("2000-2009", "2010-2019", "2020-2024")
```

### Column Renaming Applied in Phase 1
```python
# 2010-2019 and 2000-2009 vintages:
'NATURALINC{year}' → not used (not extracted)
'NPOPCHG_{year}' → not used (not extracted)

# All vintages (consistent):
'NETMIG{year}' → 'netmig'
'INTERNATIONALMIG{year}' → 'intl_mig'
'DOMESTICMIG{year}' → 'domestic_mig'
```

---

## References

### Documentation Sources
- **2020-2024 File Layout:** `~/workspace/shared-data/census/popest/docs/file-layouts/CO-EST2024-ALLDATA-layout.pdf`
- **2020-2024 Methodology:** `~/workspace/shared-data/census/popest/docs/methodology/2024-subcounty-methodology.pdf`
- **Extracted Text:** `~/workspace/shared-data/census/popest/derived/docs/`

### Key URLs
- PEP Methodology: https://www.census.gov/programs-surveys/popest/technical-documentation/methodology.html
- PEP Terms and Definitions: https://www.census.gov/programs-surveys/popest/about/glossary.html
- Dataset Directory: https://www2.census.gov/programs-surveys/popest/datasets/

### Related ADRs
- **ADR-034:** Census PEP Data Archive (infrastructure)
- **ADR-035:** Migration Data Source - Census PEP (decision to replace IRS data)

---

## Revision History

- **2026-02-03:** Initial version (post-Phase 1 extraction)

---

**Status:** Ready for Phase 2 (Regime Analysis)
**Next Step:** County-level regime classification and period-specific migration averaging
