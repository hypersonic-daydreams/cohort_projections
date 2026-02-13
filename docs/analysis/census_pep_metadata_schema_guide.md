# Census PEP Metadata Schema Usage Guide

**Database:** `census_popest`
**Schema Version:** 1.0.0
**Last Updated:** 2026-02-03
**Related:** [ADR-035](../governance/adrs/035-migration-data-source-census-pep.md), [PEP Vintage Differences Analysis](./pep_vintage_differences_migration.md)

---

## Executive Summary

This guide explains the comprehensive metadata tracking system for Census Population Estimates Program (PEP) county migration data. The system uses **5 interconnected PostgreSQL tables** to enable rigorous time series construction from multiple vintages with varying data quality levels.

**Key Benefits:**
- **Explicit provenance**: Every observation knows its source dataset and quality level
- **Queryable uncertainty**: Filter by revision status, data quality, or uncertainty level
- **Time series clarity**: Explicit rules handle 2020 overlap (postcensal vs. intercensal)
- **Data quality transparency**: Validation failures documented and query able
- **Future-proof**: New vintages add rows, not schema changes
- **Agent-friendly**: Self-documenting for AI agents and humans

---

## Data Architecture Overview

### The Challenge

We combine migration data from **three Census PEP datasets** spanning 2000-2024:

| Dataset ID | Years | Type | Status | Uncertainty | Validation Pass Rate |
|------------|-------|------|--------|-------------|----------------------|
| `co-est2009-alldata` | 2000-2009 | Postcensal | Final | Moderate | 80% (8/10 years) |
| `co-est2019-alldata` | 2010-2019 | Postcensal | Superseded | Moderate | **30% (3/10 years)** ⚠️ |
| `co-est2024-alldata` | 2020-2024 | Postcensal | Current | **High** | 100% (5/5 years) ✅ |

**Key Issues Addressed:**
1. **2020 overlap**: Appears in both 2010-2019 and 2020-2024 files - which to use?
2. **Quality variation**: 2010-2019 has poor hierarchical validation (Bakken boom/bust volatility)
3. **Uncertainty levels**: Recent data not yet census-aligned
4. **Intercensal limitation**: Intercensal components of change **not published by Census Bureau**

---

## Schema Components

### 1. census_pep_datasets
**Purpose:** Master registry of all PEP datasets/vintages

**Key Columns:**
- `dataset_id` (PK): Unique identifier (e.g., `co-est2009-alldata`)
- `vintage_label`: Human-readable label (e.g., `2000-2009`)
- `estimate_type`: `postcensal`, `intercensal`, or `vintage`
- `revision_status`: `final`, `current`, or `superseded`
- `uncertainty_level`: `low`, `moderate`, or `high`
- `hierarchical_validation_pass_rate`: % of years passing county ↔ state validation
- `source_file_sha256`: File integrity verification

**Example Query:**
```sql
-- Get all datasets sorted by quality
SELECT
    dataset_id,
    vintage_label,
    uncertainty_level,
    revision_status,
    hierarchical_validation_pass_rate
FROM census_pep_datasets
ORDER BY year_range_start;
```

**Result:**
```
     dataset_id     | vintage_label | uncertainty_level | revision_status | pass_rate
--------------------+---------------+-------------------+-----------------+-----------
 co-est2009-alldata | 2000-2009     | moderate          | final           | 0.80
 co-est2019-alldata | 2010-2019     | moderate          | superseded      | 0.30
 co-est2024-alldata | 2020-2024     | high              | current         | 1.00
```

---

### 2. census_pep_county_migration
**Purpose:** County-year migration observations with observation-level metadata

**Key Columns:**
- `geoid` + `year` + `dataset_id` (Composite PK)
- `netmig`, `intl_mig`, `domestic_mig`, `residual`: Migration values
- `estimate_type`, `revision_status`, `uncertainty_level`: Denormalized metadata
- `data_quality_score`: `pass`, `warning`, or `fail`
- `is_preferred_estimate`: **TRUE for recommended time series values**

**Example Query:**
```sql
-- Get recommended time series for a specific county
SELECT
    year,
    county_name,
    netmig,
    intl_mig,
    domestic_mig,
    data_quality_score,
    dataset_id
FROM census_pep_county_migration
WHERE geoid = '38017'  -- Cass County, ND
  AND is_preferred_estimate = TRUE
ORDER BY year;
```

**Distribution Stats (All Observations):**
- Total: 1,325 county-year observations
- Quality: 64% pass, 24% warning, 12% fail
- Preferred: 1,325 (100% - one preferred estimate per county-year)

---

### 3. census_pep_validation
**Purpose:** Validation results by year and dataset

**Key Columns:**
- `year` + `dataset_id` + `validation_type` (Composite PK)
- `passed`: Boolean validation result
- `county_sum` vs. `state_total`: Hierarchical consistency check
- `percent_difference`: Magnitude of discrepancy

**Example Query:**
```sql
-- Find years with validation failures
SELECT
    year,
    dataset_id,
    percent_difference,
    county_sum,
    state_total
FROM census_pep_validation
WHERE validation_type = 'hierarchical_consistency'
  AND NOT passed
ORDER BY ABS(percent_difference) DESC;
```

**Worst Validation Failures:**
```
 year |     dataset_id     | percent_difference | county_sum | state_total
------+--------------------+--------------------+------------+-------------
 2019 | co-est2019-alldata |              51.23 |       -316 |        -648
 2018 | co-est2019-alldata |             -36.46 |     -1,295 |        -949
 2012 | co-est2019-alldata |              -5.75 |     11,588 |      12,295
```

**Pattern:** All failures in 2010-2019 (Bakken boom/bust volatility)

---

### 4. census_pep_timeseries_rules
**Purpose:** Explicit rules for handling overlapping years

**Key Columns:**
- `year` (PK)
- `preferred_dataset_id`: Which dataset to use for this year
- `alternative_dataset_id`: Other available dataset (if any)
- `rationale`: Why this dataset is preferred

**Critical Rule - Year 2020 Overlap:**
```sql
SELECT * FROM census_pep_timeseries_rules WHERE year = 2020;
```

**Result:**
```
 year | preferred_dataset_id | alternative_dataset_id | rationale
------+----------------------+------------------------+-----------
 2020 | co-est2024-alldata   | co-est2019-alldata     | Year 2020 appears in both datasets.
                                                       | Prefer co-est2024-alldata for continuous
                                                       | series and 2020 Census base integration.
```

**Rationale:** The 2020-2024 vintage integrates the 2020 Census base, making it more accurate than the 2010-2019 extrapolation.

---

### 5. census_pep_extraction_log
**Purpose:** Audit trail of all extraction operations

**Key Columns:**
- `extraction_id` + `dataset_id` (Composite PK)
- `extraction_timestamp`: When extraction ran
- `rows_extracted`: Number of observations per dataset
- `validation_passed`: Overall validation status
- `output_file_sha256`: Checksum of output file

**Example Query:**
```sql
-- Get most recent extraction
SELECT
    extraction_id,
    dataset_id,
    extraction_timestamp,
    rows_extracted,
    validation_passed
FROM census_pep_extraction_log
ORDER BY extraction_timestamp DESC
LIMIT 5;
```

---

## Views for Common Use Cases

### View 1: census_pep_county_migration_preferred
**Use:** Get recommended time series (handles 2020 overlap automatically)

```sql
-- Get all preferred estimates for North Dakota
SELECT
    year,
    county_name,
    netmig,
    data_quality_score
FROM census_pep_county_migration_preferred
WHERE year BETWEEN 2010 AND 2020
ORDER BY year, county_name;
```

**Note:** This view filters `WHERE is_preferred_estimate = TRUE`, giving you exactly one row per county-year.

---

### View 2: census_pep_county_migration_highquality
**Use:** Restrict analysis to high-quality observations only

```sql
-- Regime analysis using only high-quality data
SELECT
    CASE
        WHEN year BETWEEN 2000 AND 2010 THEN 'Pre-Bakken'
        WHEN year BETWEEN 2011 AND 2015 THEN 'Boom'
        WHEN year BETWEEN 2016 AND 2021 THEN 'Bust+COVID'
        WHEN year BETWEEN 2022 AND 2024 THEN 'Recovery'
    END AS regime,
    COUNT(*) AS n_obs,
    AVG(netmig) AS avg_netmig
FROM census_pep_county_migration_highquality
GROUP BY regime
ORDER BY MIN(year);
```

**Filters applied:**
- `data_quality_score = 'pass'`
- `uncertainty_level IN ('low', 'moderate')`
- `is_preferred_estimate = TRUE`

---

### View 3: census_pep_county_migration_enriched
**Use:** Diagnostic analysis with full metadata context

```sql
-- Investigate 2010-2019 validation failures
SELECT
    year,
    county_name,
    netmig,
    validation_error_pct,
    hierarchical_validation_passed,
    vintage_pass_rate
FROM census_pep_county_migration_enriched
WHERE vintage_label = '2010-2019'
  AND hierarchical_validation_passed = FALSE
ORDER BY ABS(validation_error_pct) DESC;
```

**Includes:** Dataset metadata + validation results joined

---

### View 4: census_pep_dataset_summary
**Use:** Overview of all datasets with statistics

```sql
SELECT * FROM census_pep_dataset_summary;
```

**Shows:**
- Coverage (years, counties, observations)
- Quality metrics (pass rate, mean residual)
- Descriptive stats (mean/stddev netmig)

---

### View 5: census_pep_validation_summary
**Use:** Year-level validation overview

```sql
-- Identify problematic years
SELECT
    year,
    hierarchical_pass,
    hierarchical_fail,
    mean_pct_error,
    max_pct_error
FROM census_pep_validation_summary
WHERE hierarchical_fail > 0
ORDER BY mean_pct_error DESC;
```

---

## Time Series Construction Guide

### Recommended Approach

**For most analyses, use the preferred estimates view:**

```sql
CREATE TABLE my_analysis AS
SELECT * FROM census_pep_county_migration_preferred
WHERE data_quality_score != 'fail';  -- Optional: exclude failures
```

This handles:
- ✅ 2020 overlap (prefers 2020-2024 version)
- ✅ One observation per county-year
- ✅ Proper metadata (estimate type, uncertainty, quality)

---

### Advanced: Custom Time Series

**If you need custom rules (e.g., prefer high-quality only, drop 2020-2024):**

```sql
SELECT
    geoid,
    year,
    county_name,
    netmig,
    intl_mig,
    domestic_mig,
    dataset_id
FROM census_pep_county_migration
WHERE (
    -- Use 2000-2009 for all years
    (year BETWEEN 2000 AND 2009 AND dataset_id = 'co-est2009-alldata')
    OR
    -- Use 2010-2019 for 2010-2019 (including 2020 from this file)
    (year BETWEEN 2010 AND 2020 AND dataset_id = 'co-est2019-alldata')
    -- Exclude 2020-2024 entirely
)
AND data_quality_score IN ('pass', 'warning')  -- Exclude failures
ORDER BY year, geoid;
```

**⚠️ Warning:** This gives you 2020 from the 2010-2019 file instead of 2020-2024. Only do this if you have a specific reason.

---

## Data Quality Assessment

### Understanding Quality Scores

| Score | Meaning | Hierarchical Validation | Action |
|-------|---------|-------------------------|--------|
| **pass** | County sums match state totals within 1% | ✅ Passed | Use normally |
| **warning** | Discrepancy between 1% and 5% | ⚠️ Failed but close | Use with caution |
| **fail** | Discrepancy > 5% | ❌ Failed significantly | Consider excluding |

### Quality by Vintage

```sql
SELECT
    d.vintage_label,
    d.uncertainty_level,
    m.data_quality_score,
    COUNT(*) AS observations,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY d.vintage_label), 1) AS pct
FROM census_pep_county_migration m
JOIN census_pep_datasets d USING (dataset_id)
GROUP BY d.vintage_label, d.uncertainty_level, m.data_quality_score
ORDER BY d.vintage_label, m.data_quality_score;
```

**Expected Pattern:**
- **2000-2009**: Mostly pass (~80%)
- **2010-2019**: More failures (~30% pass) - **This is documented and expected**
- **2020-2024**: All pass (100%)

---

## Handling Uncertainty

### Uncertainty Levels Explained

| Level | Definition | Example | Implication |
|-------|-----------|---------|-------------|
| **low** | Intercensal (census-aligned) | N/A (components not published) | Highest confidence |
| **moderate** | Postcensal from completed decade | 2000-2009 | Good confidence |
| **high** | Recent postcensal (not yet revised) | 2020-2024 | Lower confidence, subject to revision |

### Filtering by Uncertainty

```sql
-- Conservative analysis: Exclude high-uncertainty data
SELECT * FROM census_pep_county_migration_preferred
WHERE uncertainty_level IN ('low', 'moderate');

-- Inclusive analysis: Use all data but weight by uncertainty
SELECT
    year,
    AVG(netmig) AS mean_netmig,
    -- Weight: low=1.0, moderate=0.8, high=0.6
    AVG(netmig * CASE uncertainty_level
        WHEN 'low' THEN 1.0
        WHEN 'moderate' THEN 0.8
        WHEN 'high' THEN 0.6
    END) AS weighted_mean_netmig
FROM census_pep_county_migration_preferred
GROUP BY year
ORDER BY year;
```

---

## Why Intercensal Components Don't Exist

**Finding:** The Census Bureau publishes intercensal **population estimates** (by age/sex/race) but NOT intercensal **components of change** (births, deaths, migration).

**Files Available:**
- ✅ `cc-est2020int-alldata`: Intercensal population by age/sex/race (2010-2020)
- ❌ Intercensal components of change (2010-2020): **Not published**

**Implications:**
1. We use **2010-2019 postcensal** components (best available)
2. Lower validation pass rate (30%) is **documented and accepted**
3. Metadata tables track this as `revision_status = 'superseded'` with notes

**Alternative:** Calculate migration residually from intercensal population + vital stats (deferred - see ADR-035)

---

## Helper Functions

### update_preferred_estimates()
**Purpose:** Recalculate `is_preferred_estimate` flags based on time series rules

```sql
-- After modifying timeseries rules, refresh preferred flags
SELECT update_preferred_estimates();
```

**Returns:** Number of rows updated

**When to use:**
- After inserting/updating `census_pep_timeseries_rules`
- After loading new data
- To verify rule application

---

## Example Workflows

### Workflow 1: Basic Time Series Extraction

```sql
-- Export preferred estimates to CSV
COPY (
    SELECT
        year,
        geoid,
        county_name,
        netmig,
        intl_mig,
        domestic_mig,
        data_quality_score,
        dataset_id
    FROM census_pep_county_migration_preferred
    ORDER BY year, geoid
) TO '/tmp/pep_migration_2000_2024.csv' WITH CSV HEADER;
```

---

### Workflow 2: Regime-Specific Averages

```sql
-- Calculate averages by county and regime
CREATE TABLE migration_regime_averages AS
WITH regimes AS (
    SELECT
        geoid,
        county_name,
        'Pre-Bakken' AS regime,
        AVG(netmig) AS avg_netmig,
        STDDEV(netmig) AS stddev_netmig,
        COUNT(*) AS n_years
    FROM census_pep_county_migration_preferred
    WHERE year BETWEEN 2000 AND 2010
    GROUP BY geoid, county_name

    UNION ALL

    SELECT
        geoid,
        county_name,
        'Boom' AS regime,
        AVG(netmig),
        STDDEV(netmig),
        COUNT(*)
    FROM census_pep_county_migration_preferred
    WHERE year BETWEEN 2011 AND 2015
    GROUP BY geoid, county_name

    -- ... repeat for other regimes
)
SELECT * FROM regimes
ORDER BY geoid, regime;
```

---

### Workflow 3: Data Quality Report

```sql
-- Generate data quality report by county
SELECT
    county_name,
    COUNT(*) AS total_years,
    SUM(CASE WHEN data_quality_score = 'pass' THEN 1 ELSE 0 END) AS years_pass,
    SUM(CASE WHEN data_quality_score = 'warning' THEN 1 ELSE 0 END) AS years_warning,
    SUM(CASE WHEN data_quality_score = 'fail' THEN 1 ELSE 0 END) AS years_fail,
    ROUND(100.0 * SUM(CASE WHEN data_quality_score = 'pass' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pass_rate
FROM census_pep_county_migration_preferred
GROUP BY county_name
ORDER BY pass_rate, county_name;
```

---

## FAQ

### Q: Why does 2010-2019 have such poor validation (30% pass rate)?

**A:** This period includes the Bakken oil boom/bust with extreme population volatility in western ND counties (Williams, McKenzie, Mountrail). The residual method may have larger errors during periods of rapid change. This is documented in metadata and expected.

### Q: Should I use 2020 from 2010-2019 or 2020-2024?

**A:** Use **2020-2024** (the default in `census_pep_county_migration_preferred`). The 2020-2024 vintage integrates the 2020 Census base, making it more accurate than the 2010-2019 extrapolation.

### Q: Why not use intercensal data for 2010-2020?

**A:** Intercensal **population estimates** exist, but intercensal **components of change** (migration, births, deaths) are not published by the Census Bureau. We use the best available: 2010-2019 postcensal components.

### Q: Can I trust 2022-2024 data given "high" uncertainty?

**A:** Yes, but with caveats:
- These are postcensal estimates not yet revised by the 2030 Census
- 100% validation pass rate suggests good quality
- Use for recent trends, but acknowledge higher uncertainty in documentation

### Q: How do I add a new vintage (e.g., 2025)?

**Steps:**
1. Add to `VINTAGES` dict in extraction script
2. Run extraction script → populates all metadata tables automatically
3. Add timeseries rules for new years
4. Run `SELECT update_preferred_estimates()`

---

## Related Documentation

- **[ADR-035: Migration Data Source - Census PEP](../governance/adrs/035-migration-data-source-census-pep.md)** - Decision rationale
- **[PEP Vintage Differences Analysis](./pep_vintage_differences_migration.md)** - Detailed vintage comparison
- **[ADR-034: Census PEP Data Archive](../governance/adrs/034-census-pep-data-archive.md)** - Data infrastructure

---

## Maintenance

### Schema Migrations

**Current Version:** 1.0.0

**Check version:**
```sql
SELECT * FROM census_pep_schema_version ORDER BY applied_at DESC;
```

**Future migrations:** Add new rows to track changes

---

## Contact & Support

For questions about this metadata system:
- Review [ADR-035](../governance/adrs/035-migration-data-source-census-pep.md) for design decisions
- Check [PEP Vintage Differences](./pep_vintage_differences_migration.md) for data quality details
- Consult PostgreSQL views for common queries

**Database:** `census_popest` (via `$CENSUS_POPEST_PG_DSN`)

---

**Last Updated:** 2026-02-03
**Schema Version:** 1.0.0
**Status:** ✅ Production-ready
