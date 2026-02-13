-- Census PEP Metadata Schema
-- Database: census_popest
--
-- This schema provides comprehensive metadata tracking for Census Population
-- Estimates Program (PEP) data used in cohort projections. It enables rigorous
-- time series construction from multiple vintages with varying quality levels.
--
-- Design Principles:
-- 1. Explicit provenance: Every observation knows its source
-- 2. Queryable uncertainty: Filter by quality/revision status
-- 3. Time series clarity: Explicit rules for overlapping years
-- 4. Data quality transparency: Validation results stored and queryable
-- 5. Future-proof: New vintages add rows, not schema changes
-- 6. Agent-friendly: Self-documenting metadata for AI agents
--
-- Usage:
--   psql -d census_popest -f scripts/db/census_pep_metadata_schema.sql
--
-- Related: ADR-035 (Migration Data Source)

-- =============================================================================
-- ENUMS
-- =============================================================================

-- Estimate type classification
CREATE TYPE estimate_type AS ENUM (
    'postcensal',    -- Produced after census but before next census (not revised)
    'intercensal',   -- Revised after next census to align with census counts
    'vintage'        -- Specific release/vintage (may be postcensal or intercensal)
);

-- Revision status for vintages
CREATE TYPE revision_status AS ENUM (
    'final',         -- Revised and aligned with subsequent census (intercensal)
    'current',       -- Most recent available (postcensal, not yet revised)
    'superseded'     -- Replaced by a newer/revised version
);

-- Uncertainty level assessment
CREATE TYPE uncertainty_level AS ENUM (
    'low',           -- Intercensal estimates (revised)
    'moderate',      -- Postcensal estimates from completed decades
    'high'           -- Recent postcensal estimates (not yet census-aligned)
);

-- Data quality assessment
CREATE TYPE data_quality_score AS ENUM (
    'pass',          -- Passed all validation checks
    'warning',       -- Passed with minor issues
    'fail'           -- Failed validation checks
);

-- =============================================================================
-- TABLE 1: DATASET-LEVEL METADATA
-- =============================================================================

-- Master table of PEP datasets/vintages
CREATE TABLE census_pep_datasets (
    -- Primary identification
    dataset_id TEXT PRIMARY KEY,           -- e.g., 'co-est2020int-alldata'
    vintage_label TEXT NOT NULL,           -- e.g., '2010-2020'

    -- Classification
    estimate_type estimate_type NOT NULL,
    revision_status revision_status NOT NULL,
    uncertainty_level uncertainty_level NOT NULL,

    -- Temporal coverage
    year_range_start INT NOT NULL,
    year_range_end INT NOT NULL,

    -- Relationships
    supersedes_dataset TEXT REFERENCES census_pep_datasets(dataset_id),

    -- Provenance
    source_file_path TEXT NOT NULL,        -- Relative path from CENSUS_POPEST_DIR
    source_file_sha256 TEXT NOT NULL,
    extraction_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extracted_by TEXT NOT NULL,            -- Script name/version

    -- Documentation references (path/URL + checksums for verification)
    methodology_doc_path TEXT,
    methodology_doc_sha256 TEXT,
    file_layout_doc_path TEXT,
    file_layout_doc_sha256 TEXT,

    -- Quality summary
    hierarchical_validation_pass_rate FLOAT,  -- % of years passing validation
    mean_absolute_residual FLOAT,             -- Average |RESIDUAL| for this vintage

    -- Metadata
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CHECK (year_range_start <= year_range_end),
    CHECK (hierarchical_validation_pass_rate >= 0 AND hierarchical_validation_pass_rate <= 1),
    CHECK (mean_absolute_residual IS NULL OR mean_absolute_residual >= 0)
);

-- Indexes for common queries
CREATE INDEX idx_census_pep_datasets_vintage ON census_pep_datasets(vintage_label);
CREATE INDEX idx_census_pep_datasets_years ON census_pep_datasets(year_range_start, year_range_end);
CREATE INDEX idx_census_pep_datasets_estimate_type ON census_pep_datasets(estimate_type);
CREATE INDEX idx_census_pep_datasets_revision_status ON census_pep_datasets(revision_status);

-- =============================================================================
-- TABLE 2: OBSERVATION-LEVEL DATA (COUNTY-YEAR MIGRATION)
-- =============================================================================

-- Core migration data with observation-level metadata
CREATE TABLE census_pep_county_migration (
    -- Identifiers
    geoid TEXT NOT NULL,                   -- State + County FIPS (e.g., '38001')
    year INT NOT NULL,
    state_fips TEXT NOT NULL,
    county_fips TEXT NOT NULL,
    county_name TEXT NOT NULL,

    -- Migration values
    netmig FLOAT,                          -- Net migration (domestic + international)
    intl_mig FLOAT,                        -- International migration component
    domestic_mig FLOAT,                    -- Domestic migration component
    residual FLOAT,                        -- Unexplained population change

    -- Metadata (denormalized for query performance)
    dataset_id TEXT NOT NULL REFERENCES census_pep_datasets(dataset_id),
    estimate_type estimate_type NOT NULL,
    revision_status revision_status NOT NULL,
    uncertainty_level uncertainty_level NOT NULL,

    -- Data quality
    data_quality_score data_quality_score,
    validation_notes TEXT,                 -- e.g., "Hierarchical validation failed by 3.2%"

    -- Time series construction
    is_preferred_estimate BOOLEAN NOT NULL DEFAULT FALSE,  -- For overlapping years

    -- Constraints
    PRIMARY KEY (geoid, year, dataset_id),
    CHECK (LENGTH(state_fips) = 2),
    CHECK (LENGTH(county_fips) = 3),
    CHECK (LENGTH(geoid) = 5)
);

-- Indexes for common query patterns
CREATE INDEX idx_pep_migration_geoid_year ON census_pep_county_migration(geoid, year);
CREATE INDEX idx_pep_migration_year ON census_pep_county_migration(year);
CREATE INDEX idx_pep_migration_dataset ON census_pep_county_migration(dataset_id);
CREATE INDEX idx_pep_migration_preferred ON census_pep_county_migration(is_preferred_estimate)
    WHERE is_preferred_estimate = TRUE;
CREATE INDEX idx_pep_migration_quality ON census_pep_county_migration(data_quality_score);
CREATE INDEX idx_pep_migration_uncertainty ON census_pep_county_migration(uncertainty_level);

-- Composite index for time series queries
CREATE INDEX idx_pep_migration_timeseries ON census_pep_county_migration(geoid, year, is_preferred_estimate);

-- =============================================================================
-- TABLE 3: VALIDATION RESULTS (YEAR-LEVEL QUALITY METRICS)
-- =============================================================================

-- Track validation results by year and dataset
CREATE TABLE census_pep_validation (
    year INT NOT NULL,
    dataset_id TEXT NOT NULL REFERENCES census_pep_datasets(dataset_id),
    validation_type TEXT NOT NULL,         -- e.g., 'hierarchical_consistency', 'residual_magnitude'

    -- Metrics
    passed BOOLEAN NOT NULL,
    county_sum FLOAT,                      -- Sum of county values
    state_total FLOAT,                     -- Published state total
    absolute_difference FLOAT,
    percent_difference FLOAT,

    -- Context
    notes TEXT,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    PRIMARY KEY (year, dataset_id, validation_type)
);

-- Indexes
CREATE INDEX idx_pep_validation_year ON census_pep_validation(year);
CREATE INDEX idx_pep_validation_dataset ON census_pep_validation(dataset_id);
CREATE INDEX idx_pep_validation_passed ON census_pep_validation(passed);
CREATE INDEX idx_pep_validation_type ON census_pep_validation(validation_type);

-- =============================================================================
-- TABLE 4: TIME SERIES CONSTRUCTION RULES
-- =============================================================================

-- Explicit rules for handling overlapping years (e.g., 2020)
CREATE TABLE census_pep_timeseries_rules (
    rule_id SERIAL PRIMARY KEY,
    year INT NOT NULL UNIQUE,

    -- Preferred source
    preferred_dataset_id TEXT NOT NULL REFERENCES census_pep_datasets(dataset_id),

    -- Alternative sources
    alternative_dataset_id TEXT REFERENCES census_pep_datasets(dataset_id),

    -- Documentation
    rationale TEXT NOT NULL,
    effective_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_by TEXT NOT NULL DEFAULT CURRENT_USER,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_pep_timeseries_year ON census_pep_timeseries_rules(year);
CREATE INDEX idx_pep_timeseries_preferred ON census_pep_timeseries_rules(preferred_dataset_id);

-- =============================================================================
-- TABLE 5: EXTRACTION PROVENANCE LOG
-- =============================================================================

-- Audit trail of all data extraction operations
CREATE TABLE census_pep_extraction_log (
    extraction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    extraction_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dataset_id TEXT NOT NULL REFERENCES census_pep_datasets(dataset_id),

    -- Script information
    script_name TEXT NOT NULL,
    script_version TEXT,
    configuration_yaml TEXT,               -- Full config used for extraction

    -- Results
    rows_extracted INT NOT NULL,
    validation_passed BOOLEAN NOT NULL,
    output_file_path TEXT NOT NULL,
    output_file_sha256 TEXT NOT NULL,

    -- Context
    notes TEXT,

    -- Constraints
    CHECK (rows_extracted >= 0)
);

-- Indexes
CREATE INDEX idx_pep_extraction_timestamp ON census_pep_extraction_log(extraction_timestamp DESC);
CREATE INDEX idx_pep_extraction_dataset ON census_pep_extraction_log(dataset_id);
CREATE INDEX idx_pep_extraction_script ON census_pep_extraction_log(script_name);

-- =============================================================================
-- VIEWS FOR COMMON USE CASES
-- =============================================================================

-- View 1: "Best available" time series (preferred estimates only)
CREATE VIEW census_pep_county_migration_preferred AS
SELECT
    geoid,
    year,
    state_fips,
    county_fips,
    county_name,
    netmig,
    intl_mig,
    domestic_mig,
    residual,
    dataset_id,
    estimate_type,
    revision_status,
    uncertainty_level,
    data_quality_score
FROM census_pep_county_migration
WHERE is_preferred_estimate = TRUE
ORDER BY geoid, year;

COMMENT ON VIEW census_pep_county_migration_preferred IS
'Recommended time series for analysis: one row per county-year using preferred estimates (handles 2020 overlap by preferring intercensal over postcensal)';

-- View 2: High-quality observations only
CREATE VIEW census_pep_county_migration_highquality AS
SELECT
    geoid,
    year,
    state_fips,
    county_fips,
    county_name,
    netmig,
    intl_mig,
    domestic_mig,
    residual,
    dataset_id,
    estimate_type,
    revision_status,
    uncertainty_level
FROM census_pep_county_migration
WHERE data_quality_score = 'pass'
  AND uncertainty_level IN ('low', 'moderate')
  AND is_preferred_estimate = TRUE
ORDER BY geoid, year;

COMMENT ON VIEW census_pep_county_migration_highquality IS
'High-quality subset: passed validation, low-to-moderate uncertainty, preferred estimates only';

-- View 3: Time series with enriched metadata
CREATE VIEW census_pep_county_migration_enriched AS
SELECT
    m.geoid,
    m.year,
    m.state_fips,
    m.county_fips,
    m.county_name,
    m.netmig,
    m.intl_mig,
    m.domestic_mig,
    m.residual,
    m.dataset_id,
    m.estimate_type,
    m.revision_status,
    m.uncertainty_level,
    m.data_quality_score,
    m.is_preferred_estimate,
    -- Dataset metadata
    d.vintage_label,
    d.source_file_path,
    d.extraction_timestamp,
    d.hierarchical_validation_pass_rate AS vintage_pass_rate,
    -- Validation results
    v.passed AS hierarchical_validation_passed,
    v.percent_difference AS validation_error_pct,
    v.county_sum,
    v.state_total
FROM census_pep_county_migration m
JOIN census_pep_datasets d USING (dataset_id)
LEFT JOIN census_pep_validation v
    ON m.year = v.year
    AND m.dataset_id = v.dataset_id
    AND v.validation_type = 'hierarchical_consistency'
ORDER BY m.geoid, m.year, m.dataset_id;

COMMENT ON VIEW census_pep_county_migration_enriched IS
'Full diagnostic view with dataset and validation metadata joined for analysis and quality assessment';

-- View 4: Dataset summary statistics
CREATE VIEW census_pep_dataset_summary AS
SELECT
    d.dataset_id,
    d.vintage_label,
    d.estimate_type,
    d.revision_status,
    d.uncertainty_level,
    d.year_range_start,
    d.year_range_end,
    d.hierarchical_validation_pass_rate,
    d.mean_absolute_residual,
    d.extraction_timestamp,
    -- Computed statistics
    COUNT(DISTINCT m.year) AS years_with_data,
    COUNT(DISTINCT m.geoid) AS counties_covered,
    COUNT(*) AS total_observations,
    AVG(m.netmig) AS mean_netmig,
    STDDEV(m.netmig) AS stddev_netmig,
    COUNT(*) FILTER (WHERE m.data_quality_score = 'pass') AS obs_passed,
    COUNT(*) FILTER (WHERE m.data_quality_score = 'warning') AS obs_warning,
    COUNT(*) FILTER (WHERE m.data_quality_score = 'fail') AS obs_failed,
    COUNT(*) FILTER (WHERE m.is_preferred_estimate = TRUE) AS obs_preferred
FROM census_pep_datasets d
LEFT JOIN census_pep_county_migration m USING (dataset_id)
GROUP BY
    d.dataset_id,
    d.vintage_label,
    d.estimate_type,
    d.revision_status,
    d.uncertainty_level,
    d.year_range_start,
    d.year_range_end,
    d.hierarchical_validation_pass_rate,
    d.mean_absolute_residual,
    d.extraction_timestamp
ORDER BY d.year_range_start;

COMMENT ON VIEW census_pep_dataset_summary IS
'Summary statistics for each dataset/vintage: coverage, quality, and basic descriptive stats';

-- View 5: Validation summary by year
CREATE VIEW census_pep_validation_summary AS
SELECT
    year,
    COUNT(DISTINCT dataset_id) AS num_datasets,
    COUNT(*) FILTER (WHERE validation_type = 'hierarchical_consistency' AND passed) AS hierarchical_pass,
    COUNT(*) FILTER (WHERE validation_type = 'hierarchical_consistency' AND NOT passed) AS hierarchical_fail,
    AVG(ABS(percent_difference)) FILTER (WHERE validation_type = 'hierarchical_consistency') AS mean_pct_error,
    MAX(ABS(percent_difference)) FILTER (WHERE validation_type = 'hierarchical_consistency') AS max_pct_error
FROM census_pep_validation
GROUP BY year
ORDER BY year;

COMMENT ON VIEW census_pep_validation_summary IS
'Year-level validation summary showing consistency across all available datasets for each year';

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to update preferred estimates based on timeseries rules
CREATE OR REPLACE FUNCTION update_preferred_estimates()
RETURNS INTEGER AS $$
DECLARE
    rows_updated INTEGER := 0;
BEGIN
    -- Reset all is_preferred_estimate flags
    UPDATE census_pep_county_migration
    SET is_preferred_estimate = FALSE;

    -- Set preferred estimates based on timeseries rules
    UPDATE census_pep_county_migration m
    SET is_preferred_estimate = TRUE
    FROM census_pep_timeseries_rules r
    WHERE m.year = r.year
      AND m.dataset_id = r.preferred_dataset_id;

    GET DIAGNOSTICS rows_updated = ROW_COUNT;

    RETURN rows_updated;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_preferred_estimates() IS
'Apply time series construction rules to set is_preferred_estimate flags. Run after loading new data or updating rules.';

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Trigger to automatically update dataset summary statistics
CREATE OR REPLACE FUNCTION update_dataset_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update hierarchical_validation_pass_rate for the dataset
    UPDATE census_pep_datasets d
    SET hierarchical_validation_pass_rate = (
        SELECT CAST(COUNT(*) FILTER (WHERE passed) AS FLOAT) / NULLIF(COUNT(*), 0)
        FROM census_pep_validation v
        WHERE v.dataset_id = d.dataset_id
          AND v.validation_type = 'hierarchical_consistency'
    ),
    mean_absolute_residual = (
        SELECT AVG(ABS(residual))
        FROM census_pep_county_migration m
        WHERE m.dataset_id = d.dataset_id
    )
    WHERE d.dataset_id = NEW.dataset_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger on validation table
CREATE TRIGGER update_dataset_stats_on_validation
    AFTER INSERT OR UPDATE ON census_pep_validation
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_stats();

-- Trigger on migration table
CREATE TRIGGER update_dataset_stats_on_migration
    AFTER INSERT OR UPDATE ON census_pep_county_migration
    FOR EACH ROW
    EXECUTE FUNCTION update_dataset_stats();

-- =============================================================================
-- TABLE COMMENTS
-- =============================================================================

COMMENT ON TABLE census_pep_datasets IS
'Master registry of Census PEP datasets/vintages with quality metrics and provenance';

COMMENT ON TABLE census_pep_county_migration IS
'County-year migration observations with observation-level metadata and quality flags';

COMMENT ON TABLE census_pep_validation IS
'Validation results by year and dataset for quality assessment';

COMMENT ON TABLE census_pep_timeseries_rules IS
'Explicit rules for handling overlapping years when constructing time series from multiple vintages';

COMMENT ON TABLE census_pep_extraction_log IS
'Audit trail of data extraction operations for provenance and reproducibility';

-- =============================================================================
-- COLUMN COMMENTS (Key fields)
-- =============================================================================

COMMENT ON COLUMN census_pep_datasets.estimate_type IS
'postcensal: not yet revised; intercensal: revised to align with both censuses';

COMMENT ON COLUMN census_pep_datasets.revision_status IS
'final: intercensal/revised; current: most recent; superseded: replaced by newer version';

COMMENT ON COLUMN census_pep_datasets.uncertainty_level IS
'low: intercensal; moderate: postcensal from completed decades; high: recent postcensal';

COMMENT ON COLUMN census_pep_datasets.supersedes_dataset IS
'Reference to older dataset that this one replaces (e.g., intercensal supersedes postcensal)';

COMMENT ON COLUMN census_pep_county_migration.is_preferred_estimate IS
'For overlapping years (e.g., 2020 in both intercensal and postcensal), TRUE indicates the recommended value';

COMMENT ON COLUMN census_pep_county_migration.residual IS
'Unexplained population change not attributable to births, deaths, or migration. Used for data quality assessment.';

COMMENT ON COLUMN census_pep_timeseries_rules.rationale IS
'Explanation for why this dataset is preferred (e.g., "intercensal 2020 is census-aligned, prefer over postcensal 2020")';

-- =============================================================================
-- USAGE EXAMPLES
-- =============================================================================

-- Example 1: Get recommended time series for analysis
-- SELECT * FROM census_pep_county_migration_preferred
-- WHERE geoid = '38101'  -- Cass County
-- ORDER BY year;

-- Example 2: Check data quality by vintage
-- SELECT * FROM census_pep_dataset_summary
-- ORDER BY year_range_start;

-- Example 3: Find years with validation failures
-- SELECT year, dataset_id, percent_difference
-- FROM census_pep_validation
-- WHERE validation_type = 'hierarchical_consistency'
--   AND NOT passed
-- ORDER BY year;

-- Example 4: Compare estimates for overlapping years
-- SELECT year, dataset_id, COUNT(*) as num_counties, AVG(netmig) as avg_netmig
-- FROM census_pep_county_migration
-- WHERE year = 2020
-- GROUP BY year, dataset_id;

-- =============================================================================
-- SCHEMA VERSION
-- =============================================================================

-- Track schema version for migrations
CREATE TABLE IF NOT EXISTS census_pep_schema_version (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description TEXT
);

INSERT INTO census_pep_schema_version (version, description)
VALUES ('1.0.0', 'Initial schema: 5 metadata tables + 5 views + helper functions');

COMMENT ON TABLE census_pep_schema_version IS
'Track schema version for future migrations and compatibility checking';
