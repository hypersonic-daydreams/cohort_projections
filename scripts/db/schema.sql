-- Data Manifest PostgreSQL Schema
-- Database: cohort_projections_meta
--
-- This schema stores metadata about all data sources used in the
-- cohort_projections project, including temporal alignment information.
--
-- Usage:
--   createdb cohort_projections_meta
--   psql -d cohort_projections_meta -f scripts/db/schema.sql

-- =============================================================================
-- ENUMS
-- =============================================================================

-- Temporal basis for data sources
CREATE TYPE temporal_basis AS ENUM (
    'fiscal_year',      -- Oct 1 - Sep 30 (federal FY)
    'calendar_year',    -- Jan 1 - Dec 31
    'tax_year',         -- Same as CY but reflects prior year income
    'rolling_5year',    -- 5-year rolling average (e.g., ACS)
    'point_in_time',    -- Single snapshot (e.g., Census geography)
    'intercensal'       -- Census intercensal periods (e.g., 2000-2005)
);

-- Data format types
CREATE TYPE data_format AS ENUM (
    'csv',
    'parquet',
    'excel_xlsx',
    'excel_xls',
    'stata_dta',
    'json',
    'pdf',
    'spss_sav',
    'r_rds',
    'other'
);

-- Data source categories
CREATE TYPE source_category AS ENUM (
    'refugee_immigration',
    'census_population',
    'vital_statistics',
    'migration',
    'geographic',
    'projections_source',
    'other'
);

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Main data sources table
CREATE TABLE data_sources (
    id SERIAL PRIMARY KEY,

    -- Identification
    name VARCHAR(200) NOT NULL UNIQUE,
    short_name VARCHAR(50),  -- For display/referencing
    description TEXT,
    category source_category NOT NULL,

    -- Source information
    source_organization VARCHAR(200) NOT NULL,
    source_url TEXT,

    -- Format and location
    format data_format NOT NULL,
    location VARCHAR(500) NOT NULL,  -- Relative path from project root

    -- Temporal information (critical for FY/CY alignment)
    temporal_basis temporal_basis NOT NULL,
    years_available VARCHAR(100),  -- e.g., "2010-2024" or "FY2002-FY2024"
    reference_date VARCHAR(100),   -- e.g., "July 1" for Census PEP

    -- Alignment and processing
    alignment_notes TEXT,          -- How to handle FY/CY mismatch
    processing_script VARCHAR(500),  -- Script that processes this source

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT CURRENT_USER
);

-- Index for location lookups (used by pre-commit hook)
CREATE INDEX idx_data_sources_location ON data_sources(location);
CREATE INDEX idx_data_sources_category ON data_sources(category);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_data_sources_modtime
    BEFORE UPDATE ON data_sources
    FOR EACH ROW
    EXECUTE FUNCTION update_modified_column();

-- =============================================================================
-- CHANGELOG TABLE
-- =============================================================================

-- Track changes to the manifest (Type II SCD-like history)
CREATE TABLE manifest_changelog (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL,  -- Semantic version
    change_date DATE NOT NULL DEFAULT CURRENT_DATE,
    changes TEXT NOT NULL,
    changed_by VARCHAR(100) DEFAULT CURRENT_USER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- SUPPORTING TABLES
-- =============================================================================

-- Files within each data source (optional, for detailed tracking)
CREATE TABLE data_files (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES data_sources(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    description TEXT,
    year_or_period VARCHAR(50),  -- e.g., "FY2024" or "2019-2023"
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_id, filename)
);

CREATE INDEX idx_data_files_source ON data_files(source_id);

-- Temporal alignment rules between sources
CREATE TABLE temporal_alignments (
    id SERIAL PRIMARY KEY,
    source_a_id INTEGER REFERENCES data_sources(id),
    source_b_id INTEGER REFERENCES data_sources(id),
    alignment_issue TEXT NOT NULL,
    handling_strategy TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_a_id, source_b_id)
);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Summary view for quick lookups
CREATE VIEW data_sources_summary AS
SELECT
    id,
    name,
    short_name,
    category,
    source_organization,
    format,
    location,
    temporal_basis,
    years_available
FROM data_sources
ORDER BY category, name;

-- Temporal alignment matrix view
CREATE VIEW temporal_alignment_matrix AS
SELECT
    sa.name AS source_a,
    sb.name AS source_b,
    ta.alignment_issue,
    ta.handling_strategy
FROM temporal_alignments ta
JOIN data_sources sa ON ta.source_a_id = sa.id
JOIN data_sources sb ON ta.source_b_id = sb.id;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE data_sources IS 'Canonical metadata for all data sources in cohort_projections';
COMMENT ON COLUMN data_sources.temporal_basis IS 'Critical: FY vs CY alignment. See DATA_MANIFEST.md for handling strategies.';
COMMENT ON COLUMN data_sources.location IS 'Relative path from project root. Used by pre-commit hook for enforcement.';
COMMENT ON COLUMN data_sources.alignment_notes IS 'How to handle temporal misalignment with other sources';

COMMENT ON TABLE manifest_changelog IS 'History of manifest changes for audit trail';
COMMENT ON TABLE temporal_alignments IS 'Documents how to align sources with different temporal bases';
