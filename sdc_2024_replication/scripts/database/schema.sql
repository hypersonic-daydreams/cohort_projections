
-- Schema for Demography Data
-- Includes raw data tables and provenance metadata

DROP SCHEMA IF EXISTS meta CASCADE;
DROP SCHEMA IF EXISTS census CASCADE;
DROP SCHEMA IF EXISTS acs CASCADE;
DROP SCHEMA IF EXISTS dhs CASCADE;
DROP SCHEMA IF EXISTS rpc CASCADE;

-- 1. Metadata Schema
CREATE SCHEMA IF NOT EXISTS meta;

CREATE TABLE IF NOT EXISTS meta.source_files (
    source_id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT UNIQUE NOT NULL, -- SHA-256
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

-- 2. Census Schema (Population Estimates)
CREATE SCHEMA IF NOT EXISTS census;

CREATE TABLE IF NOT EXISTS census.population_estimates (
    id SERIAL PRIMARY KEY,
    source_file_id INTEGER REFERENCES meta.source_files(source_id),

    region VARCHAR(10),
    division VARCHAR(10),
    state_fips VARCHAR(2),
    county_fips VARCHAR(3),
    state_name TEXT,
    county_name TEXT,

    -- Estimates base
    estimates_base_2020 INTEGER,

    -- Pop Estimates
    pop_estimate_2020 INTEGER,
    pop_estimate_2021 INTEGER,
    pop_estimate_2022 INTEGER,
    pop_estimate_2023 INTEGER,
    pop_estimate_2024 INTEGER,

    -- Components of Change (Net Migration, etc.)
    net_mig_2020 INTEGER,
    net_mig_2021 INTEGER,
    net_mig_2022 INTEGER,
    net_mig_2023 INTEGER,
    net_mig_2024 INTEGER,

    international_mig_2020 INTEGER,
    international_mig_2021 INTEGER,
    international_mig_2022 INTEGER,
    international_mig_2023 INTEGER,
    international_mig_2024 INTEGER,

    domestic_mig_2020 INTEGER,
    domestic_mig_2021 INTEGER,
    domestic_mig_2022 INTEGER,
    domestic_mig_2023 INTEGER,
    domestic_mig_2024 INTEGER,

    -- Vital Stats
    births_2020 INTEGER,
    births_2021 INTEGER,
    births_2022 INTEGER,
    births_2023 INTEGER,
    births_2024 INTEGER,

    deaths_2020 INTEGER,
    deaths_2021 INTEGER,
    deaths_2022 INTEGER,
    deaths_2023 INTEGER,
    deaths_2024 INTEGER
);

-- 3. ACS Schema (Foreign Born)
CREATE SCHEMA IF NOT EXISTS acs;

CREATE TABLE IF NOT EXISTS acs.foreign_born (
    id SERIAL PRIMARY KEY,
    source_file_id INTEGER REFERENCES meta.source_files(source_id),

    calendar_year INTEGER,
    geo_id VARCHAR(20),
    state_name TEXT,
    state_fips VARCHAR(2),

    country_code VARCHAR(50),
    country_name TEXT,
    estimate INTEGER,
    margin_of_error INTEGER,

    UNIQUE(calendar_year, state_name, country_name)
);

CREATE TABLE IF NOT EXISTS census.state_components (
    id SERIAL PRIMARY KEY,
    source_file_id INTEGER REFERENCES meta.source_files(source_id),
    state_name TEXT,
    state_fips INTEGER,
    year INTEGER,
    population INTEGER,
    pop_change INTEGER,
    births INTEGER,
    deaths INTEGER,
    natural_change INTEGER,
    intl_migration INTEGER,
    domestic_migration INTEGER,
    net_migration INTEGER,
    UNIQUE(state_name, year)
);

-- 4. DHS Schema (LPR)
CREATE SCHEMA IF NOT EXISTS dhs;

CREATE TABLE IF NOT EXISTS dhs.lpr_arrivals (
    id SERIAL PRIMARY KEY,
    source_file_id INTEGER REFERENCES meta.source_files(source_id),

    fiscal_year INTEGER,
    state_name TEXT,
    county_name TEXT,
    country_of_birth TEXT,
    region_of_birth TEXT,
    lpr_count INTEGER -- or "<D>" for withheld
);

-- 5. RPC Schema (Refugee)
CREATE SCHEMA IF NOT EXISTS rpc;

CREATE TABLE IF NOT EXISTS rpc.refugee_arrivals (
    id SERIAL PRIMARY KEY,
    source_file_id INTEGER REFERENCES meta.source_files(source_id),

    fiscal_year INTEGER, -- RPC is typically FY
    destination_state TEXT,
    destination_city TEXT,
    nationality TEXT,
    arrivals INTEGER
);
