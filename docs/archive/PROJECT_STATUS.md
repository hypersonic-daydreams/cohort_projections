# Cohort Projections Project - Current Status

## Executive Summary

**Previous Session:** The project was interrupted due to prompt length (exit code 1)

**Current State:** Core projection engine is complete. Analysis suite (`sdc_2024_replication`) has been refactored to use PostgreSQL (`data_loader.py`) and legacy scripts archived.

**Approach:** Use focused sub-agents to complete remaining components without overwhelming context

---

## ✅ COMPLETED Components

### 1. Core Projection Engine (100% Complete)
**Location:** `cohort_projections/core/`

- ✅ `cohort_component.py` (564 lines) - Main projection orchestration
- ✅ `fertility.py` (247 lines) - Fertility/births calculation
- ✅ `mortality.py` (354 lines) - Survival rates application
- ✅ `migration.py` (395 lines) - Net migration application
- ✅ `__init__.py` - Package exports

**Status:** Production-ready, fully documented, validated

### 2. Data Fetching (Partial - 33% Complete)
**Location:** `cohort_projections/data/fetch/`

- ✅ `census_api.py` - Census Bureau API integration (PEP, ACS)
- ❌ `vital_stats.py` - SEER/NVSS data fetching (NEEDED)
- ❌ `migration_data.py` - IRS/ACS migration data (NEEDED)
- ❌ `geographic.py` - Geographic reference data (NEEDED)

### 3. Data Processing (Partial - 25% Complete)
**Location:** `cohort_projections/data/process/`

- ✅ `base_population.py` - Base population processing
- ❌ `fertility_rates.py` - Process fertility rates from SEER/NVSS (NEEDED)
- ❌ `survival_rates.py` - Process life tables into survival rates (NEEDED)
- ❌ `migration_rates.py` - Process IRS/ACS migration data (NEEDED)

### 4. Utilities (100% Complete)
**Location:** `cohort_projections/utils/`

- ✅ `config_loader.py` - Configuration management
- ✅ `logger.py` - Logging utilities
- ✅ `demographic_utils.py` - Helper functions

### 5. Configuration (100% Complete)
**Location:** `config/`

- ✅ `projection_config.yaml` - Complete configuration file

### 6. Documentation (Partial - 60% Complete)
- ✅ Core module documentation (README, ARCHITECTURE, QUICKSTART)
- ✅ Implementation summary
- ✅ Data fetching/processing READMEs
- ❌ API documentation (NEEDED)
- ❌ User guide (NEEDED)
- ❌ Methodology documentation (NEEDED)

---

## ❌ NEEDED Components

### Priority 1: Data Pipeline (Critical Path)

#### A. Data Fetching Modules
**Estimated Effort:** 3 focused sub-agents

1. **`vital_stats.py`** - Fetch SEER/NVSS data
   - Age-specific fertility rates by race
   - Mortality rates / life tables by age/sex/race
   - ~300-400 lines

2. **`migration_data.py`** - Fetch migration data
   - IRS county-to-county flows
   - ACS mobility data
   - International migration estimates
   - ~350-450 lines

3. **`geographic.py`** - Geographic reference data
   - FIPS codes and names
   - County/place listings
   - Geographic hierarchy
   - ~200-300 lines

#### B. Data Processing Modules
**Estimated Effort:** 3 focused sub-agents

1. **`fertility_rates.py`** - Process fertility rates
   - Convert SEER data to age-specific rates
   - Average over period (2018-2022)
   - Format for projection engine
   - ~250-350 lines

2. **`survival_rates.py`** - Process survival rates
   - Convert life tables to survival rates
   - Handle open-ended age group (90+)
   - Apply mortality improvement trends
   - ~300-400 lines

3. **`migration_rates.py`** - Process migration rates
   - Process IRS county flows
   - Distribute to age/sex/race cohorts
   - Combine domestic + international
   - ~350-450 lines

### Priority 2: Geographic Module
**Estimated Effort:** 1-2 focused sub-agents

**Location:** `cohort_projections/geographic/`

- `multi_geography.py` - Run projections for multiple geographies
- `aggregation.py` - Aggregate from places → counties → state
- Parallel processing for counties/places
- ~400-500 lines total

### Priority 3: Output Module
**Estimated Effort:** 1-2 focused sub-agents

**Location:** `cohort_projections/output/`

1. **`writers/`**
   - `parquet_writer.py` - Write Parquet files
   - `csv_writer.py` - Write CSV files
   - `excel_writer.py` - Write Excel reports
   - ~200-300 lines total

2. **`reporting/`**
   - `summary_reports.py` - Generate summary statistics
   - `visualization.py` - Create charts/graphs
   - `comparison.py` - Compare scenarios
   - ~300-400 lines total

### Priority 4: Validation Module
**Estimated Effort:** 1 focused sub-agent

**Location:** `cohort_projections/validation/`

- `validators.py` - Plausibility checks
- `benchmarking.py` - Compare to Census benchmarks
- ~200-300 lines

### Priority 5: Executable Scripts
**Estimated Effort:** 2-3 focused sub-agents

**Location:** `scripts/`

1. **`data_pipeline/`**
   - `01_fetch_all_data.py` - Download all required data
   - `02_process_base_population.py` - Process base pop
   - `03_process_rates.py` - Process fertility/mortality/migration
   - `04_validate_inputs.py` - Validate all inputs
   - ~400-500 lines total

2. **`projections/`**
   - `run_state_projection.py` - State-level projection
   - `run_county_projections.py` - All counties
   - `run_place_projections.py` - All places
   - `run_all_projections.py` - Complete pipeline
   - ~300-400 lines total

3. **`export/`**
   - `export_results.py` - Export to various formats
   - `generate_reports.py` - Create summary reports
   - ~150-200 lines total

### Priority 6: Tests
**Estimated Effort:** 2-3 focused sub-agents

**Location:** `tests/`

- `test_core/` - Core engine tests (some may exist)
- `test_data/` - Data pipeline tests
- `test_output/` - Output module tests
- `test_geographic/` - Geographic module tests
- `test_integration/` - End-to-end tests
- ~800-1000 lines total

---

## Recommended Implementation Strategy

### Phase 1: Data Pipeline (Weeks 1-2)
**Goal:** Ability to fetch and process all required input data

**Sub-agents to launch:**
1. Vital stats fetcher (`vital_stats.py`)
2. Migration data fetcher (`migration_data.py`)
3. Geographic data fetcher (`geographic.py`)
4. Fertility rates processor (`fertility_rates.py`)
5. Survival rates processor (`survival_rates.py`)
6. Migration rates processor (`migration_rates.py`)

**Deliverable:** Complete data pipeline from raw data → projection inputs

### Phase 2: Pipeline Scripts (Week 3)
**Goal:** Automated data pipeline scripts

**Sub-agents to launch:**
1. Data fetching scripts
2. Data processing scripts

**Deliverable:** Run `python scripts/data_pipeline/run_pipeline.py` to get all inputs

### Phase 3: Geographic & Output Modules (Week 3-4)
**Goal:** Multi-geography projections and reporting

**Sub-agents to launch:**
1. Geographic module (multi-geography projections)
2. Output writers (parquet, CSV, Excel)
3. Reporting module (summaries, visualizations)

**Deliverable:** Run projections for all geographies, generate reports

### Phase 4: Projection Scripts (Week 4)
**Goal:** End-to-end projection pipeline

**Sub-agents to launch:**
1. Projection scripts (state, county, place)

**Deliverable:** `python scripts/projections/run_all_projections.py` works end-to-end

### Phase 5: Validation & Testing (Week 5)
**Goal:** Validate results, comprehensive tests

**Sub-agents to launch:**
1. Validation module
2. Test suite

**Deliverable:** Validated projections with comprehensive test coverage

---

## Focused Sub-Agent Approach

To avoid context overflow, each sub-agent should:

1. **Single responsibility** - One module or closely related set of files
2. **Clear inputs/outputs** - Specify expected data formats
3. **Minimal dependencies** - Focus on standalone components
4. **Use existing patterns** - Follow `census_api.py` and `base_population.py` patterns
5. **Include documentation** - Docstrings and usage examples
6. **Test locally** - Basic validation before moving on

### Example Sub-Agent Task Structure

```
Task: Implement fertility_rates.py processor

Context:
- Input: Raw SEER fertility data (DataFrame)
- Output: Age-specific fertility rates by race (DataFrame format specified in core/fertility.py)
- Pattern: Follow base_population.py structure
- Dependencies: pandas, numpy, utils.logger, utils.config_loader

Requirements:
- Read SEER CSV/text format
- Average over 5-year period (configurable)
- Map to 6 race categories
- Apply to ages 15-49
- Validate rate plausibility (0.001-0.13)
- Save to data/processed/fertility/
- Include validation function

Deliverables:
- fertility_rates.py (~300 lines)
- Unit test (basic)
- Update __init__.py exports
```

---

## Risk Mitigation

### Context Overflow Prevention
- ✅ Use sub-agents for each component
- ✅ Keep tasks focused (one module at a time)
- ✅ Minimize cross-references
- ✅ Document interfaces clearly
- ✅ Test components independently

### Data Availability
- ⚠️ SEER data may require manual download (web interface)
- ⚠️ IRS data may require pre-processing
- ⚠️ Some APIs may have rate limits
- ✅ Include sample/test data for development

### Integration Risks
- ✅ Core engine is complete and stable
- ✅ Data formats are well-documented
- ✅ Configuration system is in place
- ⚠️ Need integration tests to verify pipeline

---

## Next Immediate Steps

### Option A: Start with Data Fetching
Launch 3 sub-agents in parallel to implement:
1. `vital_stats.py` fetcher
2. `migration_data.py` fetcher
3. `geographic.py` fetcher

### Option B: Start with Data Processing
Launch 3 sub-agents in parallel to implement:
1. `fertility_rates.py` processor
2. `survival_rates.py` processor
3. `migration_rates.py` processor

### Option C: Complete One Full Pipeline First
Launch sequential sub-agents to complete fertility pipeline:
1. `vital_stats.py` fetcher (fertility data)
2. `fertility_rates.py` processor
3. Test with core engine
4. Repeat for mortality and migration

---

## Estimated Completion

**With focused sub-agent approach:**
- Phase 1 (Data Pipeline): 6 sub-agents × ~2-3 hours = 12-18 hours
- Phase 2 (Scripts): 2 sub-agents × ~2 hours = 4 hours
- Phase 3 (Geographic/Output): 3 sub-agents × ~2-3 hours = 6-9 hours
- Phase 4 (Projection Scripts): 1 sub-agent × ~2 hours = 2 hours
- Phase 5 (Validation/Tests): 2 sub-agents × ~3-4 hours = 6-8 hours

**Total: ~30-40 hours of development time**

**Calendar time:** 2-3 weeks with focused work sessions

---

## Questions for User

1. **Priority order:** Which phase would you like to start with?
2. **Parallel vs Sequential:** Should we launch multiple sub-agents in parallel or go one-by-one?
3. **Data availability:** Do you have access to SEER, NVSS, IRS data? Or should we use sample/mock data initially?
4. **Testing level:** How comprehensive should tests be? (smoke tests vs full coverage)
5. **Documentation depth:** Should we prioritize working code or comprehensive documentation?
