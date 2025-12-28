# Development Tracker

**Single source of truth for North Dakota Population Projections development state**

This project implements cohort component population projections for North Dakota state, counties, and places (2025-2045).

---

## Current Status Summary

| Metric | Status |
|--------|--------|
| **Overall Progress** | ~75% complete |
| **Current Phase** | Data Pipeline Setup |
| **Key Milestone** | Core projection engine complete |
| **Blocking Issue** | Data files not yet acquired |

**What's Done:** Core projection engine (~1,600 lines), data processing pipeline (~4,500 lines), geographic module (~1,400 lines), output/reporting (~2,300 lines), pipeline scripts (~2,400 lines), 15 ADRs, documentation.

**What's Missing:** Actual data files, comprehensive tests (~5% coverage), some geographic reference data, user documentation.

---

## >>> NEXT TASK <<<

**When asked to "work on the next task", do this:**

### Task: Fetch and Validate Data from Sibling Repositories

**Priority:** HIGH (blocking)
**Estimated Time:** 1-2 hours
**Type:** Data Pipeline

**Overview:**
The projection engine is complete but has no data to process. Most required data exists in sibling repositories on this machine. Fetch it, validate columns, and run an end-to-end test.

**Before Starting:**
1. Read: `docs/adr/016-raw-data-management-strategy.md` - Data management approach
2. Check: `scripts/fetch_data.py` - Data fetching script
3. Review: `docs/DATA_ACQUISITION.md` - External data sources

**Steps:**
1. Run `python scripts/fetch_data.py --list` to see available sources
2. Copy data from sibling repositories (popest, ndx-econareas, maps)
3. Download SEER fertility/mortality rates (manual - see blockers)
4. Validate all data files have required columns
5. Run end-to-end projection test

**Key Resources:**
- Population data: `/home/nigel/projects/popest/data/raw/`
- Geographic reference: `/home/nigel/projects/ndx-econareas/data/processed/reference/`
- PUMS microdata: `/home/nigel/maps/data/raw/pums_person.parquet`

---

## Active Tasks

Tasks for current phase. States: `[ ]` pending | `[x]` complete

### Data Acquisition (Priority 1 - BLOCKING)

- [ ] Fetch population data from popest repository
- [ ] Fetch geographic reference data from ndx-econareas
- [ ] Download SEER fertility/mortality rates (manual)
- [ ] Download IRS migration data
- [ ] Validate all data files have required columns
- [ ] Run end-to-end projection test

### Testing (Priority 2)

- [ ] Create core engine unit tests (~300-400 lines)
- [ ] Create data processor tests (~300-400 lines)
- [ ] Create geographic module tests (~200-300 lines)
- [ ] Create output module tests (~200-300 lines)
- [ ] Create integration/end-to-end tests (~300-400 lines)

### Incomplete Features (Priority 3)

- [ ] Complete TIGER geographic data loading (falls back to defaults)
- [ ] Implement geospatial shapefile export (`output/writers.py:684`)
- [ ] Build validation module (`cohort_projections/validation/`)

### Documentation (Priority 4)

- [ ] Write user guide for running complete system
- [ ] Create API reference documentation
- [ ] Document methodology with mathematical formulas
- [ ] Add troubleshooting guide

---

## Completed Tasks

### Infrastructure & Core Development

- [x] Core projection engine implementation (~1,600 lines)
  - `cohort_component.py` - Main orchestration
  - `fertility.py` - Age-specific fertility rates
  - `mortality.py` - Survival rates
  - `migration.py` - Net migration component

- [x] Data processing pipeline (~4,500 lines)
  - `base_population.py` - Base year processing
  - `fertility_rates.py` - SEER/NVSS processing
  - `survival_rates.py` - Life table conversion
  - `migration_rates.py` - IRS migration processing

- [x] Geographic module (~1,400 lines)
  - `geography_loader.py` - Reference data management
  - `multi_geography.py` - Multi-geography orchestration

- [x] Output & reporting (~2,300 lines)
  - `writers.py` - Multi-format export (Excel, CSV, Parquet)
  - `reports.py` - Summary statistics
  - `visualizations.py` - Charts and population pyramids

- [x] Pipeline scripts (~2,400 lines)
  - `01_process_demographic_data.py`
  - `02_run_projections.py`
  - `03_export_results.py`

### Project Setup

- [x] Data management infrastructure (ADR-016)
- [x] rclone bisync setup for multi-computer sync
- [x] Repository hygiene (pyproject.toml, pre-commit, CLAUDE.md)
- [x] 15 Architecture Decision Records
- [x] 7 working examples
- [x] Data acquisition guide

---

## Known Blockers

### Active Blockers

| Blocker | Impact | Workaround |
|---------|--------|------------|
| **SEER data requires manual download** | Cannot run projections without fertility/mortality rates | Download from seer.cancer.gov or cdc.gov |
| **CDC WONDER data requires account** | Alternative to SEER for vital statistics | Request account or use SEER |
| **IRS migration data may need preprocessing** | Migration component incomplete | Download from irs.gov/statistics/soi-tax-stats-migration-data |

### Data Source Status

| Data Component | Available Locally? | Source Location |
|----------------|-------------------|-----------------|
| Base Population (State) | Yes | popest |
| Base Population (County) | Yes | popest |
| Base Population (Place) | Yes | popest |
| Base Population (Age-Sex-Race) | Yes | maps/PUMS |
| Geographic Reference (Counties) | Yes | ndx-econareas |
| Geographic Reference (Places) | Yes | popest |
| **Fertility Rates (ASFR)** | **No** | Must download from SEER |
| **Survival Rates (Life Tables)** | **No** | Must download from CDC/SEER |
| **Migration Rates (Age-Specific)** | **No** | Must download from IRS SOI |

---

## Session Log

### Template

```markdown
### YYYY-MM-DD - Session Focus

**Duration:** ~X hours
**Focus:** What was worked on

**Accomplishments:**
- What was completed
- Files created/modified

**Next:**
- What to do next session
```

### Recent Sessions

*(Add session entries here as work progresses)*

---

## Pipeline Overview

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Infrastructure | Done | Project setup, config, ADRs |
| 2. Data Pipeline | Active | Fetch, validate, process data |
| 3. Projections | Pending | Run projections for all geographies |
| 4. Output | Pending | Generate reports and exports |
| 5. Validation | Pending | Compare with external benchmarks |

### Phase Details

**Phase 1: Infrastructure** - COMPLETE
- Repository structure and configuration
- Database schema design
- ADR documentation (15 records)
- rclone bisync for data sync

**Phase 2: Data Pipeline** - IN PROGRESS (~20%)
- Data fetching script exists but needs data
- Processing modules complete but untested with real data
- Geographic reference partially populated (10/53 counties)

**Phase 3: Projections** - NOT STARTED
- Engine complete, awaiting data
- Multi-geography orchestration ready
- Parallel processing configured

**Phase 4: Output** - NOT STARTED
- Writers complete (Excel, CSV, Parquet)
- Visualization modules ready
- Report templates defined

**Phase 5: Validation** - NOT STARTED
- Validation module empty
- Need comparison benchmarks (Census Bureau projections)
- Quality checks designed but not implemented

---

## Quick Reference

### Key Commands

```bash
# Environment setup
micromamba activate cohort_proj

# Check available data sources
python scripts/fetch_data.py --list

# Fetch data from sibling repos
python scripts/fetch_data.py

# Run full projection pipeline (once data is loaded)
python scripts/projections/run_all_projections.py

# Sync data between computers
./scripts/bisync.sh
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `cohort_projections/core/` | Projection engine |
| `cohort_projections/data/process/` | Data processors |
| `cohort_projections/geographic/` | Geography handling |
| `cohort_projections/output/` | Writers and reports |
| `data/raw/` | Input data files |
| `data/processed/` | Processed data |
| `data/output/` | Projection results |

### Key Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview |
| `docs/adr/016-raw-data-management-strategy.md` | Data management approach |
| `docs/DATA_ACQUISITION.md` | External data sources |
| `docs/REPOSITORY_EVALUATION.md` | Status assessment |

---

**Last Updated:** 2025-12-28
**Tracker Status:** Initialized - ready for data pipeline work
