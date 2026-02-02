# Development Tracker

**Single source of truth for North Dakota Population Projections development state**

> **Note:** This is the canonical project status tracker. Historical session notes have been
> archived to [docs/archive/](./docs/archive/) for reference.

This project implements cohort component population projections for North Dakota state, counties, and places (2025-2045).

---

## Current Status Summary

| Metric | Status |
|--------|--------|
| **Overall Progress** | 100% complete |
| **Current Phase** | Production Complete |
| **Key Milestone** | All Scenarios Complete with Reports & Visualizations |
| **Blocking Issue** | None |

**What's Done:** Core projection engine (~1,600 lines), data processing pipeline (~4,500 lines), geographic module (~1,400 lines), output/reporting (~2,300 lines), pipeline scripts (~2,400 lines), 15 ADRs, documentation, comprehensive test suite (464 tests), complete data pipeline with validated data, **full state projection for all 53 counties (2025-2045)**, **all 3 scenarios (baseline, high_growth, low_growth)**, **exports to CSV/Excel**, **population pyramids and trend visualizations**, **summary statistics reports**.

**What's Missing:** Optional enhancements only (place-level projections, TIGER data integration).

---

## >>> NEXT TASK <<<

**When asked to "work on the next task", do this:**

### Task: Documentation Polish and Optional Enhancements

**Priority:** LOW
**Type:** Documentation & Polish

**Overview:**
Core projection system is complete. Remaining work is documentation polish and optional enhancements.

**Before Starting:**
1. Review: `docs/guides/` - Existing documentation guides
2. Check: `AGENTS.md` - Agent guidance document
3. Review: Test coverage with `pytest --cov`

**Potential Tasks:**
1. Write user guide for running complete system
2. Create production deployment guide
3. Document methodology with mathematical formulas
4. Complete TIGER geographic data loading (optional)
5. Implement geospatial shapefile export (optional)
6. Add place-level projections (optional)

**Success Criteria:**
- All documentation up to date
- All tests passing
- Pre-commit hooks passing

---

## Current Sprint Status

| Sprint | Phase | Status | Progress | Notes |
|--------|-------|--------|----------|-------|
| ADR-021 Phase B | Complete | **Done** | 100% | All 8 recommendations implemented |
| SPRINT_ANALYSIS_B_INFRASTRUCTURE | B | Complete | 100% | Refactored Analysis Scripts, Archived Legacy Code |
| SPRINT_ANALYSIS_B_001 | B | Complete | 100% | Deep Research (Manual), Revision v0.8.5 Polish |
| Repository Hygiene | - | Complete | 100% | Documentation guides added, AGENTS.md established |

## Recently Completed: ADR-021 Policy Integration (2026-01-02)

**ADR-021: Immigration Status Durability and Policy-Regime Methodology** - COMPLETE

All 8 recommendations from external AI analysis (ChatGPT 5.2 Pro) have been implemented:

| Wave | Recommendations | Key Outputs |
|------|-----------------|-------------|
| 1 | Rec #3, #4, #7, #8 | Regime framework, LSSND capacity (67.2%), secondary migration |
| 2 | Rec #2 | Status durability (parole hazard 11.29x) |
| 3 | Rec #6 | 5 policy-lever scenarios with Monte Carlo |
| 4 | Rec #1 | Two-component estimand (Y_t^dur + Y_t^temp) |

New modules in `sdc_2024_replication/scripts/statistical_analysis/`:
- `module_regime_framework.py`, `module_7b_lssnd_synthetic_control.py`
- `module_8b_status_durability.py`, `module_9b_policy_scenarios.py`
- `module_10_two_component_estimand.py`, `module_secondary_migration.py`

See: [ADR-021](docs/governance/adrs/021-immigration-status-durability-methodology.md) | [Phase B Tracker](docs/governance/adrs/021-reports/PHASE_B_IMPLEMENTATION_TRACKER.md)

## Active Phase

**Phase B**: Correction Methods Implementation
- See: `docs/governance/adrs/020-reports/PHASE_B/PHASE_METADATA.md`
- Decision: Option C (Hybrid approach) for extended time series
- Planning complete (Agents B0a-B6), implementation in progress

## Completed Phases

- **Refactoring & Hygiene** (2026-01-04) - COMPLETE
  - Consolidated analysis modules to use `data_loader.py` (SQL-based).
  - Archived 9 legacy scripts (ARIMA, etc.) to `_archive/`.
  - Confirmed "Deep Research" as a manual workflow (docs at `docs/research/`).

- **ADR-021 Phase B**: Immigration Status Durability (2026-01-02) - COMPLETE
  - See: `docs/governance/adrs/021-reports/PHASE_B_IMPLEMENTATION_TRACKER.md`
  - All 8 recommendations implemented across 4 waves
- **Phase A**: Validity Risk Assessment (2025-12-31)
  - See: `docs/governance/adrs/020-reports/PHASE_A/PHASE_METADATA.md`
  - External Review: ChatGPT 5.2 Pro confirmed Option C recommendation

---

## Active Tasks

Tasks for current phase. States: `[ ]` pending | `[x]` complete

### Full State Projection (Priority 1) - COMPLETE

- [x] Run projections for all 53 North Dakota counties
- [x] Generate baseline scenario (2025-2045)
- [x] Generate high/low growth scenarios
- [x] Export results to all formats (Excel, CSV, Parquet)
- [x] Generate summary reports and visualizations
- [x] Validate county totals sum to state

### Documentation Polish (Priority 2)

- [ ] Write user guide for running complete system
- [ ] Create production deployment guide
- [ ] Document methodology with mathematical formulas
- [x] Add troubleshooting guide (docs/guides/troubleshooting.md)
- [x] Add data sources workflow guide (docs/guides/data-sources-workflow.md)
- [x] Add geographic hierarchy reference (docs/reference/geographic-hierarchy.md)

### v0.8.6 Critique Implementation

- [ ] Data extension and provenance updates (see `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`)
  - In progress: ACS B07007 moved-from-abroad proxy processed (2010-2023), LPR state totals now cover FY2000-2023 (FY2012 state totals extracted from yearbook PDF), PEP regime markers added, DHS yearbook raw downloads organized with manifests updated; OCR extraction improved FY2022/FY2024 refugee coverage (FY2022 ND-only placeholder removed; missing states left omitted; post-2020 state-panel analyses now drop missing states per ADR-025).
- [x] Pre-2007 DHS LPR yearbook downloads (manual browser; CDN blocks automated fetch)
- Status update memo (2026-01-04): `sdc_2024_replication/revisions/v0.8.6/status_update_2026-01-04.md`
- Archived full tracker: `sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md`

### Optional Enhancements (Priority 3)

- [ ] Complete TIGER geographic data loading (falls back to defaults)
- [ ] Implement geospatial shapefile export (`output/writers.py:684`)
- [ ] Add place-level projections (cities/towns)

---

## Completed Tasks

### 2025-12-28: Data Pipeline Complete

- [x] **Wave 1: Data Fetch**
  - Fetched local data from sibling repos (popest, ndx-econareas, maps)
  - Downloaded CDC life tables (2022, 2023) for mortality rates
  - Downloaded CDC fertility data (NVSR, quarterly rates)
  - Downloaded IRS migration data (2018-2022 county flows)

- [x] **Wave 2: Data Processing**
  - Processed fertility data → `asfr_processed.csv` (42 rows, 7 age groups x 6 races)
  - Processed mortality data → `survival_rates_processed.csv` (1,212 rows)
  - Processed migration data → `nd_migration_processed.csv` (212 rows, 53 counties x 4 years)
  - Processed population data → `nd_county_population.csv` (53 counties) + `nd_age_sex_race_distribution.csv`

- [x] **Wave 3: Data Validation**
  - Created validation script and report
  - All 34 validation checks passed

- [x] **Wave 4: Test Suite**
  - Created 464 unit tests across all modules
  - Tests for: core/, data/, output/, geographic/
  - All tests passing

- [x] **Wave 5: Integration Test**
  - End-to-end projection working
  - Cass County 5-year projection (2025-2030) successful
  - Results saved to `data/processed/integration_test_results.csv`

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
| None | - | Data pipeline complete |

### Data Source Status

| Data Component | Available? | Location |
|----------------|------------|----------|
| Base Population (State) | Yes | `data/processed/nd_county_population.csv` |
| Base Population (County) | Yes | `data/processed/nd_county_population.csv` |
| Base Population (Age-Sex-Race) | Yes | `data/processed/nd_age_sex_race_distribution.csv` |
| Geographic Reference (Counties) | Yes | `data/processed/` |
| **Fertility Rates (ASFR)** | **Yes** | `data/processed/asfr_processed.csv` |
| **Survival Rates (Life Tables)** | **Yes** | `data/processed/survival_rates_processed.csv` |
| **Migration Rates** | **Yes** | `data/processed/nd_migration_processed.csv` |

---

## Session Log

### 2025-12-28 - Full State Projection Complete

**Duration:** Extended session
**Focus:** Running full state projections for all 53 counties

**Accomplishments:**

**Pipeline Fixes:**

- Fixed function name mismatch (`run_multiple_geography_projections` → `run_multi_geography_projections`)
- Fixed `load_nd_counties` argument handling (was receiving whole config dict)
- Disabled parallel processing due to pickle serialization issues with nested functions
- Fixed JSON metadata serialization with `default=str` parameter

**Data Format Transformations:**

- Created data transformation functions to convert processed CSV/Parquet to engine-expected format
- Fertility: `asfr` → `fertility_rate`, age groups (15-19) → single-year ages, race code mapping
- Survival: race code to full name mapping, sex capitalization
- Migration: Created cohort-level rates from county-level totals

**Full State Projection Run:**

### 2025-12-31 - Citation Audit Hardening

**Focus:** Improve citation audit robustness and agent workflows

**Accomplishments:**
- Added include-following TeX discovery, multiline citation handling, and nocite-aware key matching
- Expanded BibTeX parsing with @string macro expansion and duplicate key detection
- Added optional JSONL fixes input for agent-provided corrections
- Added citation audit unit tests and updated documentation

- Successfully ran baseline projections for all 53 North Dakota counties
- 20-year horizon: 2025-2045
- Base population: 796,568 (2025)
- Projected population: 754,882 (2045)
- Change: -41,686 (-5.2%)
- 57,876 cohorts per year (53 counties × 1,092 age-sex-race combinations)

**Files Created/Modified:**

- `scripts/pipeline/02_run_projections.py` - Added data transformation functions
- `cohort_projections/geographic/multi_geography.py` - Fixed JSON serialization
- `config/projection_config.yaml` - Disabled parallel processing
- `data/projections/baseline/county/*.parquet` - 53 county projection files

**Output:**

- 53 Parquet files with detailed cohort projections
- Each county has 21 years × 1,092 cohorts = 22,932 rows

---

### 2025-12-28 - Complete Data Pipeline Implementation

**Duration:** Full day session
**Focus:** Data acquisition, processing, validation, testing, and integration

**Accomplishments:**

**Wave 1: Data Fetch**
- Fetched local data from sibling repositories (popest, ndx-econareas, maps)
- Downloaded CDC life tables (2022, 2023) for mortality rates
- Downloaded CDC fertility data (NVSR, quarterly rates)
- Downloaded IRS migration data (2018-2022 county flows)

**Wave 2: Data Processing**
- Processed fertility data → `asfr_processed.csv` (42 rows, 7 age groups x 6 races)
- Processed mortality data → `survival_rates_processed.csv` (1,212 rows)
- Processed migration data → `nd_migration_processed.csv` (212 rows, 53 counties x 4 years)
- Processed population data → `nd_county_population.csv` (53 counties) + `nd_age_sex_race_distribution.csv`

**Wave 3: Data Validation**
- Created comprehensive validation script
- Generated validation report
- All 34 validation checks passed

**Wave 4: Test Suite**
- Created 464 unit tests across all modules
- Tests cover: core/, data/, output/, geographic/
- All tests passing

**Wave 5: Integration Test**
- Ran end-to-end projection successfully
- Cass County 5-year projection (2025-2030) completed
- Results saved to `data/processed/integration_test_results.csv`

**Files Created/Modified:**
- `data/processed/asfr_processed.csv`
- `data/processed/survival_rates_processed.csv`
- `data/processed/nd_migration_processed.csv`
- `data/processed/nd_county_population.csv`
- `data/processed/nd_age_sex_race_distribution.csv`
- `data/processed/integration_test_results.csv`
- `tests/` - 464 new unit tests

**Next:**
- Run full state projection for all 53 counties
- Generate multiple scenarios (baseline, high, low growth)
- Create final reports and visualizations

---

## Pipeline Overview

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Infrastructure | Done | Project setup, config, ADRs |
| 2. Data Pipeline | Done | Fetch, validate, process data |
| 3. Projections | Done | Run projections for all geographies |
| 4. Output | Pending | Generate reports and exports |
| 5. Validation | Pending | Compare with external benchmarks |

### Phase Details

**Phase 1: Infrastructure** - COMPLETE
- Repository structure and configuration
- Database schema design
- ADR documentation (15 records)
- rclone bisync for data sync

**Phase 2: Data Pipeline** - COMPLETE (100%)
- Data fetching from local repos and CDC/IRS downloads
- Processing modules validated with real data
- All 53 counties have population data
- Fertility, mortality, and migration rates processed
- 34 validation checks passing
- 464 unit tests passing
- Integration test successful (Cass County)

**Phase 3: Projections** - COMPLETE (100%)

- Engine complete and validated
- Integration test passed (Cass County 2025-2030)
- Full state run complete (all 53 counties, 2025-2045)
- Baseline scenario: 796,568 → 754,882 (-5.2%)

**Phase 4: Output** - READY
- Writers complete (Excel, CSV, Parquet)
- Visualization modules ready
- Report templates defined

**Phase 5: Validation** - NOT STARTED
- Validation module ready
- Need comparison benchmarks (Census Bureau projections)
- Quality checks designed

---

## Quick Reference

### Key Commands

```bash
# Environment setup
micromamba activate cohort_proj

# Run tests
pytest tests/ -v

# Run integration test
python scripts/run_integration_test.py

# Run full projection pipeline
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
| `tests/` | Test suite (464 tests) |

### Key Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview |
| `docs/governance/adrs/016-raw-data-management-strategy.md` | Data management approach |
| `docs/DATA_ACQUISITION.md` | External data sources |
| `docs/REPOSITORY_EVALUATION.md` | Status assessment |

---

### Session Log

#### 2025-12-30 - Journal Article Revision Work
- Updated causal inference and scenario arithmetic outputs for `sdc_2024_replication` journal article.
- Added pipeline diagram and module classification table; regenerated publication figures and captions.
- Refreshed summary text and intervals to align with updated Monte Carlo outputs.
- Integrated figure captions into the main build and smoothed LaTeX layout issues in data tables and appendix wording.
- Completed a full LaTeX build after layout fixes.

#### 2025-12-31 - Claim Review Workspace (v5_p305_complete)
- Extracted page-anchored text for the claim review PDF into `sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/extracted/`.
- Generated a draft `claims_manifest.jsonl` (1332 claims) via ad-hoc parsing; recorded cleanup issues and next steps in `sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/STATUS.md`.
- Added a reusable Introduction-claim parser (`sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/claims/build_intro_claims.py`) and refreshed Introduction claims with corrected page anchors in `claims_manifest.jsonl`.
- Logged parsing rules and pitfalls for the Introduction pass in `sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/STATUS.md`.
- Added a QA layer (`sdc_2024_replication/scripts/statistical_analysis/journal_article/claim_review/v3_phase3/claims/qa_claims.py`) and documented usage in the claim review README and status notes.
- Pending: refine claim splitting, remove residual plot-label artifacts, and reclassify claim types where needed.

#### 2026-01-04 - Finalized Revision v0.8.5
- **Documentation**: Harmonized Git/Rclone documentation (`docs/GIT_RCLONE_SYNC.md`).
- **Triage Fixes**: Corrected DiD interpretation and ITS framing in LaTeX source.
- **Pipeline**: Verified data integrity with `verify_pipeline.py`.
- **Output**: Compiled final PDF `sdc_2024_replication/revisions/v0.8.5/article_draft_v0.8.5.pdf`.

#### 2026-01-04 - v0.8.6 Critique Intake
- Created intake placeholder for the new ChatGPT 5.2 Pro critique to start the v0.8.6 revision cycle.
- Location: `sdc_2024_replication/revisions/v0.8.6/critique_chatgpt_5_2_pro_v0.8.5.md`.
- Created a progress tracker to split critique advice into discrete tasks for future agents.
- Location: `sdc_2024_replication/revisions/v0.8.6/progress_tracker_v0.8.6_critique_v0.8.5.md`.
- Created a response-to-critique stub for v0.8.6 decisions and approvals.
- Location: `sdc_2024_replication/revisions/v0.8.6/response_to_critique.md`.
- Drafted ADR-024 for immigration data extension and fusion strategy.
- Location: `docs/governance/adrs/024-immigration-data-extension-fusion.md`.

**Last Updated:** 2026-02-02
**Tracker Status:** Production complete, documentation polish in progress
