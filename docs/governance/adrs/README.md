# Architecture Decision Records (ADRs)

## Overview

This directory contains Architecture Decision Records (ADRs) documenting the key architectural and design decisions made during the development of the North Dakota Cohort Component Population Projection System.

## What are ADRs?

Architecture Decision Records are documents that capture important architectural decisions made along with their context and consequences. Each ADR describes:

- **Context**: The situation requiring a decision
- **Decision**: The choice that was made
- **Consequences**: The results of the decision (both positive and negative)
- **Alternatives**: Other options considered and why they were rejected

ADRs serve as a historical record, helping current and future developers understand **why** the system is built the way it is, not just **how** it works.

## Large Artifact Directories

ADR-020 and ADR-021 include large generated report bundles under:

- `docs/governance/adrs/020-reports/`
- `docs/governance/adrs/021-reports/`

These directories are intentionally excluded from git and synced externally.
Treat them as evidence/output stores rather than canonical source documents.

## How to Create a New ADR

### Step 1: Determine the Next ADR Number

Check existing ADRs and use the next sequential number:
```bash
ls docs/governance/adrs/[0-9][0-9][0-9]-*.md | sort | tail -1
```

### Step 2: Create the ADR File

Use the naming convention: `NNN-short-title.md`

Examples:
- `001-fertility-rate-processing.md`
- `016-raw-data-management-strategy.md`
- `017-api-design.md`

### Step 3: Copy the Template

Copy `TEMPLATE.md` to your new file:
```bash
cp docs/governance/adrs/TEMPLATE.md docs/governance/adrs/NNN-your-title.md
```

### Step 4: Fill in the Template

At minimum, complete these sections:
- **Title**: Use format `ADR-NNN: Descriptive Title`
- **Status**: Start with `Proposed`, change to `Accepted` when approved
- **Date**: Use `YYYY-MM-DD` format
- **Context**: Explain what problem requires a decision
- **Decision**: Document what was decided and why
- **Consequences**: List positive and negative outcomes

### Step 5: Update This README

Add an entry to the ADR Index below with:
- Title and status
- One-line summary
- Link to the file

### Naming Convention

All ADRs must follow this naming pattern:

```
NNN-short-descriptive-title.md
```

Where:
- `NNN` = Three-digit zero-padded number (001, 002, ..., 016, 017)
- `short-descriptive-title` = Lowercase words separated by hyphens
- `.md` = Markdown file extension

**Good examples:**
- `017-api-design.md`
- `018-caching-strategy.md`
- `019-error-recovery-mechanism.md`

**Avoid:**
- `17-api-design.md` (missing zero padding)
- `ADR-017-api-design.md` (don't include ADR prefix in filename)
- `017_api_design.md` (use hyphens, not underscores)
- `017-API-Design.md` (use lowercase)

## ADR Statuses

| Status | Description |
|--------|-------------|
| **Proposed** | Decision documented but not yet approved |
| **Accepted** | Decision has been made and implemented |
| **Deprecated** | Decision no longer applicable |
| **Superseded** | Decision replaced by a later ADR (note which one) |

---

## ADR Index

### Data Processing

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [001](001-fertility-rate-processing.md) | Fertility Rate Processing | Accepted | 2025-12-18 | 5-year weighted averaging, zero-fill missing data, SEER race mapping |
| [002](002-survival-rate-processing.md) | Survival Rate Processing | Accepted | 2025-12-18 | Multi-method life table conversion, Lee-Carter mortality improvement |
| [003](003-migration-rate-processing.md) | Migration Rate Processing | Accepted | 2025-12-18 | IRS flow processing, ACS integration, age-sex-race distribution |

### System Architecture

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [004](004-core-projection-engine-architecture.md) | Core Projection Engine | Accepted | 2025-12-18 | Modular cohort-component method, vectorized pandas operations |
| [005](005-configuration-management-strategy.md) | Configuration Management | Accepted | 2025-12-18 | Centralized YAML config, ConfigLoader class, sensible defaults |
| [006](006-data-pipeline-architecture.md) | Data Pipeline Architecture | Accepted | 2025-12-18 | Two-stage Fetch/Process pipeline, Parquet primary storage |

### Demographic Methodology

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [007](007-race-ethnicity-categorization.md) | Race/Ethnicity Categorization | Accepted | 2025-12-18 | 6-category system, Hispanic as single category, explicit mappings |
| [010](010-geographic-scope-granularity.md) | Geographic Scope | Accepted | 2025-12-18 | State/County/Place levels, FIPS identifiers, age 90+ open-ended |

### Technical Infrastructure

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [008](008-bigquery-integration-design.md) | BigQuery Integration | Accepted | 2025-12-18 | Supplementary data source, service account auth, graceful degradation |
| [009](009-logging-error-handling-strategy.md) | Logging and Error Handling | Accepted | 2025-12-18 | Python standard logging, hierarchical levels, defensive programming |
| [011](011-testing-strategy.md) | Testing Strategy | Accepted | 2025-12-18 | Pragmatic approach, built-in validation, example scripts as tests |
| [012](012-output-export-format-strategy.md) | Output and Export Formats | Accepted | 2025-12-18 | Parquet + CSV dual format, JSON metadata, optional Excel export |
| [013](013-multi-geography-projection-design.md) | Multi-Geography Projections | Accepted | 2025-12-18 | Design for running projections across multiple geographic levels |
| [014](014-pipeline-orchestration-design.md) | Pipeline Orchestration | Accepted | 2025-12-18 | Complete pipeline orchestration design |
| [015](015-output-format-visualization-design.md) | Output Visualization | Accepted | 2025-12-18 | Output formats and visualization capabilities |

### Data Management

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [016](016-raw-data-management-strategy.md) | Raw Data Management | Accepted | 2025-12-28 | Hybrid git/rclone strategy, data manifest, fetch script |
| [034](034-census-pep-data-archive.md) | Census PEP Data Archive | Accepted | 2026-02-02 | Shared data archive for Census PEP across 8 vintages; parquet conversion and PostgreSQL analytics |

### Scenario and Replication

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [017](017-sdc-2024-methodology-comparison.md) | SDC 2024 Methodology Comparison and Scenario | Accepted | 2025-12-28 | Compare SDC methods to baseline and document divergence drivers |
| [018](018-immigration-policy-scenario-methodology.md) | Immigration Policy Scenario Methodology | Proposed | 2025-12-28 | Empirical adjustment method for policy-driven migration scenarios |

### Review Process

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [019](019-argument-mapping-claim-review-process.md) | Claim and Argument Mapping Review Process | Accepted | 2025-12-31 | Structured claim inventory and Toulmin-based argument mapping with graphs |

### Methodology Analysis

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [020](020-extended-time-series-methodology-analysis.md) | Extended Time Series Methodology Analysis | Accepted | 2026-01-01 | Option C (Hybrid) for 2000-2024 vintage series; primary inference on 2010-2024 |
| [020a](020a-vintage-methodology-investigation-plan.md) | Vintage Methodology Investigation Plan | Accepted | 2026-01-01 | Sub-agent investigation plan for PEP vintage methodology transitions |
| [021](021-immigration-status-durability-methodology.md) | Immigration Status Durability and Policy-Regime Methodology | Accepted | 2026-01-02 | Parole cohort durability, two-component estimand, policy-lever scenarios |
| [022](022-unified-documentation-strategy.md) | Unified Documentation and Reproducibility Strategy | Proposed | 2026-01-01 | Auto-generated documentation index backed by PostgreSQL metadata |
| [024](024-immigration-data-extension-fusion.md) | Immigration Data Extension and Fusion Strategy | Proposed | 2026-01-04 | Extend refugee/LPR series, align time bases, and incorporate regime-aware modeling |
| [025](025-refugee-coverage-missing-state-handling.md) | Post-2020 Refugee Coverage and Missing-State Handling | Accepted | 2026-01-04 | Missing states left unknown; drop missing post-2020 in state panels; official national totals |
| [026](026-amerasian-siv-handling-forecasting.md) | Amerasian/SIV Handling in Status Decomposition and Scenario Forecasts | Accepted | 2026-01-06 | Keep USRAP exposure strict; treat SIV as separate durable series linked to capacity with default sunset |
| [027](027-travel-ban-extended-dynamics-supplement.md) | Supplemental Travel Ban Regime-Dynamics Extension | Accepted | 2026-01-06 | Keep pre-COVID Travel Ban DiD primary; add FY2024 regime-dynamics supplement; exclude pseudo-nationalities |
| [028](028-monte-carlo-simulation-rigor-parallelization.md) | Monte Carlo Simulation Rigor and Parallelization | Accepted | 2026-01-06 | Increase draw counts, deterministic multi-process chunking, reproducible parallel MC |
| [029](029-wave-duration-refit-right-censoring-hazard.md) | Wave Duration Refit for Right-Censoring and Hazard Stability | Accepted | 2026-01-07 | Fixed baseline + year completion; tagged specs; Module 9 duration-tag integration |
| [030](030-pep-regime-aware-modeling-long-run-series.md) | Regime-Aware Modeling for Long-Run PEP Net International Migration Series | Accepted | 2026-01-07 | Use regime dummies + interventions (COVID) for long-run PEP; defer full state-space fusion |
| [031](031-covariate-conditioned-near-term-forecast-anchor.md) | Covariate-Conditioned Near-Term Forecast Anchor (Appendix-Only) | Accepted | 2026-01-07 | Keep Moderate baseline; add appendix-only covariate-conditioned near-term diagnostic |
| [032](032-uncertainty-envelopes-two-band-approach.md) | Two-Band Uncertainty After Fusion (Avoid Double-Counting Variance) | Accepted | 2026-01-07 | Report baseline-only PI plus wave-adjusted conservative envelope (nested bands) |

### Package Extraction

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [023](023-package-extraction-strategy.md) | Package Extraction Strategy | Accepted | 2026-01-02 | Strategy for extracting reusable features into standalone packages |
| [023a](023a-evidence-review-package.md) | Evidence Review Package | Accepted | 2026-01-02 | Citation audit, claims analysis, and Toulmin argument mapping package |
| [023b](023b-project-utils-package.md) | Project Utils Package | Accepted | 2026-01-02 | Configuration loading and logging setup utilities |
| [023c](023c-codebase-catalog-package.md) | Codebase Catalog Package | Accepted | 2026-01-02 | Codebase scanning, code inventory, and pre-commit hooks |

### Projection Methodology

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [033](033-city-level-projection-methodology.md) | City-Level Projection Methodology | Accepted | 2026-02-02 | Implemented share-of-county place projections with B-II winner selection and county-constrained QA validation |
| [057](057-rolling-origin-backtests.md) | Rolling-Origin Backtests | Proposed | 2026-03-01 | Rolling-origin cross-validation for place projection variant selection |
| [058](058-multicounty-place-splitting.md) | Multi-County Place Splitting | Proposed | 2026-03-01 | Allocate multi-county place population to constituent counties for projection |
| [060](060-housing-unit-method.md) | Housing-Unit Method | Proposed | 2026-03-01 | Complementary short-term place projections using housing units × persons-per-household |

### Migration and Scenario Methodology

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [035](035-migration-data-source-census-pep.md) | Census PEP Components of Change for Migration Inputs | Accepted | 2026-02-03 | Replace IRS flows with Census PEP components; addresses 74K-80K projection divergence |
| [036](036-migration-averaging-methodology.md) | Migration Averaging Methodology | Proposed | 2026-02-12 | Multi-period and interpolation approaches for PEP migration averaging |
| [037](037-cbo-grounded-scenario-methodology.md) | CBO-Grounded Scenario Methodology | Accepted | 2026-02-17 | CBO time-varying migration factor replaces arbitrary scenario multipliers |
| [039](039-international-only-migration-factor.md) | International-Only Migration Factor | Accepted | 2026-02-17 | CBO migration factor applies to international migration only (intl_share decomposition) |
| [040](040-extend-boom-dampening-2015-2020.md) | Extend Bakken Boom Dampening to 2015-2020 | Accepted | 2026-02-17 | Adds 2015-2020 to boom dampening periods for oil-impacted counties |
| [041](041-census-pums-hybrid-base-population.md) | Census+PUMS Hybrid Base Population | Superseded by ADR-044 | 2026-02-17 | Census age-sex + PUMS race allocation; replaced by full-count Census data |
| [044](044-census-full-count-race-distribution.md) | Census Full-Count Race Distribution | Accepted | 2026-02-18 | Replace PUMS race allocation with Census cc-est2024-alldata full-count estimates; fixes zero Black females and Hispanic concentration |
| [047](047-county-specific-age-sex-race-distributions.md) | County-Specific Age-Sex-Race Distributions | Accepted | 2026-02-18 | Replace statewide distribution with county-specific distributions from cc-est2024-alldata; fixes 18-76% misallocation |
| [048](048-single-year-of-age-base-population.md) | Single-Year-of-Age Base Population | Accepted | 2026-02-18 | Use Census SC-EST single-year data instead of uniform 5-year splitting; eliminates step-function artifacts |
| [045](045-reservation-county-pep-recalibration.md) | Reservation County PEP-Anchored Migration Recalibration | Accepted | 2026-02-18 | Hybrid PEP scaling + Rogers-Castro fallback for reservation counties where residual method overestimates out-migration 2-3x |
| [046](046-high-growth-bebr-convergence.md) | High Growth Scenario via BEBR Convergence Rates | Accepted | 2026-02-18 | Replace broken multiplicative +15% with BEBR-derived additive convergence rates; guarantees high > baseline for all counties |
| [050](050-restricted-growth-additive-migration-adjustment.md) | Restricted Growth Additive Migration Adjustment | Accepted | 2026-02-18 | Replace multiplicative CBO factor with additive per-capita reduction; fixes restricted > baseline ordering violation for 39 counties |
| [051](051-oil-county-dampening-recalibration.md) | Oil County Dampening Recalibration | Rejected | 2026-02-18 | Rejected after calibration review found current dampening was adequate for target behavior. |
| [052](052-ward-county-high-growth-floor.md) | Ward County High-Growth Scenario Floor | Accepted | 2026-02-18 | High-growth migration floor at zero for counties with negative BEBR rates; prevents Ward from declining in all scenarios |

### Export Format

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [038](038-multi-workbook-export-format.md) | Multi-Workbook Export Format | Accepted | 2026-02-17 | Multi-workbook Excel export with state, region, and county sheets |
| [059](059-tiger-geospatial-exports.md) | TIGER Geospatial Exports | Proposed | 2026-03-01 | GeoJSON/Shapefile export using Census TIGER boundary files |

### Presentation and Quality Assurance

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [042](042-baseline-projection-presentation-requirements.md) | Baseline Projection Presentation Requirements | Accepted | 2026-02-18 | Mandatory caveats and pairing with restricted_growth; baseline is trend-continuation, not forecast |
| [043](043-migration-rate-cap.md) | Age-Aware Migration Rate Cap | Accepted | 2026-02-18 | Age-aware asymmetric cap (+/-15% for ages 15-24, +/-8% for others) clips statistical noise in convergence rates |
| [049](049-college-age-smoothing-convergence-pipeline.md) | College-Age Smoothing in Convergence Pipeline | Accepted | 2026-02-18 | Fix bug: propagate college-age migration smoothing to convergence pipeline input; fixes Cass +63% → ~+48% |

### Vital Rates Calibration

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [053](053-nd-specific-vital-rates.md) | ND-Specific Fertility and Mortality Rates | Accepted | 2026-02-20 | Replace national ASFR/survival rates with ND-adjusted rates via CDC WONDER and NVSR 74-12 state life tables; fixes 14-20% fertility undercount |

### Aggregation and Population Structure

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [054](054-state-county-aggregation-reconciliation.md) | State-County Aggregation Reconciliation | Accepted | 2026-02-23 | Align state totals with county rollups and close long-horizon divergence between independent runs. |
| [055](055-group-quarters-separation.md) | Group Quarters Population Separation | Accepted | 2026-02-23 | Separate institutional and household populations to prevent GQ turnover from distorting migration dynamics. |

---

## ADR Status Summary

| Status | Count |
|--------|-------|
| Accepted | 53 |
| Proposed | 8 |
| Rejected | 1 |
| Deprecated | 0 |
| Superseded | 1 |

**Naming Convention Compliance**: All 59 ADRs follow the `NNN-short-title.md` naming convention (including child ADRs with letter suffixes like `020a`, `023a`).

---

## Quick Reference

### For New Team Members

Start with these ADRs to understand the system:
1. **ADR-004**: Core Projection Engine Architecture (system overview)
2. **ADR-007**: Race and Ethnicity Categorization (demographic structure)
3. **ADR-010**: Geographic Scope (what geographies are projected)
4. **ADR-006**: Data Pipeline Architecture (how data flows)

### By Topic

**Demographic Methodology**: ADR-001, ADR-002, ADR-003, ADR-007, ADR-053

**System Design**: ADR-004, ADR-005, ADR-006, ADR-008

**Quality and Operations**: ADR-009, ADR-011, ADR-012

**Data Management**: ADR-016, ADR-034

**Migration Methodology**: ADR-003, ADR-035, ADR-036, ADR-039, ADR-040, ADR-043, ADR-045, ADR-049, ADR-050, ADR-051

**Scenarios**: ADR-017, ADR-018, ADR-037, ADR-042, ADR-046, ADR-050, ADR-052

**Base Population**: ADR-041, ADR-044, ADR-047, ADR-048

---

## Decision Principles

Across all ADRs, these principles guided decisions:

1. **Demographic Correctness**: Follow established demographic methodology
2. **Data Standards**: Use Census Bureau and SEER conventions
3. **Transparency**: Document all assumptions and transformations
4. **Reproducibility**: Ensure projections can be recreated
5. **Usability**: Balance technical sophistication with accessibility
6. **Performance**: Efficient for state/county/place-level projections
7. **Maintainability**: Code should be understandable and modifiable
8. **Flexibility**: Support multiple scenarios and configurations

---

## Resources

### ADR Methodology
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) - Michael Nygard
- [ADR GitHub Organization](https://adr.github.io/)

### Demographic Methodology
- Preston, Heuveline, Guillot: "Demography: Measuring and Modeling Population Processes" (2001)
- Smith, Tayman, Swanson: "State and Local Population Projections" (2001)
- U.S. Census Bureau Methodology Documentation

### Data Sources
- SEER: https://seer.cancer.gov/popdata/
- Census Bureau: https://www.census.gov/programs-surveys/popproj.html
- CDC NVSS: https://www.cdc.gov/nchs/nvss/index.htm

---

**Last Updated**: 2026-03-01

**Total ADRs**: 63 (53 accepted, 8 proposed, 1 rejected, 1 superseded)

**Template**: See [TEMPLATE.md](TEMPLATE.md) for creating new ADRs
