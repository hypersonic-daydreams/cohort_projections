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

## How to Create a New ADR

### Step 1: Determine the Next ADR Number

Check existing ADRs and use the next sequential number:
```bash
ls docs/adr/*.md | grep -E '^docs/adr/[0-9]{3}' | sort | tail -1
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
cp docs/adr/TEMPLATE.md docs/adr/NNN-your-title.md
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

### Scenario and Replication

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [017](017-sdc-2024-methodology-comparison.md) | SDC 2024 Methodology Comparison and Scenario | Accepted | 2025-12-28 | Compare SDC methods to baseline and document divergence drivers |
| [018](018-immigration-policy-scenario-methodology.md) | Immigration Policy Scenario Methodology | Proposed | 2025-12-28 | Empirical adjustment method for policy-driven migration scenarios |

### Review Process

| ADR | Title | Status | Date | Summary |
|-----|-------|--------|------|---------|
| [019](019-argument-mapping-claim-review-process.md) | Claim and Argument Mapping Review Process | Accepted | 2025-12-31 | Structured claim inventory and Toulmin-based argument mapping with graphs |

---

## ADR Status Summary

| Status | Count |
|--------|-------|
| Accepted | 18 |
| Proposed | 1 |
| Deprecated | 0 |
| Superseded | 0 |

**Naming Convention Compliance**: All 19 ADRs follow the `NNN-short-title.md` naming convention.

---

## Quick Reference

### For New Team Members

Start with these ADRs to understand the system:
1. **ADR-004**: Core Projection Engine Architecture (system overview)
2. **ADR-007**: Race and Ethnicity Categorization (demographic structure)
3. **ADR-010**: Geographic Scope (what geographies are projected)
4. **ADR-006**: Data Pipeline Architecture (how data flows)

### By Topic

**Demographic Methodology**: ADR-001, ADR-002, ADR-003, ADR-007

**System Design**: ADR-004, ADR-005, ADR-006, ADR-008

**Quality and Operations**: ADR-009, ADR-011, ADR-012

**Data Management**: ADR-016

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

**Last Updated**: 2025-12-28

**Total ADRs**: 16 (16 accepted)

**Template**: See [TEMPLATE.md](TEMPLATE.md) for creating new ADRs
