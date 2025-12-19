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

## ADR Index

### Data Processing

#### ADR-001: Fertility Rate Processing Methodology
**Status**: Accepted | **Date**: 2025-12-18

Documents the methodology for processing SEER/NVSS fertility data into age-specific fertility rates suitable for population projections.

**Key Decisions**:
- 5-year weighted averaging for stability
- Zero-fill for missing age-race combinations
- SEER race code mapping to 6 standard categories
- Plausibility thresholds for validation
- Multi-format input support (CSV, TXT, Excel, Parquet)
- Metadata generation for provenance

**File**: [`001-fertility-rate-processing.md`](001-fertility-rate-processing.md)

---

#### ADR-002: Survival Rate Processing Methodology
**Status**: Accepted | **Date**: 2025-12-18

Documents the methodology for converting SEER/CDC life tables into survival rates for the cohort-component projection engine.

**Key Decisions**:
- Multi-method life table conversion (lx, qx, Lx) with automatic selection
- Special handling for age 90+ open-ended group using Tx-based formula
- Lee-Carter style mortality improvement (0.5% annual default)
- Age-specific plausibility thresholds
- Age-appropriate default values for missing data
- Life expectancy calculation for quality assurance

**File**: [`002-survival-rate-processing.md`](002-survival-rate-processing.md)

---

#### ADR-003: Migration Rate Processing Methodology
**Status**: Planned | **Date**: TBD

Will document the methodology for processing IRS county flows and ACS migration data into net migration rates.

**Planned Topics**:
- IRS county-to-county flow processing
- ACS mobility data integration
- International migration allocation
- Age-sex-race distribution of migration
- Smoothing extreme outliers

**File**: `003-migration-rate-processing.md` (to be created)

---

### System Architecture

#### ADR-004: Core Projection Engine Architecture
**Status**: Accepted | **Date**: 2025-12-18

Documents the architecture of the main cohort-component projection engine that implements the standard demographic method.

**Key Decisions**:
- Modular component architecture (fertility, mortality, migration modules)
- Cohort-component method as core algorithm (Census Bureau standard)
- Vectorized pandas operations (no explicit loops)
- DataFrame-based data structures with standardized schemas
- Annual projection intervals (not monthly/quarterly)
- Built-in scenario support (baseline, high/low growth, zero migration)
- Comprehensive validation at every step

**File**: [`004-core-projection-engine-architecture.md`](004-core-projection-engine-architecture.md)

---

#### ADR-005: Configuration Management Strategy
**Status**: Accepted | **Date**: 2025-12-18

Documents how configuration is managed across the projection system.

**Key Decisions**:
- YAML as configuration format (not JSON, TOML, or code)
- Single centralized configuration file (`projection_config.yaml`)
- Nested sections by functional area
- ConfigLoader class for programmatic access
- Sensible default values throughout codebase
- No environment-specific configuration files
- Configuration validation at load time
- Inline documentation via YAML comments

**File**: [`005-configuration-management-strategy.md`](005-configuration-management-strategy.md)

---

#### ADR-006: Data Pipeline Architecture
**Status**: Accepted | **Date**: 2025-12-18

Documents the architecture of the data pipeline that transforms raw data into projection-ready inputs.

**Key Decisions**:
- Two-stage pipeline architecture (Fetch → Process)
- Standardized processor pattern (Load → Harmonize → Process → Validate → Save)
- Parquet as primary storage format (CSV as secondary)
- Multi-format input support with flexible column naming
- Metadata generation for provenance
- Defensive programming with comprehensive validation
- Directory structure by data type and processing stage

**File**: [`006-data-pipeline-architecture.md`](006-data-pipeline-architecture.md)

---

### Demographic Methodology

#### ADR-007: Race and Ethnicity Categorization
**Status**: Accepted | **Date**: 2025-12-18

Documents the race and ethnicity classification system used throughout the projection system.

**Key Decisions**:
- 6-category system balancing detail with data availability
- Hispanic ethnicity as single category (not crossed with race)
- Combined Asian/Pacific Islander category
- "Two or more races" as distinct category
- Explicit mapping from source categories (Census, SEER)
- Consistent ordering across all outputs
- No residual/unknown category

**Categories**:
1. White alone, Non-Hispanic
2. Black alone, Non-Hispanic
3. AIAN alone, Non-Hispanic
4. Asian/PI alone, Non-Hispanic
5. Two or more races, Non-Hispanic
6. Hispanic (any race)

**File**: [`007-race-ethnicity-categorization.md`](007-race-ethnicity-categorization.md)

---

#### ADR-010: Geographic Scope and Granularity
**Status**: Accepted | **Date**: 2025-12-18

Documents the geographic levels at which projections are produced.

**Key Decisions**:
- Three geographic levels (State, County, Place)
- FIPS codes as primary geographic identifier
- All 53 counties, subset of places (configuration-driven)
- Single-year ages (0-90+), not age groups
- Age 90+ as open-ended group
- Geographic hierarchy with aggregation capabilities
- No sub-county geographies beyond places (no tracts/blocks)

**File**: [`010-geographic-scope-granularity.md`](010-geographic-scope-granularity.md)

---

### Technical Infrastructure

#### ADR-008: BigQuery Integration Design
**Status**: Accepted | **Date**: 2025-12-18

Documents how Google BigQuery is integrated for supplementary data access.

**Key Decisions**:
- BigQuery as supplementary (not primary) data source
- Service account authentication
- Query result caching
- Public dataset usage over custom storage
- Wrapper client class for convenience
- Graceful degradation without BigQuery
- No BigQuery for storing primary projection outputs

**Use Cases**: Geographic reference data, historical validation, data exploration

**File**: [`008-bigquery-integration-design.md`](008-bigquery-integration-design.md)

---

#### ADR-009: Logging and Error Handling Strategy
**Status**: Accepted | **Date**: 2025-12-18

Documents how logging and error handling are implemented across the system.

**Key Decisions**:
- Python standard logging module (not print statements)
- Centralized logger configuration
- Hierarchical log levels philosophy (DEBUG, INFO, WARNING, ERROR)
- Log to both file and console
- Defensive programming with explicit validation
- Error vs. warning philosophy (errors for fatal, warnings for recoverable)
- Structured logging for key events

**File**: [`009-logging-error-handling-strategy.md`](009-logging-error-handling-strategy.md)

---

#### ADR-011: Testing Strategy
**Status**: Accepted | **Date**: 2025-12-18

Documents the testing approach for ensuring system correctness.

**Key Decisions**:
- Pragmatic testing approach (not full TDD)
- Built-in validation functions as primary quality assurance
- Example scripts as integration tests
- Selective unit tests for critical functions
- Manual validation against known benchmarks
- pytest as unit test framework
- Synthetic test data generation

**Testing Pyramid**: Manual Review > Integration Examples > Built-In Validation > Unit Tests (selective)

**File**: [`011-testing-strategy.md`](011-testing-strategy.md)

---

#### ADR-012: Output and Export Format Strategy
**Status**: Accepted | **Date**: 2025-12-18

Documents how projection results are exported and stored.

**Key Decisions**:
- Dual format strategy (Parquet primary + CSV secondary)
- Gzip compression for Parquet
- Include zero cells in output
- Two decimal places for population values
- Separate files by geography and year range
- JSON metadata files with comprehensive provenance
- Optional Excel export for stakeholder reports
- Summary tables in addition to detailed outputs

**File**: [`012-output-export-format-strategy.md`](012-output-export-format-strategy.md)

---

## ADR Status Summary

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| 001 | Fertility Rate Processing | Accepted | 2025-12-18 |
| 002 | Survival Rate Processing | Accepted | 2025-12-18 |
| 003 | Migration Rate Processing | Planned | TBD |
| 004 | Core Projection Engine Architecture | Accepted | 2025-12-18 |
| 005 | Configuration Management Strategy | Accepted | 2025-12-18 |
| 006 | Data Pipeline Architecture | Accepted | 2025-12-18 |
| 007 | Race and Ethnicity Categorization | Accepted | 2025-12-18 |
| 008 | BigQuery Integration Design | Accepted | 2025-12-18 |
| 009 | Logging and Error Handling Strategy | Accepted | 2025-12-18 |
| 010 | Geographic Scope and Granularity | Accepted | 2025-12-18 |
| 011 | Testing Strategy | Accepted | 2025-12-18 |
| 012 | Output and Export Format Strategy | Accepted | 2025-12-18 |

**Status Definitions**:
- **Accepted**: Decision has been made and implemented
- **Planned**: Decision documented but implementation pending
- **Superseded**: Decision replaced by later ADR
- **Deprecated**: Decision no longer applicable

## How to Use These ADRs

### For New Team Members

Start with these ADRs to understand the system:
1. **ADR-004**: Core Projection Engine Architecture (system overview)
2. **ADR-007**: Race and Ethnicity Categorization (demographic structure)
3. **ADR-010**: Geographic Scope (what geographies are projected)
4. **ADR-006**: Data Pipeline Architecture (how data flows)

### For Understanding a Specific Topic

**Demographic Methodology**:
- ADR-001: Fertility processing
- ADR-002: Survival/mortality processing
- ADR-003: Migration processing (planned)
- ADR-007: Race/ethnicity categories

**System Design**:
- ADR-004: Core engine
- ADR-005: Configuration
- ADR-006: Data pipeline
- ADR-008: BigQuery integration

**Quality and Operations**:
- ADR-009: Logging and errors
- ADR-011: Testing approach
- ADR-012: Output formats

### For Making New Decisions

When making a new architectural decision:

1. **Review Related ADRs**: Check if similar decisions were made
2. **Document Your Decision**: Create new ADR using template below
3. **Update This Index**: Add entry to appropriate section
4. **Reference in Code**: Link to ADR in comments for critical code

### ADR Template

```markdown
# ADR-XXX: [Decision Title]

## Status
[Proposed | Accepted | Superseded | Deprecated]

## Date
YYYY-MM-DD

## Context
[What is the situation requiring a decision?]
[What constraints exist?]
[What requirements drive this?]

## Decision
[What was decided?]
[How does it work?]

## Consequences

### Positive
- [Benefit 1]
- [Benefit 2]

### Negative
- [Tradeoff 1]
- [Tradeoff 2]

## Alternatives Considered

### Alternative 1: [Name]
- **Description**: ...
- **Pros**: ...
- **Cons**: ...
- **Why rejected**: ...

## References
- [Relevant documentation]
- [External resources]
```

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

## Cross-Cutting Concerns

### Data Quality
- Comprehensive validation (ADR-006, ADR-009, ADR-011)
- Plausibility checks (ADR-001, ADR-002)
- Metadata and provenance (ADR-006, ADR-012)

### Performance
- Vectorized operations (ADR-004)
- Parquet compression (ADR-006, ADR-012)
- Geographic modularity (ADR-010)

### Usability
- Multiple output formats (ADR-012)
- Clear error messages (ADR-009)
- Configuration flexibility (ADR-005)
- Example scripts (ADR-011)

### Maintainability
- Modular architecture (ADR-004, ADR-006)
- Consistent patterns (ADR-006)
- Comprehensive logging (ADR-009)
- Documentation throughout

## Future ADRs

Potential topics for future ADRs:

- **ADR-013**: Scenario Development and Management
- **ADR-014**: Uncertainty Quantification Methods
- **ADR-015**: Visualization and Dashboard Integration
- **ADR-016**: Multi-Region Aggregation Rules
- **ADR-017**: Historical Backcasting Methodology
- **ADR-018**: API Design for Programmatic Access

## Resources

### Demographic Methodology
- Preston, Heuveline, Guillot: "Demography: Measuring and Modeling Population Processes" (2001)
- Smith, Tayman, Swanson: "State and Local Population Projections" (2001)
- U.S. Census Bureau Methodology Documentation

### Technical References
- Pandas Documentation: https://pandas.pydata.org/docs/
- Apache Parquet: https://parquet.apache.org/
- Python Logging: https://docs.python.org/3/howto/logging.html

### Data Sources
- SEER: https://seer.cancer.gov/popdata/
- Census Bureau: https://www.census.gov/programs-surveys/popproj.html
- CDC NVSS: https://www.cdc.gov/nchs/nvss/index.htm

## Contact

For questions about these ADRs or to propose new architectural decisions, contact the project team.

---

**Last Updated**: 2025-12-18

**Total ADRs**: 12 (11 accepted, 1 planned)
