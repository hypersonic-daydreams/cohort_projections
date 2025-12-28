# ADR-016: Raw Data Management and Cross-Computer Synchronization Strategy

## Status
Accepted

## Date
2025-12-28

## Context

### The Problem

The North Dakota Cohort Component Projection System requires multiple raw data files from external sources:

1. **Census Population Data**: Base population by age, sex, race (from Census Bureau)
2. **Geographic Reference Data**: County and place FIPS codes (from Census Bureau)
3. **Fertility Rates**: Age-specific fertility rates by race (from SEER/CDC)
4. **Life Tables**: Survival rates by age, sex, race (from SEER/CDC)
5. **Migration Data**: IRS county-to-county flows (from IRS SOI)

Several of these data files already exist in sibling repositories on the development machine:
- `/home/nigel/projects/popest` - Census Population Estimates
- `/home/nigel/projects/ndx-econareas` - Geographic reference data
- `/home/nigel/maps` - PUMS microdata

### Multi-Computer Development Environment

Development occurs on two different computers, synchronized using:
- **Git**: For code and small configuration files
- **rclone bisync**: For larger data files that shouldn't be in git

This creates a challenge: how do we manage raw data files so that:
1. Git repositories stay lightweight (no large data files in git history)
2. Data files are available on both development computers
3. Data provenance is documented for the official report
4. Updates from source repositories can be easily incorporated

### Official Report Requirements

This project will produce an official demographic report. All data sources, transformations, and methodologies must be:
- Fully documented
- Reproducible
- Auditable

### Options Considered

1. **Symlinks**: Link to files in sibling repos
   - Won't work: Symlinks don't transfer across machines via git or rclone

2. **Copy files into git**: Commit data files to repository
   - Won't work: Large files bloat git history, poor practice

3. **Git LFS**: Use Git Large File Storage
   - Adds complexity, still puts data in git

4. **Git submodules**: Reference sibling repos as submodules
   - Adds complexity, repos may not have ideal structure

5. **Data manifest + fetch script + rclone**: Document sources, script to populate, sync via rclone
   - Best fit for multi-computer workflow

## Decision

### Decision 1: Hybrid Data Management Strategy

We adopt a hybrid approach:

1. **`data/raw/` contains actual data files** (not symlinks)
2. **`data/raw/` is excluded from git** (via `.gitignore`)
3. **`data/raw/` IS synced via rclone bisync** (between computers)
4. **`config/data_sources.yaml` documents all data sources** (in git)
5. **`scripts/fetch_data.py` populates data from sibling repos** (when available)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA MANAGEMENT STRATEGY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  COMPUTER A (primary)                COMPUTER B (secondary)          │
│  ┌─────────────────────┐            ┌─────────────────────┐         │
│  │ Sibling Repos       │            │ Sibling Repos       │         │
│  │ - popest/           │            │ (may or may not     │         │
│  │ - ndx-econareas/    │            │  exist here)        │         │
│  │ - maps/             │            │                     │         │
│  └──────────┬──────────┘            └─────────────────────┘         │
│             │                                                        │
│             │ fetch_data.py                                          │
│             ▼                                                        │
│  ┌─────────────────────┐   rclone   ┌─────────────────────┐         │
│  │ cohort_projections/ │  bisync   │ cohort_projections/ │         │
│  │   data/raw/         │ ◄───────► │   data/raw/         │         │
│  │   (actual files)    │           │   (actual files)    │         │
│  └─────────────────────┘           └─────────────────────┘         │
│                                                                      │
│  Git tracks:                        Git tracks:                      │
│  - config/data_sources.yaml         - config/data_sources.yaml      │
│  - scripts/fetch_data.py            - scripts/fetch_data.py         │
│  - data/raw/.gitkeep files          - data/raw/.gitkeep files       │
│                                                                      │
│  rclone syncs:                      rclone syncs:                    │
│  - data/raw/**/*.csv                - data/raw/**/*.csv             │
│  - data/raw/**/*.parquet            - data/raw/**/*.parquet         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Decision 2: Data Sources Manifest (`config/data_sources.yaml`)

A YAML manifest documents all data sources with:
- **Description**: What the data is and why it's needed
- **Source paths**: Where to find data in sibling repos (if available)
- **Destination path**: Where to copy within this project
- **External source**: Where to download if not available locally
- **Required columns**: What the data must contain
- **Notes**: Processing or quality notes

**Example Structure**:
```yaml
data_sources:
  geographic:
    nd_counties:
      description: "North Dakota county reference data (53 counties)"
      source_paths:
        - "${HOME}/projects/ndx-econareas/data/processed/reference/popest/2024/population_county_2024.csv"
        - "${HOME}/projects/popest/data/raw/counties/totals/co-est2024-alldata.csv"
      destination: "data/raw/geographic/nd_counties.csv"
      external_source: "Census Bureau Population Estimates Program"
      external_url: "https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html"
      required_columns: ["county_fips", "county_name", "state_fips"]
      notes: "Filter to STATE=38 (North Dakota) if using national file"
```

**Benefits**:
- Documents data provenance for official report
- Enables automated fetching when sources available
- Provides download instructions when sources unavailable
- Version-controlled documentation

### Decision 3: Fetch Script (`scripts/fetch_data.py`)

A Python script that:
1. Reads `config/data_sources.yaml`
2. For each data source:
   - Checks if source paths exist
   - Copies to destination if found
   - Reports what's missing and where to get it
3. Validates that required columns exist
4. Generates a fetch report

**Usage Patterns**:
```bash
# Fetch all available data from sibling repos
python scripts/fetch_data.py

# Fetch only geographic data
python scripts/fetch_data.py --category geographic

# Check what's missing without fetching
python scripts/fetch_data.py --dry-run

# Force re-fetch even if destination exists
python scripts/fetch_data.py --force
```

**Not a requirement**: The fetch script is a convenience tool. Data can also be:
- Manually copied
- Downloaded directly from external sources
- Synced via rclone from another computer

### Decision 4: Directory Structure with `.gitkeep` Files

Ensure `data/raw/` directory structure is preserved in git while excluding actual data files.

**Structure**:
```
data/
├── raw/
│   ├── .gitkeep
│   ├── geographic/
│   │   └── .gitkeep
│   ├── population/
│   │   └── .gitkeep
│   ├── fertility/
│   │   └── .gitkeep
│   ├── mortality/
│   │   └── .gitkeep
│   └── migration/
│       └── .gitkeep
└── processed/
    └── (generated by processing pipeline)
```

**`.gitignore` rules**:
```gitignore
# Raw data files (synced via rclone, not git)
data/raw/**/*.csv
data/raw/**/*.parquet
data/raw/**/*.xlsx
data/raw/**/*.xls
data/raw/**/*.json

# Keep directory structure
!data/raw/.gitkeep
!data/raw/**/.gitkeep
```

### Decision 5: Data Categories and Sources

Based on the repository evaluation, here are the data sources:

**Available Locally (from sibling repos)**:

| Category | Data | Source Repository | Source Path |
|----------|------|-------------------|-------------|
| Geographic | ND Counties (53) | ndx-econareas | `data/processed/reference/popest/2024/population_county_2024.csv` |
| Geographic | ND Places (355) | popest | `data/raw/cities/totals/sub-est2024_38.csv` |
| Geographic | Metro Crosswalk | ndx-econareas | `data/processed/reference/omb/2023/county_to_cbsa_2023.csv` |
| Population | County Population | popest | `data/raw/counties/totals/co-est2024-alldata.csv` |
| Population | State Population | popest | `data/raw/state/totals/NST-EST2024-ALLDATA.csv` |
| Population | PUMS (age-sex-race) | maps | `data/raw/pums_person.parquet` |

**Must Be Downloaded Externally**:

| Category | Data | External Source | URL |
|----------|------|-----------------|-----|
| Fertility | ASFR by race | SEER or CDC WONDER | seer.cancer.gov or wonder.cdc.gov |
| Mortality | Life tables | SEER or CDC NVSS | cdc.gov/nchs/products/life_tables.htm |
| Migration | IRS flows | IRS SOI | irs.gov/statistics/soi-tax-stats-migration-data |

## Consequences

### Positive

1. **Multi-Computer Compatibility**: Data files sync via rclone between computers
2. **Git Stays Lightweight**: No large data files in git history
3. **Documented Provenance**: YAML manifest provides full data lineage for official report
4. **Flexible Sourcing**: Can use sibling repos, manual download, or rclone sync
5. **Reproducible**: Clear instructions for obtaining all data
6. **Auditable**: Data sources documented for stakeholder review
7. **DRY Principle**: Avoids duplicating data that exists in sibling repos
8. **Graceful Degradation**: Works even if sibling repos don't exist

### Negative

1. **Initial Setup Required**: Must run fetch script or manually copy data on each computer
2. **Dependency on Sibling Repos**: Some data sources depend on other repos being present
3. **Sync Discipline Required**: Must remember to rclone bisync after fetching new data
4. **External Downloads Still Manual**: SEER/CDC/IRS data requires manual download

### Risks and Mitigations

**Risk**: Sibling repos change structure, breaking source paths
- **Mitigation**: Multiple source paths in manifest (fallback options)
- **Mitigation**: Fetch script logs clear errors about missing sources
- **Mitigation**: Can always manually copy or download data

**Risk**: Data files get out of sync between computers
- **Mitigation**: rclone bisync handles synchronization
- **Mitigation**: Fetch script can re-populate from sources

**Risk**: Data provenance unclear for official report
- **Mitigation**: Comprehensive YAML manifest with all source details
- **Mitigation**: Processing pipeline generates metadata files

## Alternatives Considered

### Alternative 1: Symlinks to Sibling Repos

**Description**: Create symbolic links from `data/raw/` to files in sibling repos.

```bash
ln -s ~/projects/popest/data/raw/counties/totals/co-est2024-alldata.csv \
      data/raw/population/co-est2024-alldata.csv
```

**Pros**:
- No data duplication
- Always uses latest version from source

**Cons**:
- Symlinks don't transfer via git (commits dangling links)
- Symlinks don't transfer via rclone (or transfer incorrectly)
- Requires identical paths on both computers
- Breaks if sibling repo moves or is renamed

**Why Rejected**: Fundamentally incompatible with multi-computer git+rclone workflow.

### Alternative 2: Commit Data Files to Git

**Description**: Add data files directly to git repository.

**Pros**:
- Simple - files travel with code
- No external dependencies

**Cons**:
- Bloats git history (data files are 10-100MB+)
- Slow clone/pull operations
- Can't easily update data without new commits
- Poor practice for binary/large files

**Why Rejected**: Git is not designed for large data files.

### Alternative 3: Git LFS (Large File Storage)

**Description**: Use Git LFS extension for large data files.

**Pros**:
- Works with git workflow
- Handles large files properly

**Cons**:
- Adds complexity (LFS setup, bandwidth limits)
- Still puts data in git (conceptually)
- Requires LFS support on all clients
- May have storage costs

**Why Rejected**: Adds complexity without significant benefit over rclone approach.

### Alternative 4: Git Submodules

**Description**: Reference sibling repos as git submodules.

```bash
git submodule add ../popest data/external/popest
```

**Pros**:
- Formal dependency relationship
- Version-controlled references

**Cons**:
- Submodules are notoriously complex
- Requires sibling repos to have ideal structure
- Doesn't solve sync problem (still need data in this repo)
- Adds checkout complexity

**Why Rejected**: Adds complexity; sibling repos not structured for submodule use.

### Alternative 5: Cloud Storage Only (S3, GCS)

**Description**: Store all data in cloud storage, fetch on demand.

**Pros**:
- Single source of truth
- Accessible from anywhere

**Cons**:
- Adds cloud dependency and cost
- Requires network access
- More complex setup
- Overkill for two-computer workflow

**Why Rejected**: rclone bisync already solves the sync problem more simply.

## Implementation

### Files to Create

1. **`config/data_sources.yaml`**: Complete manifest of all data sources
2. **`scripts/fetch_data.py`**: Script to fetch data from sibling repos
3. **`.gitkeep` files**: In each `data/raw/` subdirectory
4. **Update `.gitignore`**: Exclude raw data files

### Workflow

**Initial Setup (Computer A - has sibling repos)**:
```bash
# 1. Clone repository
git clone <repo-url>
cd cohort_projections

# 2. Fetch data from sibling repos
python scripts/fetch_data.py

# 3. Sync to remote storage
rclone bisync cohort_projections/data/raw remote:cohort_projections/data/raw
```

**Initial Setup (Computer B)**:
```bash
# 1. Clone repository
git clone <repo-url>
cd cohort_projections

# 2. Sync data from remote storage
rclone bisync remote:cohort_projections/data/raw cohort_projections/data/raw

# (Or run fetch_data.py if sibling repos exist here too)
```

**Ongoing Development**:
```bash
# After making changes on either computer
rclone bisync cohort_projections/data/raw remote:cohort_projections/data/raw
```

### Data Quality Validation

The fetch script performs basic validation:
- Verifies required columns exist
- Checks file is not empty
- Reports row counts

Full data validation occurs in the processing pipeline (see ADR-006).

## References

1. **ADR-006**: Data Pipeline Architecture (processing stage)
2. **rclone Documentation**: https://rclone.org/bisync/
3. **Git Large Files Best Practices**: https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
4. **Census Bureau Data Standards**: https://www.census.gov/programs-surveys/popest.html

## Revision History

- **2025-12-28**: Initial version (ADR-016) - Raw data management strategy

## Related ADRs

- **ADR-005**: Configuration Management Strategy (YAML configuration)
- **ADR-006**: Data Pipeline Architecture (processing of raw data)
- **ADR-008**: BigQuery Integration Design (alternative data source)
- **ADR-012**: Output and Export Format Strategy (processed data formats)
