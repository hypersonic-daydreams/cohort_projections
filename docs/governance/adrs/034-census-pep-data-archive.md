# ADR-034: Census Population Estimates Program (PEP) Data Archive

## Status
Accepted

## Date
2026-02-02

## Context

The cohort projections system requires historical population data at multiple geographic levels (state, county, place) spanning multiple decades. The Census Bureau's Population Estimates Program (PEP) provides annual population estimates between decennial censuses.

### Challenge

Census PEP data presents several data management challenges:

1. **Vintage differences**: Column names, geographic codes, and methodologies change between decades
2. **Multiple geographic levels**: State, county, place (city), and various demographic breakdowns
3. **Provenance preservation**: Need to retain original data for reproducibility
4. **Harmonization complexity**: Combining vintages requires careful crosswalking

### Data Source

Primary source: https://www2.census.gov/programs-surveys/popest/datasets/

Key datasets:
- **Place (city) estimates**: `cities/totals/sub-est*.csv`
- **County estimates**: `counties/totals/co-est*.csv`
- **State estimates**: `state/totals/`
- **Age/sex/race detail**: Various subdirectories

### Storage Considerations

Some individual PEP files are small, but a full national archive across decades includes a few large “detail” tables
and can exceed the original rough estimate.

**Current archive (as of 2026-02-03)**:
- Raw files downloaded: **23 files**
- Raw staging size: **~278 MB** (`~/workspace/shared-data/census/popest/raw/`)
- Largest single file: `cc-est2020int-alldata.csv` (**~169 MB**, county intercensal 2010–2020)
- Technical documentation included (PDFs for 2020–2024 file layouts + methodology)

## Decision

### Decision 1: Shared Data Directory for Cross-Project Use

**Decision**: Store Census PEP data in a shared workspace directory (`~/workspace/shared-data/census/popest/`) rather than within individual repositories. Projects reference this location via an environment variable.

**Rationale**:
- Census PEP data is canonical and used across multiple demographic projects
- Prevents duplication of ~50-60MB datasets across repositories
- Single source of truth for data updates and versioning
- Already integrated with existing rclone bisync workflow
- Projects remain portable through environment variable configuration

**Configuration**: Projects access the data via `CENSUS_POPEST_DIR` environment variable:

```bash
# In .env
CENSUS_POPEST_DIR=~/workspace/shared-data/census/popest
```

**PostgreSQL (analytics layer)**: Use a dedicated DSN via `CENSUS_POPEST_PG_DSN`:

```bash
# In .env (example)
CENSUS_POPEST_PG_DSN="postgresql://user:password@localhost:5432/census_popest"

# Local peer-auth example (common on Linux):
CENSUS_POPEST_PG_DSN="postgresql:///census_popest"
```

**Directory Location**: `~/workspace/shared-data/census/popest/`

This directory is:
- Synced to Google Drive via rclone bisync
- Excluded from git in all repositories
- Shared by: cohort_projections, sdc_2024_replication, city_health_dashboard (future)

### Decision 2: Raw Preservation with Parquet Conversion

**Decision**: Download and preserve raw source files, then create 1:1 parquet conversions for efficient querying.
After parquet extraction and integrity checks, compress raw source files **by vintage** into a single archive per
vintage for long-term storage in Google Drive (via rclone bisync), and remove the uncompressed staging files.

**Rationale**:
- Raw files serve as immutable archive (provenance)
- Parquet provides 60-80% compression and fast columnar queries
- 1:1 conversion means no information loss
- Harmonization deferred to query time (not storage time)
- Archiving by vintage keeps related files together and reduces archive sprawl
- Removing uncompressed staging reduces sync time and disk churn while preserving reproducibility

### Decision 3: Directory Structure

**Decision**: Organize by decade/vintage, then geographic level. Treat `raw/` as a staging area; store canonical raw
inputs in per-vintage archives after extraction/validation.

```
~/workspace/shared-data/census/popest/
├── catalog.yaml                    # Master inventory
├── raw/                            # Staging downloads (temporary; deleted after archiving)
│   ├── 2020-2024/
│   │   ├── place/
│   │   │   ├── sub-est2024.csv
│   │   │   └── sub-est2024_layout.txt
│   │   ├── county/
│   │   │   ├── co-est2024-alldata.csv
│   │   │   └── co-est2024-alldata_layout.txt
│   │   └── state/
│   │       └── nst-est2024-alldata.csv
│   ├── 2010-2019/
│   │   ├── place/
│   │   ├── county/
│   │   └── state/
│   └── 2000-2009/
│       ├── place/
│       ├── county/
│       └── state/
├── raw-archives/                   # Canonical raw (compressed), synced to Drive
│   ├── 2020-2024-raw.zip
│   ├── 2010-2020-raw.zip
│   ├── 2000-2010-raw.zip
│   ├── 2000-2009-raw.zip
│   ├── 1990-2000-raw.zip
│   ├── 1980-1990-raw.zip
│   └── 1970-1980-raw.zip
├── parquet/                        # Converted files (1:1 from raw)
│   ├── 2020-2024/
│   │   ├── place/
│   │   │   └── sub-est2024.parquet
│   │   ├── county/
│   │   │   └── co-est2024-alldata.parquet
│   │   └── state/
│   └── ...
├── derived/                        # Derived artifacts (regenerable)
│   └── docs/                       # Extracted PDF text/tables with page references
├── metadata/                       # Per-dataset metadata JSON (machine-readable)
└── docs/                           # Methodology documentation
    ├── vintage_differences.md
    └── geographic_changes.md
```

**Rationale**:
- Decade grouping matches Census methodology vintages
- Geographic level separation keeps files organized
- Parallel raw/parquet structure makes provenance clear
- Documentation lives alongside data
- Entire directory synced as a unit via rclone

### Decision 4: Catalog Schema

**Decision**: Maintain a YAML catalog tracking all datasets with metadata.

```yaml
# catalog.yaml
version: "1.0"
last_updated: "2026-02-02"
source_base_url: "https://www2.census.gov/programs-surveys/popest/datasets/"

datasets:
  # 2020s Vintage
  - id: sub-est2024
    vintage: "2020-2024"
    level: place
    geography: national
    years_covered: [2020, 2021, 2022, 2023, 2024]
    census_base: 2020
    source_path: "2020-2024/cities/totals/sub-est2024.csv"
    downloaded: "2026-02-02"
    raw_file: "raw/2020-2024/place/sub-est2024.csv"
    parquet_file: "parquet/2020-2024/place/sub-est2024.parquet"
    layout_file: "raw/2020-2024/place/sub-est2024_layout.txt"
    row_count: null  # Populated after download
    file_size_bytes: null
    notes: "Postcensal estimates based on 2020 Census"

  - id: co-est2024-alldata
    vintage: "2020-2024"
    level: county
    geography: national
    years_covered: [2020, 2021, 2022, 2023, 2024]
    census_base: 2020
    source_path: "2020-2024/counties/totals/co-est2024-alldata.csv"
    # ... etc

  # 2010s Vintage
  - id: sub-est2019-all
    vintage: "2010-2019"
    level: place
    geography: national
    years_covered: [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    census_base: 2010
    source_path: "2010-2019/cities/totals/sub-est2019_all.csv"
    notes: "Intercensal estimates revised after 2020 Census"
    # ... etc
```

**Rationale**:
- Single source of truth for what's been downloaded
- Captures provenance (source URL, download date)
- Enables programmatic discovery of available data
- Notes field captures vintage-specific caveats

### Decision 5: Vintage-Aware Design (No Premature Harmonization)

**Decision**: Store each vintage as-is. Do not rename columns or transform data during storage. Handle harmonization at query time.

**Known Vintage Differences**:

| Aspect | 2000s | 2010s | 2020s |
|--------|-------|-------|-------|
| Column prefix | `POPESTIMATE` | `POPESTIMATE` | `POPESTIMATE` |
| Race categories | 4 + Hispanic | 6 + combinations | 6 + combinations |
| Geographic base | Census 2000 | Census 2010 | Census 2020 |
| Estimate type | Intercensal (revised) | Intercensal (revised) | Postcensal |

**Rationale**:
- Avoids information loss from premature standardization
- Preserves original column names for documentation reference
- Allows vintage-specific analysis when needed
- Harmonization logic lives in code, not data transforms

### Decision 6: North Dakota Focus with National Archive

**Decision**: Download national files and keep them “raw-as-provided”. Defer ND-filtered derived files until there is
a demonstrated performance need.

**Rationale**:
- National files support future multi-state analysis
- ND subsets can be created later (and can always be regenerated from canonical raw + parquet)
- Preserves option to expand scope later
- Avoids premature creation of multiple “derived” variants before query patterns are clear

## Operational Decisions (As of 2026-02-03)

These choices guide Phase 2+ implementation and are intended to make the archive easy to use by multiple AI agents
and humans, while remaining reproducible.

1. **Metadata for AI agents: Rich**
   - Create per-dataset metadata artifacts (machine-readable) capturing:
     - schema (column names, dtypes), row counts, and file fingerprints
     - geographic keys and year coverage
     - column meanings/value dictionaries when available (from PDFs/docs)
     - vintage-specific “gotchas” and cross-vintage mapping hints

2. **Parquet partitioning: None (initially)**
   - Write **one parquet per source file** (1:1, no transforms).
   - Revisit partitioning only if concrete workloads demand it.

3. **Query layer: PostgreSQL**
   - Build a Postgres-backed “analytics layer” for harmonized cross-vintage tables.
   - Rationale: expected need for harmonization soon, plus concurrent access from multiple AI agents.

4. **Documentation packaging**
   - Include **original** technical documentation (PDFs/layouts/readmes) inside each vintage raw archive.
   - Keep extracted text/tables **outside** the raw archives, clearly labeled as derived, with an index mapping back
     to source PDFs + page references.

## Implementation Plan

### Phase 1: Download + Catalog (Complete ✅)
- [x] Create shared directory structure
- [x] Download national raw files (all targeted vintages/levels)
- [x] Download key technical documentation PDFs (2020–2024)
- [x] Populate `catalog.yaml` with MD5 + file sizes

### Phase 2: Extract (Parquet 1:1, No Transforms)
- [ ] Convert each CSV to parquet (1:1, no transforms)
- [ ] Record parquet row counts + schema fingerprints in `catalog.yaml`
- [ ] Validate conversions (row counts, basic sanity checks)

### Phase 3: Archive Raw by Vintage (Google Drive via bisync)
- [ ] For each vintage, create a single archive containing:
  - the raw source files (CSV/ZIP/TXT) exactly as downloaded
  - the relevant technical documentation for that vintage (PDFs, readmes, layouts)
  - a machine-readable manifest (`manifest.json`) with checksums and original source URLs
- [ ] Verify archive integrity (manifest checksums, unzip test)
- [ ] Delete `raw/` staging files after verification (keep only `raw-archives/`)
- [ ] Update `catalog.yaml` with archive location + manifest checksum

### Phase 4: Documentation Extraction (Careful, Verifiable)
- [ ] Extract text and tables from PDFs into a “derived docs” folder with page references
- [ ] Keep originals unchanged; extracted text is explicitly labeled as derived
- [ ] Create an index mapping extracted artifacts back to the source PDF + page ranges
- [ ] Generate rich per-dataset metadata files (JSON) and link them from `catalog.yaml`

### Phase 5: Query Layer (PostgreSQL Analytics Layer)
- [ ] Define database connection configuration (`CENSUS_POPEST_PG_DSN`) for a local Postgres service
- [ ] Create schemas for:
  - raw/staging representations (optional; may remain parquet-only)
  - harmonized cross-vintage tables (primary interface for analysis/agents)
- [ ] Implement ETL/refresh scripts to (re)build harmonized tables from parquet + documented crosswalk logic
- [ ] Add indexes/materialized views aligned to common query patterns (e.g., time series by `geoid` + `year`)
- [ ] Document concurrency-safe usage patterns and refresh cadence

## Consequences

### Positive
1. **No duplication**: Single shared copy prevents redundant datasets across repos
2. **Reproducible**: Raw files preserved exactly as downloaded with provenance
3. **Efficient**: Parquet provides fast queries and compression
4. **Flexible**: Harmonization deferred, can adapt approach as needed
5. **Documented**: Catalog tracks provenance
6. **Manageable storage**: Raw staging is ~278MB as of 2026-02-03; parquet + archives remain modest
7. **Already synced**: Integrated with existing rclone bisync workflow
8. **Portable projects**: Repositories remain location-agnostic via env vars

### Negative
1. **Some redundancy**: Raw archives + parquet = 2 copies
2. **Manual download**: Census FTP doesn't have a clean API
3. **Harmonization still needed**: Query-time complexity for cross-vintage analysis
4. **Environment dependency**: Projects require CENSUS_POPEST_DIR to be configured

### Risks and Mitigations

**Risk**: Census changes file locations or formats
- **Mitigation**: Archive raw files immediately upon download
- **Mitigation**: Document source URLs in catalog

**Risk**: Column mapping errors during harmonization
- **Mitigation**: Keep raw files, can always re-derive
- **Mitigation**: Unit tests for harmonization logic

## Data Scope (Full Historical Archive)

### Downloaded inventory (as of 2026-02-03)

City/place-level estimates are only available from 2000 forward. This archive includes both “postcensal” and
“intercensal” vintages where available; superseded vintages are retained for reproducibility and comparison.

| Vintage | Places | Counties | States | Docs | Notes |
|---------|--------|----------|--------|------|-------|
| 2020–2024 | ✅ `sub-est2024.csv` | ✅ `co-est2024-alldata.csv` | ✅ `NST-EST2024-ALLDATA.csv` | ✅ (2020–2024 layouts + methodology) | Current postcensal |
| 2010–2020 | ✅ `sub-est2020int.csv` | ✅ `cc-est2020int-alldata.csv` | ✅ `sc-est2020int-alldata5/6.csv` | — | Intercensal (revised; large county file) |
| 2010–2019 | ✅ `sub-est2019_all.csv` (optional) | ✅ `co-est2019-alldata.csv` (optional) | ✅ `sc-est2019-alldata6.csv` (optional) | — | Postcensal (superseded by 2010–2020 intercensal) |
| 2000–2010 | ✅ `sub-est00int.csv`, `sub-est2010-alt.csv` | ✅ `co-est00int-tot.csv` | ✅ `st-est00int-alldata.csv` | — | Intercensal |
| 2000–2009 | — | ✅ `co-est2009-alldata.csv` | — | — | County-only “comprehensive” decade file |
| 1990–2000 | — | ✅ `co-99-10.zip` | — | — | County ASRH detail (compressed) |
| 1980–1990 | — | ✅ `comp8090.zip` | ✅ `st_int_asrh.zip` | — | Compressed archives |
| 1970–1980 | — | ✅ `e7079co.zip` | ✅ `e7080sta.txt` | — | Older national files |

### Current storage (staging)

Raw staging directory size is **~278 MB** (see `catalog.yaml` for per-file MD5 + sizes). Per-vintage archive sizes
will be recorded after archiving.

### Out of Scope
- County-to-county migration flows (separate IRS data source)
- Housing unit estimates (separate dataset)
- Puerto Rico detailed breakdowns

## References

1. Census PEP Documentation: https://www.census.gov/programs-surveys/popest/about.html
2. PEP Methodology: https://www.census.gov/programs-surveys/popest/technical-documentation/methodology.html
3. Dataset Directory: https://www2.census.gov/programs-surveys/popest/datasets/

## Revision History

- **2026-02-02**: Initial version (ADR-034)
- **2026-02-03**: Updated plan after full national downloads; added per-vintage raw archiving + operational decisions

## Related ADRs

- ADR-016: Raw data management strategy (git/rclone hybrid)
- ADR-033: City-level projection methodology (consumer of this data)
- ADR-010: Geographic scope and granularity
