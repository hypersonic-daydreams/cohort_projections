# ADR-006: Data Pipeline Architecture

## Status
Accepted

## Date
2025-12-18

## Context

The cohort component projection engine requires four types of input data, each from different sources with different formats, quality levels, and update frequencies. A systematic data pipeline is needed to transform raw data from Census, SEER, IRS, and other sources into clean, validated inputs ready for projection.

### Data Requirements

**Four Input Types**:
1. **Base Population**: Starting population by age/sex/race (Census PEP, ACS)
2. **Fertility Rates**: Age-specific fertility rates by race (SEER, NVSS)
3. **Survival Rates**: Life tables by age/sex/race (SEER, CDC)
4. **Migration Rates**: Net migration by cohort (IRS, ACS, Census)

### Challenges

1. **Multiple Data Sources**: Census API, SEER downloads, IRS files, BigQuery - each with different access methods
2. **Format Variations**: CSV, TXT, Excel, JSON, API responses - inconsistent structures
3. **Quality Issues**: Missing values, outliers, coding errors, inconsistent naming
4. **Transformation Complexity**: Raw data requires significant processing to become projection-ready
5. **Provenance**: Need to track data lineage (source → processed)
6. **Reproducibility**: Pipeline must be repeatable and auditable
7. **Updates**: Data sources update annually; pipeline must handle new vintages

### Requirements

1. **Modularity**: Separate fetching from processing
2. **Reusability**: Common patterns across different data types
3. **Validation**: Comprehensive quality checks at each stage
4. **Documentation**: Metadata tracking for all transformations
5. **Fault Tolerance**: Graceful handling of missing/corrupted data
6. **Performance**: Efficient processing of large datasets
7. **Storage**: Organized storage of raw and processed data

## Decision

### Decision 1: Two-Stage Pipeline Architecture (Fetch → Process)

**Decision**: Separate data pipeline into two distinct stages with clear responsibilities.

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  STAGE 1: FETCH                    STAGE 2: PROCESS         │
│  ┌──────────────────┐              ┌──────────────────┐     │
│  │ External Sources │              │ Raw Data         │     │
│  │ - Census API     │              │ - Various formats│     │
│  │ - SEER Downloads │  ───────>    │ - Heterogeneous  │     │
│  │ - IRS Files      │              │ - Unvalidated    │     │
│  │ - BigQuery       │              └─────────┬────────┘     │
│  └──────────────────┘                        │              │
│                                               │              │
│  data/raw/                                    ▼              │
│    population/                    ┌──────────────────┐      │
│    fertility/                     │ Processors       │      │
│    mortality/                     │ - Harmonize      │      │
│    migration/                     │ - Transform      │      │
│                                   │ - Validate       │      │
│                                   └─────────┬────────┘      │
│                                             │               │
│                                             ▼               │
│                                   ┌──────────────────┐      │
│                                   │ Projection-Ready │      │
│                                   │ - Standard schema│      │
│                                   │ - Validated      │      │
│                                   │ - Documented     │      │
│                                   └──────────────────┘      │
│                                                             │
│  data/processed/                                            │
│    population/                                              │
│    fertility/                                               │
│    mortality/                                               │
│    migration/                                               │
└─────────────────────────────────────────────────────────────┘
```

**Stage 1: Fetch** (`cohort_projections/data/fetch/`):
- **Responsibility**: Download/query raw data from external sources
- **Output**: Raw data saved to `data/raw/` in original format
- **Modules**:
  - `census_api.py`: Census API integration (PEP, ACS)
  - `vital_stats.py`: SEER/NVSS data (fertility, mortality)
  - `migration_data.py`: IRS/ACS migration data
  - `geographic.py`: Geographic reference data (FIPS codes)

**Stage 2: Process** (`cohort_projections/data/process/`):
- **Responsibility**: Transform raw data into projection-ready format
- **Output**: Standardized DataFrames saved to `data/processed/` as Parquet/CSV
- **Modules**:
  - `base_population.py`: Process base population
  - `fertility_rates.py`: Process fertility rates
  - `survival_rates.py`: Process life tables → survival rates
  - `migration_rates.py`: Process migration data → net migration rates

**Rationale**:
- **Separation of Concerns**: Fetching data ≠ processing data (different logic, dependencies)
- **Reusability**: Can re-process data without re-fetching
- **Development**: Can work offline with cached raw data
- **Testing**: Can test processors with synthetic data
- **Maintenance**: Changes to source APIs don't affect processing logic

### Decision 2: Standardized Processor Pattern

**Decision**: All data processors follow a common 5-step pattern: **Load → Harmonize → Process → Validate → Save**.

**Standard Processor Structure**:
```python
def process_[data_type](
    input_path: Path,
    output_path: Path,
    config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Process [data type] from raw format to projection-ready format.

    Args:
        input_path: Path to raw data file(s)
        output_path: Path to save processed data
        config: Configuration dictionary

    Returns:
        Processed DataFrame ready for projection engine
    """
    logger.info(f"Processing {data_type}...")

    # 1. LOAD: Read raw data (multiple formats supported)
    raw_data = load_raw_data(input_path)

    # 2. HARMONIZE: Standardize categories (race codes, sex labels, etc.)
    harmonized = harmonize_categories(raw_data, config)

    # 3. PROCESS: Apply transformations (averaging, calculation, aggregation)
    processed = apply_transformations(harmonized, config)

    # 4. VALIDATE: Check plausibility and completeness
    is_valid, issues = validate_data(processed, config)
    if not is_valid:
        logger.warning(f"Validation issues: {issues}")

    # 5. SAVE: Export to multiple formats with metadata
    save_processed_data(processed, output_path, metadata={...})

    logger.info(f"Processing complete: {output_path}")
    return processed
```

**Each Step Explained**:

**1. LOAD**: Multi-format data loading
```python
def load_raw_data(input_path):
    if input_path.suffix == '.csv':
        return pd.read_csv(input_path)
    elif input_path.suffix == '.xlsx':
        return pd.read_excel(input_path)
    elif input_path.suffix == '.txt':
        return pd.read_csv(input_path, sep='\t')
    elif input_path.suffix == '.parquet':
        return pd.read_parquet(input_path)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")
```

**2. HARMONIZE**: Category standardization
```python
def harmonize_categories(df, config):
    # Map race codes to standard 6 categories
    df = harmonize_race_categories(df)

    # Standardize sex labels (M/F → Male/Female)
    df = harmonize_sex_labels(df)

    # Standardize column names
    df = standardize_columns(df)

    return df
```

**3. PROCESS**: Data transformations
```python
def apply_transformations(df, config):
    # Average over multiple years
    df = calculate_average(df, averaging_period=5)

    # Fill missing combinations
    df = create_complete_matrix(df, ages, sexes, races)

    # Apply specific calculations (e.g., life table → survival rate)
    df = calculate_rates(df)

    return df
```

**4. VALIDATE**: Quality checks
```python
def validate_data(df, config):
    issues = []

    # Check required columns
    required = ['age', 'sex', 'race', 'rate']
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check value ranges
    if (df['rate'] < 0).any():
        issues.append("Negative rates detected")

    # Check completeness
    expected_rows = num_ages * num_sexes * num_races
    if len(df) != expected_rows:
        issues.append(f"Expected {expected_rows} rows, got {len(df)}")

    return len(issues) == 0, issues
```

**5. SAVE**: Export with metadata
```python
def save_processed_data(df, output_path, metadata):
    # Save primary format (Parquet)
    df.to_parquet(f"{output_path}.parquet", compression='gzip')

    # Save human-readable format (CSV)
    df.to_csv(f"{output_path}.csv", index=False)

    # Save metadata (JSON)
    with open(f"{output_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

**Rationale**:
- **Consistency**: All processors work the same way (easier to learn and maintain)
- **Completeness**: No steps skipped (load, harmonize, validate all happen)
- **Auditability**: Clear progression from raw → processed
- **Testability**: Each step can be tested independently
- **Reusability**: Common functions shared across processors

### Decision 3: Parquet as Primary Storage Format

**Decision**: Use Apache Parquet as the primary format for processed data, with CSV as secondary human-readable backup.

**File Outputs** (for each data type):
```
data/processed/fertility/
  fertility_rates.parquet          # Primary (efficient, typed)
  fertility_rates.csv              # Secondary (human-readable)
  fertility_rates_metadata.json    # Provenance and quality info
```

**Parquet Advantages**:
- **Performance**: 5-10x faster to read/write than CSV
- **Size**: 50-80% smaller than CSV (compression)
- **Types**: Preserves data types (int, float, datetime)
- **Schema**: Stores column metadata
- **Pandas Integration**: Native support via `pyarrow`

**CSV Advantages**:
- **Human-Readable**: Open in Excel, text editor
- **Universal**: Works everywhere
- **Debugging**: Easy to inspect
- **Compatibility**: Any tool can read

**Usage Pattern**:
```python
# Projection engine reads Parquet (fast)
fertility_rates = pd.read_parquet('data/processed/fertility/fertility_rates.parquet')

# Users inspect CSV (readable)
# Open fertility_rates.csv in Excel
```

**Rationale**:
- **Best of Both**: Performance + accessibility
- **Negligible Cost**: Parquet compression offsets CSV duplication
- **User-Friendly**: Non-programmers can inspect data
- **Production-Ready**: Fast loading for projections

### Decision 4: Multi-Format Input Support with Flexible Column Naming

**Decision**: Processors accept multiple input formats (CSV, TXT, Excel, Parquet) and flexibly recognize column names.

**Format Support**:
```python
# Automatic format detection by file extension
SUPPORTED_FORMATS = {
    '.csv': pd.read_csv,
    '.txt': lambda p: pd.read_csv(p, sep='\t'),
    '.xlsx': pd.read_excel,
    '.xls': pd.read_excel,
    '.parquet': pd.read_parquet,
    '.json': pd.read_json
}
```

**Flexible Column Recognition**:
```python
# Find "age" column (case-insensitive, variant names)
def find_column(df, variants):
    """Find column in DataFrame using variant names."""
    variants_lower = [v.lower() for v in variants]
    for col in df.columns:
        if col.lower() in variants_lower:
            return col
    return None

# Example usage
age_col = find_column(df, ['age', 'AGE', 'age_group', 'AGE_OF_MOTHER'])
race_col = find_column(df, ['race', 'RACE', 'race_ethnicity', 'RACE_ORIGIN', 'origin'])
```

**Rationale**:
- **Data Source Variability**: SEER, Census, IRS use different formats and naming
- **User Convenience**: Don't require preprocessing of input files
- **Robustness**: Handle real-world messy data
- **Fewer Errors**: Flexible matching reduces "column not found" failures

**Trade-off**: More complex column detection logic, but worth it for usability.

### Decision 5: Metadata Generation for Provenance

**Decision**: Generate comprehensive JSON metadata file with every processing run documenting source, transformations, and quality metrics.

**Metadata Schema**:
```json
{
  "processing_date": "2025-12-18T14:30:00",
  "processor_version": "1.0.0",
  "source_file": "data/raw/fertility/seer_fertility_2018_2022.csv",
  "source_date": "2023-06-15",
  "year_range": [2018, 2022],
  "averaging_period": 5,
  "transformations": [
    "Loaded raw SEER data (15,470 rows)",
    "Mapped SEER race codes to 6 categories",
    "Averaged fertility rates over 5 years (2018-2022)",
    "Filled missing age-race combinations with zeros",
    "Validated plausibility (TFR range 1.3-2.4)"
  ],
  "quality_metrics": {
    "total_records": 210,
    "missing_values": 0,
    "outliers_detected": 0,
    "tfr_by_race": {
      "White alone, Non-Hispanic": 1.65,
      "Black alone, Non-Hispanic": 1.82,
      "Hispanic (any race)": 2.05
    }
  },
  "validation_status": "passed",
  "validation_warnings": [],
  "configuration_used": {
    "averaging_period": 5,
    "apply_to_ages": [15, 49],
    "race_categories": [...]
  }
}
```

**Rationale**:
- **Reproducibility**: Can recreate processing from metadata
- **Auditability**: Full lineage from source to output
- **Quality Assurance**: Metrics allow quick validation
- **Debugging**: Easier to diagnose issues with full provenance
- **Documentation**: Self-documenting pipeline

**When Metadata is Used**:
- Quality assurance reviews
- Debugging data issues
- Comparing different processing runs
- Documentation for stakeholders
- Reproducing analyses

### Decision 6: Defensive Programming with Comprehensive Validation

**Decision**: Validate data at every pipeline stage with clear error levels (ERROR vs WARNING).

**Validation Levels**:

**ERRORS** (fail processing):
- Missing required columns
- Empty datasets
- Negative values where impossible (e.g., populations, rates)
- Values outside possible range (e.g., survival rate > 1.0)
- Missing required categories (e.g., no age 0 in base population)

**WARNINGS** (log but continue):
- Values outside typical range (unusual but possible)
- Missing optional data (fill with defaults)
- Zero values for entire cohorts
- Total Fertility Rate outside typical range (1.0-3.0)

**Validation Implementation**:
```python
def validate_fertility_rates(df, config):
    """Validate fertility rates for plausibility."""
    issues = []

    # ERROR: Check required columns
    required = ['age', 'race', 'fertility_rate']
    if not all(col in df.columns for col in required):
        issues.append("ERROR: Missing required columns")
        return False, issues

    # ERROR: Check for negative rates
    if (df['fertility_rate'] < 0).any():
        issues.append("ERROR: Negative fertility rates detected")
        return False, issues

    # ERROR: Check biological plausibility
    if (df['fertility_rate'] > 0.15).any():
        issues.append("ERROR: Implausible fertility rate > 0.15")
        return False, issues

    # WARNING: Check typical ranges
    if (df['fertility_rate'] > 0.13).any():
        issues.append("WARNING: Unusually high fertility rate > 0.13")

    # WARNING: Check TFR
    tfr = df.groupby('race')['fertility_rate'].sum()
    for race, value in tfr.items():
        if value < 1.0 or value > 3.0:
            issues.append(f"WARNING: TFR for {race} = {value:.2f} outside typical range")

    return True, issues
```

**Rationale**:
- **Data Quality**: Catch errors before they reach projection engine
- **Fail Fast**: Stop processing on critical errors
- **Informative**: Clear messages about what's wrong
- **Flexible**: Warnings allow edge cases while flagging unusual patterns

### Decision 7: Directory Structure by Data Type and Processing Stage

**Decision**: Organize data storage by type and stage with clear naming conventions.

**Directory Structure**:
```
data/
  raw/                              # Stage 1 outputs (from fetch)
    population/
      census_pep_2025.csv
      acs_5yr_2020_2024.csv
    fertility/
      seer_fertility_2018_2022.csv
      nvss_births_2020_2024.xlsx
    mortality/
      seer_lifetables_2020.csv
      cdc_life_tables_2020.xlsx
    migration/
      irs_county_flows_2018_2022.csv
      acs_mobility_2020_2024.csv
    geographic/
      nd_counties_fips.csv
      nd_places_fips.csv

  processed/                        # Stage 2 outputs (from process)
    population/
      base_population_2025.parquet
      base_population_2025.csv
      base_population_2025_metadata.json
    fertility/
      fertility_rates.parquet
      fertility_rates.csv
      fertility_rates_metadata.json
    mortality/
      survival_rates.parquet
      survival_rates.csv
      survival_rates_metadata.json
    migration/
      migration_rates.parquet
      migration_rates.csv
      migration_rates_metadata.json

  interim/                          # Optional: intermediate processing steps
    fertility/
      seer_fertility_annual.parquet
      seer_fertility_harmonized.parquet
```

**Naming Conventions**:
- **Raw data**: `{source}_{type}_{year_range}.{ext}`
- **Processed data**: `{type}_rates.{ext}` or `{type}_{year}.{ext}`
- **Metadata**: `{filename}_metadata.json`

**Rationale**:
- **Organization**: Easy to find data by type and stage
- **Clarity**: Structure mirrors pipeline stages
- **Git-Friendly**: Can `.gitignore` large raw files, commit processed summaries
- **Scalability**: Easy to add new data types
- **Reproducibility**: Clear inputs and outputs

## Consequences

### Positive

1. **Modularity**: Fetch and process stages cleanly separated
2. **Consistency**: All processors follow same pattern
3. **Reusability**: Common functions shared across processors
4. **Robustness**: Comprehensive validation catches errors early
5. **Transparency**: Metadata provides full audit trail
6. **Performance**: Parquet provides fast loading for projections
7. **User-Friendly**: CSV backup allows inspection in Excel
8. **Flexibility**: Multi-format input support handles real-world data
9. **Maintainability**: Clear structure, well-documented patterns
10. **Reproducibility**: Full provenance tracking

### Negative

1. **Disk Space**: Storing raw + processed + CSV + Parquet increases storage needs
2. **Complexity**: Two-stage pipeline adds conceptual overhead
3. **Processing Time**: Validation and multi-format output add ~20-30% overhead
4. **Redundancy**: Some logic duplicated across processors
5. **Dependencies**: Requires `pyarrow` for Parquet support
6. **Learning Curve**: New developers must understand pipeline stages

### Risks and Mitigations

**Risk**: Raw data sources change format, breaking fetch modules
- **Mitigation**: Version raw data files with date stamps
- **Mitigation**: Flexible column recognition handles naming variations
- **Mitigation**: Comprehensive error messages guide debugging

**Risk**: Processors produce invalid data that passes validation
- **Mitigation**: Multi-level validation (structure, values, plausibility)
- **Mitigation**: Compare processed data to published statistics
- **Mitigation**: Visual inspection of results (age curves, distributions)

**Risk**: Disk space exhaustion from storing multiple formats
- **Mitigation**: Parquet compression reduces size
- **Mitigation**: Can delete raw data after processing (keep metadata)
- **Mitigation**: Periodic cleanup of old vintages

**Risk**: Metadata files become stale or inaccurate
- **Mitigation**: Auto-generate metadata during processing (not manual)
- **Mitigation**: Include processing timestamp and version
- **Mitigation**: Validation checks metadata against data

## Alternatives Considered

### Alternative 1: Single-Stage Pipeline (Fetch and Process Combined)

**Description**: Combine fetching and processing into single modules.

```python
def fetch_and_process_fertility():
    # Download from SEER
    raw_data = download_seer_data()
    # Immediately process
    processed = process_fertility(raw_data)
    return processed
```

**Pros**:
- Simpler architecture (one stage)
- Fewer intermediate files

**Cons**:
- Must re-fetch to re-process
- Can't develop/test processors offline
- Tight coupling to data sources
- Harder to debug (fetch vs process issues)

**Why Rejected**:
- Separation of concerns is valuable
- Re-processing is common (new algorithms, bug fixes)
- Offline development important

### Alternative 2: Database-Backed Pipeline

**Description**: Store all data in PostgreSQL/SQLite.

**Pros**:
- Centralized data management
- SQL queries for data exploration
- ACID guarantees

**Cons**:
- Adds database dependency
- Slower than file-based for this use case
- Harder to version control
- More complex deployment

**Why Rejected**:
- File-based pipeline sufficient
- Parquet + DataFrames are fast
- Git version control preferred
- Simpler for single-user desktop application

### Alternative 3: Workflow Engine (Airflow, Prefect, Luigi)

**Description**: Use workflow orchestration tool for pipeline.

**Pros**:
- Automatic dependency management
- Retry logic, monitoring, scheduling
- DAG visualization

**Cons**:
- Adds significant complexity
- Overkill for simple linear pipeline
- Harder to set up and maintain
- Not needed for manual/ad-hoc runs

**Why Rejected**:
- Pipeline is simple enough for manual orchestration
- No scheduling requirement (annual updates)
- Can add later if needed

### Alternative 4: Streaming/Real-Time Pipeline

**Description**: Process data as it arrives (stream processing).

**Pros**:
- Real-time updates
- Lower latency

**Cons**:
- Much more complex
- Demographic data updated annually (not real-time)
- Overkill for batch projections

**Why Rejected**:
- Batch processing matches data update frequency
- No real-time requirement
- Simpler is better

### Alternative 5: NoSQL/Document Storage (MongoDB)

**Description**: Store data as JSON documents in MongoDB.

**Pros**:
- Flexible schema
- Good for heterogeneous data

**Cons**:
- Adds database dependency
- Slower for analytical queries
- Not designed for tabular data
- Overkill for structured data

**Why Rejected**:
- Data is inherently tabular (age × sex × race)
- Pandas/Parquet designed for this
- No schema flexibility needed

## Implementation Notes

### Module Organization

**Fetch Modules** (`cohort_projections/data/fetch/`):
- `census_api.py`: Census API client (PEP, ACS, geographic data)
- `vital_stats.py`: SEER/NVSS data fetcher (planned)
- `migration_data.py`: IRS/ACS migration data fetcher (planned)
- `geographic.py`: FIPS codes and geographic reference (planned)

**Process Modules** (`cohort_projections/data/process/`):
- `base_population.py`: Census population → cohort matrix
- `fertility_rates.py`: SEER data → age-specific fertility rates
- `survival_rates.py`: Life tables → survival rates
- `migration_rates.py`: IRS/ACS → net migration rates (planned)

### Common Utilities

**Shared Functions** (`cohort_projections/data/`):
```python
# Race/ethnicity harmonization
def harmonize_race_categories(df, mapping=RACE_ETHNICITY_MAP)

# Column name standardization
def standardize_column_names(df)

# Multi-format loading
def load_data_file(path, format=None)

# Validation helpers
def validate_dataframe_schema(df, required_columns, dtypes)

# Metadata generation
def generate_metadata(df, source_info, transformations, quality_metrics)
```

### Example Usage

**Fetch + Process Fertility Data**:
```bash
# Step 1: Fetch raw data (manual or scripted)
python scripts/data_pipeline/01_fetch_vital_stats.py

# Step 2: Process into projection format
python scripts/data_pipeline/02_process_fertility_rates.py
```

**Programmatic Usage**:
```python
from cohort_projections.data.process import process_fertility_rates

# Process fertility data
fertility_df = process_fertility_rates(
    input_path='data/raw/fertility/seer_fertility_2018_2022.csv',
    output_path='data/processed/fertility/fertility_rates',
    config=config
)

# Result: Creates 3 files
# - data/processed/fertility/fertility_rates.parquet
# - data/processed/fertility/fertility_rates.csv
# - data/processed/fertility/fertility_rates_metadata.json
```

## References

1. **Data Pipeline Patterns**: "Data Pipelines with Apache Airflow" - Harenslak & de Ruiter (2021)
2. **Parquet Format**: https://parquet.apache.org/docs/
3. **Pandas Best Practices**: "Pandas for Data Analysis" - McKinney (2022)
4. **Data Quality**: "Data Quality: The Accuracy Dimension" - Redman (1996)
5. **Census Data Standards**: Census Bureau Demographic Data Standards

## Revision History

- **2025-12-18**: Initial version (ADR-006) - Data pipeline architecture

## Related ADRs

- ADR-001: Fertility rate processing (process stage implementation)
- ADR-002: Survival rate processing (process stage implementation)
- ADR-003: Migration rate processing (planned)
- ADR-005: Configuration management (pipeline configuration)
- ADR-007: Race and ethnicity categorization (harmonization)
- ADR-008: BigQuery integration (fetch stage data source)
- ADR-012: Output formats (similar to processed data formats)
