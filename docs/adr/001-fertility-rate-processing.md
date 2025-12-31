# ADR-001: Fertility Rate Processing Methodology

## Status
Accepted

## Date
2025-12-18

## Last Reviewed
2025-12-31

## Context

The cohort component projection engine requires age-specific fertility rates (ASFR) by race/ethnicity for women of reproductive age (15-49). These rates must be derived from raw SEER (Surveillance, Epidemiology, and End Results Program) or NVSS (National Vital Statistics System) data files, which come in various formats and use inconsistent coding schemes.

### Challenges

1. **Data Format Variations**: SEER data comes in multiple formats (CSV, TXT, Excel) with inconsistent column naming
2. **Annual Fluctuations**: Single-year fertility rates can be volatile due to small sample sizes, especially for minority populations
3. **Missing Data**: Not all age-race combinations have data, particularly for smaller populations
4. **Race Category Mapping**: SEER uses different race codes than our standardized 6-category system
5. **Validation Requirements**: Rates must be plausible and complete for projection engine

### Requirements

- Input: SEER/NVSS fertility data with age, race, year, fertility rates
- Output: Complete age (15-49) × race (6 categories) matrix ready for projection engine
- Data quality: Smooth, plausible rates with no missing combinations
- Traceability: Document data sources, averaging periods, and processing decisions

## Decisions

### Decision 1: Multi-Year Averaging with Population Weighting

**Decision**: Use 5-year weighted average (default) based on female population size.

**Rationale**:
- **Stability**: Reduces year-to-year volatility, especially for small populations
- **Statistical Validity**: Larger sample sizes improve rate estimates
- **Demographic Standard**: 5 years aligns with Census ACS and standard practice
- **Flexibility**: Configurable averaging period for different use cases

**Implementation**:
```python
if 'population' column exists:
    weighted_rate = sum(rate × population) / sum(population)
else:
    simple_average = mean(rate)
```

**Alternatives Considered**:
- Simple mean: Simpler but gives equal weight to small and large populations
- Single-year rates: Too volatile for projections
- Trend-based extrapolation: Too complex for base rates; handled in scenario module

### Decision 2: Zero Fill for Missing Age-Race Combinations

**Decision**: Fill missing age-race combinations with `fertility_rate = 0.0` (not imputation).

**Rationale**:
- **Conservative**: Zero is safer than imputed values that may be wrong
- **Transparency**: Missing data clearly visible in validation
- **Projection Safety**: Projection engine handles zeros correctly (no births)
- **Rare Occurrence**: Most combinations should have data; zeros indicate true data gaps

**Why Not Imputation**:
- Risk of introducing bias
- Complexity of choosing appropriate imputation method
- Age-race patterns vary significantly (not suitable for simple interpolation)
- Better to flag missing data than hide it with synthetic values

**Validation**: Log warnings for any age-race combinations with zero rates

### Decision 3: SEER Race Code Mapping with Strict Validation

**Decision**: Use explicit mapping dictionary (`SEER_RACE_ETHNICITY_MAP`) and drop unmapped codes.

**Rationale**:
- **Consistency**: Ensures all data uses standard 6 categories
- **Traceability**: Explicit mapping is auditable
- **Error Detection**: Unmapped codes indicate data issues or new categories
- **Maintainability**: Easy to update mapping as SEER codes change

**Standard Categories** (from `projection_config.yaml`):
1. White alone, Non-Hispanic
2. Black alone, Non-Hispanic
3. AIAN alone, Non-Hispanic (American Indian/Alaska Native)
4. Asian/PI alone, Non-Hispanic (Asian/Pacific Islander)
5. Two or more races, Non-Hispanic
6. Hispanic (any race)

**Handling Unmapped Codes**:
- Log warning with unmapped code values
- Drop records (not silently fail)
- Suggest updating mapping dictionary

### Decision 4: Plausibility Thresholds for Validation

**Decision**: Implement multi-level validation with errors and warnings.

**Thresholds**:
- **Error** (fails validation):
  - Negative rates
  - Missing required ages (15-49)
  - Missing required race categories
  - Rate > 0.15 (150 births per 1000 women - biological implausibility)

- **Warning** (passes with flag):
  - Rate > 0.13 (unusual but possible)
  - TFR < 1.0 or > 3.0 (outside typical range for developed countries)
  - Zero rates for entire race-age combinations

**Total Fertility Rate (TFR) Validation**:
- Calculate TFR = sum of ASFRs for each race
- Typical range: 1.3-2.5 for U.S. populations
- Log warnings outside this range but don't fail validation

**Rationale**:
- Catches data errors before they reach projection engine
- Demographic plausibility based on established literature
- Warnings allow edge cases while flagging unusual patterns

### Decision 5: File Format Support and Column Name Flexibility

**Decision**: Support multiple formats (CSV, TXT, Excel, Parquet) with flexible column naming.

**Supported Formats**:
- CSV (comma-delimited)
- TXT (tab-delimited, common for SEER)
- Excel (.xlsx, .xls)
- Parquet (for processed/intermediate data)

**Flexible Column Recognition**:
- Age: `age`, `age_group`, `age_of_mother`
- Race: `race_ethnicity`, `race`, `race_origin`, `origin`, `race_code`
- Rate: `fertility_rate`, `asfr`, `rate`

**Rationale**:
- SEER data formats vary by source and year
- Reduces need for manual preprocessing
- Case-insensitive matching handles inconsistent naming
- Fail with clear error if required columns truly missing

### Decision 6: Metadata Generation and Provenance

**Decision**: Generate JSON metadata file with every processing run.

**Metadata Includes**:
- Processing date/time
- Source file path
- Year range used
- Averaging period
- TFR by race (quality check)
- Validation warnings
- Configuration parameters used

**Rationale**:
- **Reproducibility**: Can recreate processing from metadata
- **Quality Assurance**: TFR values allow quick sanity check
- **Auditing**: Track which source data produced which outputs
- **Debugging**: Easier to diagnose issues with full provenance

**File Outputs**:
- `fertility_rates.parquet` (primary, efficient storage)
- `fertility_rates.csv` (human-readable backup)
- `fertility_rates_metadata.json` (provenance)

## Consequences

### Positive

1. **Robust Processing**: Handles varied input formats with minimal manual intervention
2. **Quality Assurance**: Multi-level validation catches errors early
3. **Demographic Validity**: Averaged rates are stable and plausible for projections
4. **Transparency**: Metadata and logging provide full audit trail
5. **Maintainability**: Modular functions allow testing and reuse
6. **Consistency**: Follows same patterns as `base_population.py` processor

### Negative

1. **Zero Fills**: Missing data filled with zeros may need manual review/correction
2. **Dropped Records**: Unmapped race codes require updating mapping dictionary
3. **Complexity**: Multi-step pipeline requires understanding all components
4. **Storage**: Three output files (parquet, CSV, JSON) per processing run

### Risks and Mitigations

**Risk**: Zero fertility rates for valid age-race combinations
- **Mitigation**: Validation warnings flag any zero-rate combinations
- **Action**: Review warnings and verify if zeros are appropriate

**Risk**: SEER race codes change between data releases
- **Mitigation**: Explicit mapping with clear error messages for unmapped codes
- **Action**: Update `SEER_RACE_ETHNICITY_MAP` when new codes appear

**Risk**: Averaging period too short/long for volatile populations
- **Mitigation**: Configurable averaging period (default 5 years)
- **Action**: Adjust based on data availability and population size

## Alternatives Considered

### Alternative 1: Impute Missing Rates Instead of Zero Fill

**Not Chosen Because**:
- Imputation methods (mean, interpolation, regression) introduce assumptions
- Risk of systematic bias in projections
- Complexity not warranted given rarity of missing data
- Zero-fill is transparent and conservative

**When to Reconsider**: If missing data is >10% of combinations

### Alternative 2: Single-Year Rates with Trend Modeling

**Not Chosen Because**:
- Trend modeling belongs in scenario module, not data processing
- Base rates should be neutral, stable estimates
- Adds complexity without clear benefit
- Volatile rates problematic for validation

**When to Reconsider**: If doing trend-based scenario projections

### Alternative 3: Hierarchical Averaging (State → County → Place)

**Not Chosen Because**:
- State-level fertility rates are the primary use case
- County/place rates often unavailable or unreliable
- Can implement later if needed
- KISS principle: keep initial implementation simple

**When to Reconsider**: If county-specific projections require county-specific fertility rates

### Alternative 4: Bayesian Small Area Estimation

**Not Chosen Because**:
- Significant statistical complexity
- Requires additional population data and priors
- Overkill for state-level projections
- Expertise and validation burden

**When to Reconsider**: For small counties or places with very limited data

## Implementation Notes

### Key Functions

1. `load_seer_fertility_data()`: Multi-format loader with year filtering
2. `harmonize_fertility_race_categories()`: Maps SEER codes to standard categories
3. `calculate_average_fertility_rates()`: Weighted averaging over years
4. `create_fertility_rate_table()`: Complete age × race matrix construction
5. `validate_fertility_rates()`: Plausibility checks and TFR calculation
6. `process_fertility_rates()`: Main pipeline orchestrator

### Configuration Integration

Uses `projection_config.yaml` for:
- `demographics.race_ethnicity.categories`: Expected race categories
- `rates.fertility.apply_to_ages`: Reproductive age range (default [15, 49])
- `rates.fertility.averaging_period`: Years to average (default 5)
- `output.compression`: Parquet compression method
- `logging.*`: Logging configuration

### Testing Strategy

**Unit Tests** (recommended):
- Test race code mapping with known codes
- Test averaging with sample data
- Test validation with edge cases (zeros, high rates, missing data)
- Test file format loading

**Integration Tests** (recommended):
- End-to-end processing with synthetic SEER-format data
- Verify output format matches projection engine requirements
- Confirm metadata generation

**Manual Validation** (required):
- Compare TFRs to published CDC/NVSS statistics
- Visual inspection of age patterns (should peak ~age 28)
- Check race-specific patterns match demographic literature

## References

1. **SEER Program**: https://seer.cancer.gov/popdata/methods.html
2. **CDC NVSS Fertility Reports**: https://www.cdc.gov/nchs/products/databriefs/db-fertility.htm
3. **UN Demographic Methods**: "Manual X: Indirect Techniques for Demographic Estimation"
4. **Census Bureau Methodology**: "Methodology for the U.S. Population Projections"
5. **State Demographer Resources**: AAPOR guidelines for fertility rate estimation

## Revision History

- **2025-12-18**: Initial version (ADR-001) - Fertility rate processing methodology

## Related ADRs

- ADR-002: Mortality rate processing (planned)
- ADR-003: Migration rate processing (planned)
- ADR-004: Projection scenario design (planned)
