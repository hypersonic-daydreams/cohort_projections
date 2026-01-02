# ADR-013: Multi-Geography Projection Design

## Status
Accepted

## Date
2025-12-18

## Context

The North Dakota Population Projection System needs to run cohort-component projections for multiple geographies (state, counties, places) efficiently and consistently. This requires careful design of the orchestration layer that coordinates projections across many geographic units.

### Requirements

1. **Multiple Geographic Levels**: Support state (1), county (53), and place (406) level projections
2. **Parallel Processing**: Enable parallel execution for performance gains
3. **Geographic Hierarchy**: Maintain and validate aggregation relationships
4. **Flexible Configuration**: Allow user control over which geographies to project
5. **Geography Reference Data**: Load and manage FIPS codes and geographic attributes
6. **Rate Management**: Handle geography-specific demographic rates
7. **Validation**: Ensure aggregated results are consistent with component geographies
8. **Scalability**: Process 406+ geographies efficiently (< 10 minutes for all)
9. **Error Handling**: Robust handling of missing data or failed projections
10. **Metadata Tracking**: Record processing details for each geography

### Technical Context

**Existing Infrastructure**:
- Core projection engine (`CohortComponentProjection` class) processes single geography
- Data processors generate rates by geography
- Configuration system via YAML
- Logging and validation utilities available

**Performance Considerations**:
- Single geography projection: ~1-3 seconds
- All 53 counties: ~3 minutes serial, ~30 seconds parallel (8 cores)
- All 406 places: ~20 minutes serial, ~3 minutes parallel (8 cores)

**Geographic Reference Data**:
- Census TIGER files provide authoritative FIPS codes
- Need county-place mapping for aggregation
- Names change over time; FIPS codes stable

## Decision

### Decision 1: Separate Geography Loader and Projection Orchestrator

**Decision**: Create two distinct modules:
- `geography_loader.py`: Load and manage geographic reference data
- `multi_geography.py`: Orchestrate multi-geography projections

**Rationale**:
- **Separation of Concerns**: Reference data management vs. projection execution
- **Reusability**: Geography loader used independently for analysis
- **Testing**: Easier to test each component separately
- **Clarity**: Clear boundaries between data loading and processing

**Implementation**:
```python
# geography_loader.py - Load geographic reference data
load_nd_counties() -> DataFrame with county FIPS/names
load_nd_places() -> DataFrame with place FIPS/names
get_place_to_county_mapping() -> Maps places to counties

# multi_geography.py - Run projections
run_single_geography_projection() -> Run one geography
run_multi_geography_projections() -> Run many geographies
aggregate_to_county() -> Roll up places to county
aggregate_to_state() -> Roll up counties to state
validate_aggregation() -> Check consistency
```

### Decision 2: Support Both Serial and Parallel Processing

**Decision**: Implement both serial and parallel execution modes with configuration control.

**Parallel Processing Design**:
- Use Python's `concurrent.futures.ProcessPoolExecutor`
- Default: `max_workers = min(cpu_count(), num_geographies)`
- User-configurable parallelism
- Serial mode for debugging or resource-constrained environments

**Rationale**:

**Why Both Modes**:
- **Parallel**: Fast (3 min vs 20 min for all places)
- **Serial**: Easier debugging, lower memory, simpler logs
- **Flexibility**: Users choose based on needs

**Why ProcessPoolExecutor**:
- True parallelism (vs threading GIL limitations)
- Simple API
- Good for CPU-bound tasks
- Handles state isolation automatically

**Configuration**:
```yaml
geographic:
  parallel_processing:
    enabled: true
    max_workers: 8  # null = auto-detect
    chunk_size: 10  # geographies per batch
```

**Trade-offs**:
- Parallel uses more memory (each process loads rates)
- Parallel logs harder to follow (interleaved)
- Serial is simpler but much slower

### Decision 3: Geography-Specific Rate Files (Not Implemented Initially)

**Decision**: Design for geography-specific rates, but use state rates initially.

**Future Design**:
```python
# Geography-specific rate paths
rates/
  fertility/
    state_38_fertility.parquet
    county_38101_fertility.parquet  # Cass County
  survival/
    state_38_survival.parquet
  migration/
    county_38101_migration.parquet  # Different migration per county
```

**Initial Implementation**:
- All geographies use same state-level rates
- Migration is only rate that varies by geography (if available)
- Fertility and survival same across ND

**Rationale**:

**Why Design for Variation**:
- Migration definitely varies by geography
- Future: County-specific fertility/mortality from ACS
- Flexibility for research scenarios

**Why Start Simple**:
- State rates sufficient for most use cases
- Avoids data collection burden initially
- Can add geography-specific rates later without redesign

### Decision 4: Hierarchical Aggregation with Validation

**Decision**: Implement aggregation functions that roll up from places → counties → state with validation at each level.

**Aggregation Logic**:
```python
# County = sum of places within county (partial coverage)
county_projection = aggregate_to_county(place_projections)

# State = sum of all counties (complete coverage)
state_projection = aggregate_to_state(county_projections)

# Validation
validate_aggregation(
    component_projections=place_projections,
    aggregated_projection=county_projection,
    tolerance=0.01  # 1% tolerance for rounding
)
```

**Validation Checks**:
1. Sum of component populations matches aggregate (within tolerance)
2. All component geographies accounted for
3. No double-counting
4. Age-sex-race distributions preserved

**Rationale**:

**Why Hierarchical**:
- Matches geographic reality
- Enables urban/rural analysis
- Validates consistency
- Standard practice in demography

**Why Tolerances**:
- Floating-point rounding errors accumulate
- 1% tolerance reasonable for 400+ geographies
- Warn on discrepancies, error on large differences

**Handling Unincorporated Areas**:
- Places don't cover entire county
- Unincorporated = County - Sum(Places)
- Can be positive (rural areas) or zero (fully incorporated)
- Not directly projected; calculated as residual

### Decision 5: Census TIGER Files as Primary Geographic Reference

**Decision**: Use Census TIGER/Line files as authoritative source for FIPS codes and names.

**Data Sources**:
```python
# Primary: Census TIGER files (via tigris or direct download)
# - Most authoritative
# - Includes geographic boundaries (future use)
# - Vintage-specific (e.g., 2020 geography)

# Fallback: Local CSV reference files
# - Manually curated if needed
# - Simpler but requires maintenance
```

**Reference Data Structure**:
```python
# Counties
counties_df = pd.DataFrame({
    'state_fips': ['38', '38', ...],
    'county_fips': ['38101', '38015', ...],  # 5 digits
    'county_name': ['Cass County', 'Burleigh County', ...],
    'population_2020': [191000, 98500, ...]  # For filtering
})

# Places
places_df = pd.DataFrame({
    'state_fips': ['38', '38', ...],
    'place_fips': ['3825700', '3807200', ...],  # 7 digits
    'place_name': ['Fargo city', 'Bismarck city', ...],
    'county_fips': ['38101', '38015', ...],  # Containing county
    'population_2020': [125000, 73500, ...]
})
```

**Rationale**:

**Why TIGER**:
- Authoritative federal standard
- Includes boundary files (future spatial analysis)
- Well-documented and maintained
- Free and public

**Why Local Fallback**:
- Works without internet
- Faster loading
- Can customize (e.g., exclude territories)

**Vintage Considerations**:
- Use same vintage as base population (e.g., 2020)
- Document vintage in metadata
- Handle FIPS changes with crosswalks if needed

### Decision 6: Configuration-Driven Geography Selection

**Decision**: Use flexible configuration to specify which geographies to project.

**Configuration Options**:
```yaml
geographic:
  state: "38"  # North Dakota FIPS

  counties:
    mode: "all"  # "all", "list", or "threshold"
    # mode: "list"
    # fips_codes: ["38101", "38015", "38035"]

  places:
    mode: "threshold"  # Filter by population
    min_population: 500
    # mode: "all"  # All 406 places
    # mode: "list"
    # fips_codes: ["3825700", "3807200", "3833900", "3841500"]
```

**Rationale**:

**Flexibility**:
- Researchers may want all places
- Planners may want major cities only
- Testing may need small subset

**Population Threshold**:
- Small places (<100) have volatile projections
- Threshold filters to stable places
- Typical: 500+ population (150 places)

**List Mode**:
- Specific cities for detailed analysis
- Custom geographic focus
- Testing and development

### Decision 7: Metadata and Results Tracking

**Decision**: Generate comprehensive metadata for each geography projection run.

**Metadata Structure**:
```python
metadata = {
    'geography': {
        'level': 'county',
        'fips': '38101',
        'name': 'Cass County',
        'population_2025': 195000
    },
    'projection': {
        'base_year': 2025,
        'end_year': 2045,
        'scenario': 'baseline',
        'processing_time_seconds': 2.3
    },
    'rates_used': {
        'fertility': 'data/processed/fertility/state_38_fertility.parquet',
        'survival': 'data/processed/survival/state_38_survival.parquet',
        'migration': 'data/processed/migration/county_38101_migration.parquet'
    },
    'summary_statistics': {
        'base_population': 195000,
        'final_population': 210000,
        'total_births': 45000,
        'total_deaths': 35000,
        'net_migration': 5000,
        'growth_rate': 0.077  # 7.7% over 20 years
    },
    'validation': {
        'all_checks_passed': True,
        'negative_populations': 0,
        'extreme_growth_rates': []
    }
}
```

**Rationale**:
- Track processing details
- Enable quality control
- Support reproducibility
- Document rate sources
- Summary statistics for quick review

### Decision 8: Error Handling Strategy

**Decision**: Implement robust error handling with continue-on-error option.

**Error Handling Modes**:
```yaml
geographic:
  error_handling:
    mode: "continue"  # "continue" or "fail_fast"
    log_failed_geographies: true
    retry_failed: false
```

**Error Scenarios**:
1. **Missing rate data**: Log warning, skip geography, continue
2. **Projection failure**: Log error, skip geography, continue
3. **Validation failure**: Log warning, save results with flag, continue
4. **File I/O error**: Log error, skip geography, continue

**Rationale**:

**Why Continue-on-Error**:
- One bad geography shouldn't stop all projections
- Long-running batch jobs should complete
- Failed geographies can be re-run individually
- Log provides full diagnostic information

**When to Fail-Fast**:
- Critical rate files missing (all geographies affected)
- Configuration errors
- System resource errors
- Development/debugging mode

### Decision 9: Output File Organization

**Decision**: Organize projection outputs by geographic level with standardized naming.

**Directory Structure**:
```
data/
  output/
    projections/
      state/
        nd_state_38_projection_2025_2045_baseline.parquet
        nd_state_38_summary_2025_2045_baseline.csv
        nd_state_38_metadata_2025_2045_baseline.json
      counties/
        nd_county_38101_projection_2025_2045_baseline.parquet  # Cass
        nd_county_38015_projection_2025_2045_baseline.parquet  # Burleigh
        ...
        counties_metadata.json  # Summary of all counties
      places/
        nd_place_3825700_projection_2025_2045_baseline.parquet  # Fargo
        nd_place_3807200_projection_2025_2045_baseline.parquet  # Bismarck
        ...
        places_metadata.json  # Summary of all places
```

**Naming Convention**:
`{state}_{level}_{fips}_projection_{base_year}_{end_year}_{scenario}.{format}`

**Rationale**:

**Geographic Separation**:
- Organized by level for easy navigation
- Prevents filename clashes
- Enables level-specific analysis

**Standardized Naming**:
- Sortable and predictable
- Contains key metadata in filename
- Enables programmatic file discovery
- Glob patterns work easily

**Multiple Formats**:
- Parquet: Primary (efficient, typed)
- CSV: Summary statistics (human-readable)
- JSON: Metadata (structured documentation)

### Decision 10: Progress Tracking and Logging

**Decision**: Implement progress bars and detailed logging for long-running jobs.

**Progress Tracking**:
```python
from tqdm import tqdm

for geography in tqdm(geographies, desc="Projecting counties"):
    run_single_geography_projection(geography)
```

**Logging Levels**:
- **DEBUG**: Detailed projection steps (each year, component)
- **INFO**: Geography start/complete, summary statistics
- **WARNING**: Validation issues, missing data
- **ERROR**: Projection failures, critical issues

**Parallel Logging**:
- Each worker logs to separate file: `logs/projection_worker_{pid}.log`
- Main process aggregates to: `logs/multi_geography_projection.log`
- Console shows progress bar + high-level info

**Rationale**:
- Long jobs need progress feedback
- Debugging requires detailed logs
- Parallel execution complicates logging
- Users want to see completion estimates

## Consequences

### Positive

1. **Scalable**: Handles 1-1000+ geographies efficiently
2. **Flexible**: Configuration-driven selection of geographies
3. **Fast**: Parallel processing reduces runtime 10x
4. **Robust**: Continue-on-error prevents single failures from stopping job
5. **Validated**: Hierarchical aggregation checks ensure consistency
6. **Well-Documented**: Comprehensive metadata for each run
7. **Maintainable**: Clear separation between loader and orchestrator
8. **Reusable**: Geography loader used independently
9. **Standard Practice**: Follows demographic projection conventions
10. **Extensible**: Design supports future geography-specific rates

### Negative

1. **Complexity**: Two modes (serial/parallel) increase code complexity
2. **Memory**: Parallel mode uses significantly more memory
3. **Debugging**: Parallel execution harder to debug
4. **Log Volume**: Detailed logging generates large files
5. **Rate Files**: Current implementation uses single rate set (simplification)
6. **Unincorporated Areas**: Not directly projected, calculated as residual
7. **TIGER Dependency**: Requires Census data access (mitigated by local fallback)

### Risks and Mitigations

**Risk**: Parallel processing causes memory exhaustion
- **Mitigation**: Configurable `max_workers` and `chunk_size`
- **Mitigation**: Monitor memory usage and adjust
- **Mitigation**: Serial mode fallback

**Risk**: One bad rate file affects all geographies
- **Mitigation**: Validate rate files before starting batch
- **Mitigation**: Early detection of missing files
- **Mitigation**: Clear error messages

**Risk**: Aggregation validation fails due to rounding
- **Mitigation**: Configurable tolerance (default 1%)
- **Mitigation**: Warn but don't fail on small discrepancies
- **Mitigation**: Log discrepancies for review

**Risk**: FIPS codes change between vintages
- **Mitigation**: Document vintage used
- **Mitigation**: Include crosswalk files if needed
- **Mitigation**: Use consistent vintage throughout project

**Risk**: Output files become very large (406 places × 1092 cohorts × 21 years)
- **Mitigation**: Use Parquet with compression (10x smaller than CSV)
- **Mitigation**: Option to exclude zero-population cohorts
- **Mitigation**: Separate summary files for quick access

## Alternatives Considered

### Alternative 1: Single Module for Everything

**Description**: Combine geography loader and projection orchestrator into one module.

**Pros**:
- Simpler (one file vs two)
- Less imports

**Cons**:
- Mixing concerns
- Harder to test
- Less reusable

**Why Rejected**: Violates separation of concerns, reduces reusability.

### Alternative 2: Threading Instead of Multiprocessing

**Description**: Use `ThreadPoolExecutor` instead of `ProcessPoolExecutor`.

**Pros**:
- Lower memory overhead
- Easier to debug
- Simpler state sharing

**Cons**:
- Python GIL limits parallelism for CPU-bound work
- No actual speedup (projections are CPU-bound)
- Same complexity without benefits

**Why Rejected**: No performance benefit due to GIL, projections are CPU-intensive.

### Alternative 3: Database for Geographic Reference Data

**Description**: Store FIPS codes, names in SQLite/PostgreSQL database.

**Pros**:
- Query flexibility
- Relational structure
- Transactional integrity

**Cons**:
- Adds dependency
- Overkill for static reference data
- Slower than in-memory DataFrames
- Complicates deployment

**Why Rejected**: Reference data is small and static, CSV/Parquet sufficient.

### Alternative 4: Dask for Parallel Processing

**Description**: Use Dask for distributed parallel computation.

**Pros**:
- More sophisticated scheduling
- Handles larger-than-memory data
- Dashboard for monitoring

**Cons**:
- Heavy dependency
- Overkill for this scale
- More complex setup
- Harder to deploy

**Why Rejected**: `concurrent.futures` sufficient for 400 geographies, simpler.

### Alternative 5: Always Project Unincorporated Areas

**Description**: Directly project unincorporated areas as separate geography.

**Pros**:
- Complete coverage
- Direct projection (not residual)
- Better for rural areas

**Cons**:
- No demographic rates for "unincorporated"
- Geographic definition unclear
- Would use county rates anyway
- Adds complexity

**Why Rejected**: Unincorporated is better calculated as residual; no data advantage.

### Alternative 6: Mandatory Geography-Specific Rates

**Description**: Require geography-specific fertility, mortality, migration rates for each geography.

**Pros**:
- Maximum accuracy
- Captures local variation
- Research ideal

**Cons**:
- Massive data collection burden
- Sample sizes too small for stable rates (most places)
- State rates sufficient for most purposes
- Not operationally feasible

**Why Rejected**: Data availability insufficient, state rates adequate for initial implementation.

## Implementation Notes

### Module Structure

```
cohort_projections/
  geographic/
    __init__.py          # Exports main functions
    geography_loader.py  # Load FIPS codes, names, mappings
    multi_geography.py   # Orchestrate projections, aggregation
    README.md           # Module documentation
```

### Key Functions

**geography_loader.py**:
```python
def load_nd_counties(source='tiger', vintage=2020) -> pd.DataFrame
def load_nd_places(source='tiger', vintage=2020) -> pd.DataFrame
def get_place_to_county_mapping() -> pd.DataFrame
def load_geography_list(level, config) -> List[str]  # FIPS codes
def get_geography_name(fips) -> str
```

**multi_geography.py**:
```python
def run_single_geography_projection(fips, rates, config) -> dict
def run_multi_geography_projections(geographies, config) -> dict
def aggregate_to_county(place_projections) -> pd.DataFrame
def aggregate_to_state(county_projections) -> pd.DataFrame
def validate_aggregation(components, aggregate, tolerance) -> dict
```

### Performance Targets

- **Single County**: < 3 seconds
- **All 53 Counties (Parallel)**: < 1 minute
- **All 406 Places (Parallel)**: < 5 minutes
- **Memory (Parallel)**: < 8 GB with 8 workers

### Testing Strategy

1. **Unit Tests**: Each function separately
2. **Integration Tests**: Full multi-geography pipeline
3. **Performance Tests**: Timing benchmarks
4. **Validation Tests**: Aggregation consistency
5. **Error Handling Tests**: Missing data, failures

### Dependencies

- `pandas`: DataFrames
- `numpy`: Numerical operations
- `concurrent.futures`: Parallel processing
- `tqdm`: Progress bars
- `yaml`: Configuration
- `pathlib`: Path handling
- Existing: `CohortComponentProjection`, config_loader, logger

## References

1. **Census TIGER/Line Files**: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
2. **FIPS Codes**: https://www.census.gov/library/reference/code-lists/ansi.html
3. **Multiprocessing Best Practices**: Python concurrent.futures documentation
4. **Demographic Projection Standards**: Smith, Tayman & Swanson (2001), Chapter 8

## Revision History

- **2025-12-18**: Initial version (ADR-013) - Multi-geography projection design

## Related ADRs

- ADR-004: Core projection engine (single-geography processing)
- ADR-010: Geographic scope and granularity (what geographies to support)
- ADR-006: Data pipeline (rate file formats and loading)
- ADR-012: Output formats (file organization and formats)
