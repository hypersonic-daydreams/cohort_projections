# Geographic Module

Multi-geography cohort projection orchestrator for the North Dakota Population Projection System.

## Overview

This module enables running cohort-component population projections for multiple geographies efficiently with support for:

- **Multiple Geographic Levels**: State, county (53), and place (406) projections
- **Parallel Processing**: Run projections concurrently for significant speedup
- **Hierarchical Aggregation**: Roll up places → counties → state with validation
- **Flexible Configuration**: Control which geographies to project via YAML config
- **Geographic Reference Data**: Load FIPS codes, names, and relationships
- **Error Handling**: Robust continue-on-error processing for batch jobs
- **Metadata Tracking**: Comprehensive logging and result documentation

## Architecture

### Module Structure

```
cohort_projections/geographic/
├── __init__.py              # Module exports
├── geography_loader.py      # Geographic reference data loading
├── multi_geography.py       # Multi-geography projection orchestration
└── README.md               # This file
```

### Key Components

#### 1. Geography Loader (`geography_loader.py`)

Loads and manages geographic reference data (FIPS codes, names, relationships):

- `load_nd_counties()`: Load all 53 North Dakota counties
- `load_nd_places()`: Load all 406 incorporated places
- `get_place_to_county_mapping()`: Map places to containing counties
- `load_geography_list()`: Get list of FIPS codes to project
- `get_geography_name()`: Convert FIPS code to human-readable name

#### 2. Multi-Geography Orchestrator (`multi_geography.py`)

Orchestrates projections across multiple geographies:

- `run_single_geography_projection()`: Run projection for one geography
- `run_multi_geography_projections()`: Run projections for many geographies
- `aggregate_to_county()`: Roll up place projections to county level
- `aggregate_to_state()`: Roll up county projections to state level
- `validate_aggregation()`: Ensure aggregated totals match components

## Usage

### Basic Example: Project All Counties

```python
from cohort_projections.geographic import (
    load_geography_list,
    run_multi_geography_projections
)

# Load list of counties to project
counties = load_geography_list('county')
print(f"Projecting {len(counties)} counties")  # 53

# Prepare data (one DataFrame per county)
base_populations = {fips: load_county_population(fips) for fips in counties}
migration_rates = {fips: load_county_migration(fips) for fips in counties}

# Run projections (parallel processing)
results = run_multi_geography_projections(
    level='county',
    base_population_by_geography=base_populations,
    fertility_rates=nd_fertility_rates,  # Shared across counties
    survival_rates=nd_survival_rates,    # Shared across counties
    migration_rates_by_geography=migration_rates,  # Vary by county
    parallel=True,
    max_workers=8
)

# Access results
print(f"Successful: {results['metadata']['successful']}")
print(f"Failed: {results['metadata']['failed']}")
print(results['summary'])  # Summary DataFrame
```

### Advanced Example: Places with Population Threshold

```python
from cohort_projections.geographic import (
    load_nd_places,
    run_multi_geography_projections
)

# Load places with population >= 500
places = load_nd_places(min_population=500)
print(f"Projecting {len(places)} places")  # ~150 places

# Configure via config YAML
config = {
    'geographic': {
        'places': {
            'mode': 'threshold',
            'min_population': 500
        },
        'parallel_processing': {
            'enabled': True,
            'max_workers': 8
        }
    }
}

# Run projections
results = run_multi_geography_projections(
    level='place',
    base_population_by_geography=place_populations,
    fertility_rates=nd_fertility,
    survival_rates=nd_survival,
    migration_rates_by_geography=place_migrations,
    config=config
)
```

### Example: Hierarchical Aggregation

```python
from cohort_projections.geographic import (
    aggregate_to_county,
    aggregate_to_state,
    validate_aggregation
)

# Run place projections
place_results = run_multi_geography_projections(level='place', ...)

# Aggregate places to counties
county_aggregated = aggregate_to_county(place_results['results'])

# Validate aggregation (should match independently-run county projections)
validation = validate_aggregation(
    component_projections=place_results['results'],
    aggregated_projection=county_aggregated['38017'],  # Cass County
    component_level='place',
    aggregate_level='county',
    tolerance=0.01  # 1% tolerance
)

print(f"Validation passed: {validation['valid']}")
print(f"Difference: {validation['overall']['percent_difference']:.2%}")

# Run county projections
county_results = run_multi_geography_projections(level='county', ...)

# Aggregate counties to state
state_projection = aggregate_to_state(county_results['results'])

# Total state population
total = state_projection[state_projection['year'] == 2045]['population'].sum()
print(f"North Dakota 2045 population: {total:,.0f}")
```

## Configuration

Geographic projections are configured via `config/projection_config.yaml`:

```yaml
geography:
  state: "38"  # North Dakota FIPS

  # County-level projections
  counties:
    mode: "all"  # Project all 53 counties
    # mode: "list"
    # fips_codes: ["38101", "38015"]  # Specific counties

  # Place-level projections
  places:
    mode: "threshold"
    min_population: 500  # Only places with 500+ population
    # mode: "all"  # All 406 places
    # mode: "list"
    # fips_codes: ["3825700", "3807200"]  # Fargo, Bismarck

  # Geographic reference data
  reference_data:
    source: "local"  # or "tiger" for Census TIGER files
    vintage: 2020
    counties_file: "data/raw/geographic/nd_counties.csv"
    places_file: "data/raw/geographic/nd_places.csv"

  # Hierarchical aggregation
  hierarchy:
    validate_aggregation: true
    aggregation_tolerance: 0.01  # 1% tolerance for rounding
    include_balance: true  # Calculate unincorporated areas

# Multi-geography processing
geographic:
  parallel_processing:
    enabled: true
    max_workers: null  # Auto-detect (cpu_count)
    chunk_size: 10

  error_handling:
    mode: "continue"  # Don't stop on single geography failure
    log_failed_geographies: true
    retry_failed: false
```

## Geographic Reference Data

### County Data Format

CSV file: `data/raw/geographic/nd_counties.csv`

```csv
state_fips,county_fips,county_name,population
38,38015,Burleigh County,98458
38,38017,Cass County,184525
38,38035,Grand Forks County,73959
...
```

### Place Data Format

CSV file: `data/raw/geographic/nd_places.csv`

```csv
state_fips,place_fips,place_name,county_fips,population
38,3807200,Bismarck city,38015,73622
38,3825700,Fargo city,38017,125990
38,3833900,Grand Forks city,38035,59166
...
```

### Data Sources

1. **Local CSV Files** (default):
   - Manually curated reference data
   - Stored in `data/raw/geographic/`
   - Fast and offline-capable

2. **Census TIGER Files** (future):
   - Authoritative federal source
   - Downloaded from Census Bureau
   - Requires internet connection

## Parallel Processing

### Performance

| Geographies | Serial Time | Parallel Time (8 cores) | Speedup |
|-------------|-------------|-------------------------|---------|
| 53 counties | ~3 minutes  | ~30 seconds            | 6x      |
| 406 places  | ~20 minutes | ~3 minutes             | 7x      |

### Configuration

```python
# Enable parallel processing
results = run_multi_geography_projections(
    level='county',
    ...,
    parallel=True,          # Enable parallelism
    max_workers=8          # Number of workers (default: cpu_count)
)

# Disable for debugging (serial execution)
results = run_multi_geography_projections(
    level='county',
    ...,
    parallel=False
)
```

### Memory Considerations

- Each worker loads rate files independently
- Memory usage scales with `max_workers`
- For 8 workers: ~8 GB RAM recommended
- Reduce `max_workers` if memory-constrained

## Output Files

Projection results saved to `data/output/projections/{level}s/`:

```
data/output/projections/
├── states/
│   ├── nd_state_38_projection_2025_2045_baseline.parquet
│   ├── nd_state_38_summary_2025_2045_baseline.csv
│   └── nd_state_38_metadata_2025_2045_baseline.json
├── counties/
│   ├── nd_county_38015_projection_2025_2045_baseline.parquet  # Burleigh
│   ├── nd_county_38017_projection_2025_2045_baseline.parquet  # Cass
│   ├── ...
│   ├── counties_summary.csv      # Summary across all counties
│   └── counties_metadata.json    # Processing metadata
└── places/
    ├── nd_place_3807200_projection_2025_2045_baseline.parquet  # Bismarck
    ├── nd_place_3825700_projection_2025_2045_baseline.parquet  # Fargo
    ├── ...
    ├── places_summary.csv
    └── places_metadata.json
```

### File Formats

1. **Projection (Parquet)**: Full time series by age-sex-race
   - Columns: `year`, `age`, `sex`, `race`, `population`
   - Compressed with gzip
   - Efficient storage and fast loading

2. **Summary (CSV)**: Annual totals and key statistics
   - Human-readable
   - Easy to plot in Excel/R/Python

3. **Metadata (JSON)**: Processing details
   - Geography information
   - Rate files used
   - Processing time
   - Summary statistics
   - Validation results

## Error Handling

### Continue-on-Error Mode (Default)

```python
results = run_multi_geography_projections(
    level='county',
    ...,
    config={'geographic': {'error_handling': {'mode': 'continue'}}}
)

# Check for failures
if results['failed_geographies']:
    print(f"Failed: {results['failed_geographies']}")
    # Logs contain detailed error information
```

### Fail-Fast Mode

```python
# Stop on first error
results = run_multi_geography_projections(
    level='county',
    ...,
    config={'geographic': {'error_handling': {'mode': 'fail_fast'}}}
)
# Raises exception on first failure
```

## Validation

### Aggregation Validation

Ensures that sum of component geographies equals aggregate:

```python
validation = validate_aggregation(
    component_projections=place_results['results'],
    aggregated_projection=county_aggregate,
    component_level='place',
    aggregate_level='county',
    tolerance=0.01  # 1% tolerance for rounding
)

# Check results
if validation['valid']:
    print("Aggregation consistent")
else:
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")

# Year-by-year differences
for year_result in validation['by_year']:
    print(f"Year {year_result['year']}: {year_result['percent_difference']:.2%} diff")
```

## API Reference

### Geography Loader Functions

#### `load_nd_counties()`

```python
def load_nd_counties(
    source: Literal['local', 'tiger'] = 'local',
    vintage: int = 2020,
    reference_path: Optional[Path] = None
) -> pd.DataFrame
```

Load North Dakota county reference data.

**Returns**: DataFrame with `state_fips`, `county_fips`, `county_name`, `population`

#### `load_nd_places()`

```python
def load_nd_places(
    source: Literal['local', 'tiger'] = 'local',
    vintage: int = 2020,
    reference_path: Optional[Path] = None,
    min_population: Optional[int] = None
) -> pd.DataFrame
```

Load North Dakota place reference data, optionally filtered by population.

**Returns**: DataFrame with `state_fips`, `place_fips`, `place_name`, `county_fips`, `population`

#### `load_geography_list()`

```python
def load_geography_list(
    level: Literal['state', 'county', 'place'],
    config: Optional[Dict] = None,
    fips_codes: Optional[List[str]] = None
) -> List[str]
```

Get list of FIPS codes to project based on configuration.

**Returns**: List of FIPS codes

### Multi-Geography Functions

#### `run_single_geography_projection()`

```python
def run_single_geography_projection(
    fips: str,
    level: Literal['state', 'county', 'place'],
    base_population: pd.DataFrame,
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates: pd.DataFrame,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    save_results: bool = True
) -> Dict[str, Any]
```

Run projection for a single geography.

**Returns**: Dict with `geography`, `projection`, `summary`, `metadata`, `processing_time`

#### `run_multi_geography_projections()`

```python
def run_multi_geography_projections(
    level: Literal['state', 'county', 'place'],
    base_population_by_geography: Dict[str, pd.DataFrame],
    fertility_rates: pd.DataFrame,
    survival_rates: pd.DataFrame,
    migration_rates_by_geography: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Dict] = None,
    fips_codes: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]
```

Run projections for multiple geographies with optional parallelism.

**Returns**: Dict with `results`, `summary`, `metadata`, `failed_geographies`

#### `aggregate_to_county()`

```python
def aggregate_to_county(
    place_projections: List[Dict[str, Any]],
    config: Optional[Dict] = None
) -> Dict[str, pd.DataFrame]
```

Aggregate place-level projections to county level.

**Returns**: Dict mapping county FIPS → aggregated projection DataFrame

#### `aggregate_to_state()`

```python
def aggregate_to_state(
    county_projections: List[Dict[str, Any]],
    config: Optional[Dict] = None
) -> pd.DataFrame
```

Aggregate county-level projections to state level.

**Returns**: State-level aggregated projection DataFrame

#### `validate_aggregation()`

```python
def validate_aggregation(
    component_projections: List[Dict[str, Any]],
    aggregated_projection: pd.DataFrame,
    component_level: Literal['place', 'county'],
    aggregate_level: Literal['county', 'state'],
    tolerance: float = 0.01,
    config: Optional[Dict] = None
) -> Dict[str, Any]
```

Validate aggregated projection matches sum of components.

**Returns**: Dict with `valid`, `errors`, `warnings`, `by_year`, `overall`

## Best Practices

### 1. Use Population Thresholds

Small places (<500 population) have volatile projections:

```python
# Recommended: Filter to stable places
places = load_nd_places(min_population=500)
```

### 2. Enable Parallel Processing

Significant speedup for multiple geographies:

```python
# Enable for production runs
results = run_multi_geography_projections(..., parallel=True)

# Disable for debugging
results = run_multi_geography_projections(..., parallel=False)
```

### 3. Validate Aggregations

Always validate hierarchical aggregations:

```python
validation = validate_aggregation(
    component_projections=place_results['results'],
    aggregated_projection=county_aggregate,
    component_level='place',
    aggregate_level='county'
)

assert validation['valid'], "Aggregation validation failed"
```

### 4. Use Geography-Specific Migration Rates

Migration varies significantly by geography:

```python
# Good: Geography-specific migration
migration_by_county = {
    '38015': burleigh_migration,
    '38017': cass_migration,
    ...
}

results = run_multi_geography_projections(
    ...,
    migration_rates_by_geography=migration_by_county
)

# Avoid: Shared migration (less accurate)
results = run_multi_geography_projections(
    ...,
    migration_rates_by_geography=None  # Uses shared rates
)
```

### 5. Handle Failed Geographies

Check for and handle failures:

```python
results = run_multi_geography_projections(...)

if results['failed_geographies']:
    print(f"Warning: {len(results['failed_geographies'])} geographies failed")
    for fips in results['failed_geographies']:
        print(f"  - {get_geography_name(fips)}")
    # Check logs for detailed error information
```

## Troubleshooting

### Issue: Out of Memory with Parallel Processing

**Solution**: Reduce `max_workers`:

```python
results = run_multi_geography_projections(
    ...,
    parallel=True,
    max_workers=4  # Reduce from default (cpu_count)
)
```

### Issue: Aggregation Validation Failing

**Solution**: Check tolerance and investigate discrepancies:

```python
validation = validate_aggregation(..., tolerance=0.02)  # Increase tolerance

# Examine year-by-year differences
for year_result in validation['by_year']:
    if not year_result['within_tolerance']:
        print(f"Year {year_result['year']}: {year_result['percent_difference']:.2%}")
```

### Issue: Missing Geographic Reference Files

**Solution**: Files are auto-created with major geographies. For complete data:

1. Download Census TIGER files
2. Process to CSV format
3. Save to `data/raw/geographic/`

Or use `source='tiger'` (requires tigris library).

## Related Documentation

- **ADR-010**: Geographic Scope and Granularity
- **ADR-013**: Multi-Geography Projection Design
- **Core Projection Engine**: `cohort_projections/core/cohort_component.py`
- **Data Processors**: `cohort_projections/data/process/`

## Examples

See `examples/run_multi_geography_example.py` for complete working example.

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review ADR-013 for design decisions
3. Consult example scripts in `examples/`
