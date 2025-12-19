# Data Fetching Modules

This package contains modules for fetching demographic, geographic, and vital statistics data needed for cohort-component population projections.

## Available Modules

### census_api.py

Fetches demographic data from U.S. Census Bureau APIs:

- **Population Estimates Program (PEP)**: Annual population estimates by age, sex, race, and Hispanic origin
- **American Community Survey (ACS)**: Detailed demographic data for places and CDPs

**Key Features:**
- State and county-level PEP data for North Dakota (FIPS 38)
- Place-level ACS data including Census-Designated Places
- Automatic caching of downloaded data
- Retry logic and error handling
- Metadata tracking

**Usage:**
```python
from cohort_projections.data.fetch import CensusDataFetcher

fetcher = CensusDataFetcher()

# Fetch state-level data
state_df = fetcher.fetch_pep_state_data(vintage=2024)

# Fetch county-level data
county_df = fetcher.fetch_pep_county_data(vintage=2024)

# Fetch ACS place data
places_df = fetcher.fetch_acs_place_data(year=2023)
```

See [census_api.py](./census_api.py) for complete documentation.

## Future Modules

The following modules are planned for implementation:

### vital_stats.py (planned)

Fetch vital statistics data:
- Birth data from NVSS (National Vital Statistics System)
- Death data from NVSS
- SEER mortality rates
- Age-specific fertility rates

### migration.py (planned)

Fetch migration data:
- IRS county-to-county migration flows
- ACS mobility data
- International migration estimates

### geographic.py (planned)

Fetch geographic reference data:
- FIPS codes and names
- County/place boundaries
- Census geographic hierarchy

## Data Sources

### Census Bureau APIs
- **Base URL**: https://api.census.gov/data/
- **Documentation**: https://www.census.gov/data/developers/data-sets.html
- **API Key**: Optional but recommended (get at https://api.census.gov/data/key_signup.html)

### NVSS (National Vital Statistics System)
- **URL**: https://www.cdc.gov/nchs/nvss/
- Birth and death certificate data

### SEER (Surveillance, Epidemiology, and End Results)
- **URL**: https://seer.cancer.gov/
- Age-specific mortality rates by demographics

### IRS Statistics of Income
- **URL**: https://www.irs.gov/statistics/soi-tax-stats-migration-data
- County-to-county migration flows

## Caching

All modules use a consistent caching structure:

```
data/raw/
├── census/
│   ├── pep/           # Population Estimates Program
│   ├── acs/           # American Community Survey
│   └── decennial/     # Decennial Census
├── vital_stats/
│   ├── births/
│   └── deaths/
├── migration/
│   ├── irs/
│   └── acs/
└── geographic/
```

Each cached file includes:
- Data in Parquet format (compressed, efficient)
- JSON metadata file with source, timestamp, record count

## Configuration

Data fetching is configured via `config/projection_config.yaml`:

```yaml
geography:
  state: "38"  # North Dakota
  counties: "all"
  places: "all"

rates:
  fertility:
    source: "SEER"
    averaging_period: 5
  mortality:
    source: "SEER"
  migration:
    domestic:
      method: "IRS_county_flows"
    international:
      method: "ACS_foreign_born"
```

## Error Handling

All fetcher modules implement:
- Automatic retry logic for transient failures
- Comprehensive logging via `cohort_projections.utils.logger`
- Graceful fallback to cached data when available
- Detailed error messages for debugging

## Testing

Unit tests are located in `tests/test_data/`:
- `test_census_api.py`: Tests for Census API fetcher
- More test files will be added as modules are implemented

Run tests:
```bash
pytest tests/test_data/
```

## Dependencies

Core dependencies (from `requirements.txt`):
- pandas >= 2.0.0
- requests >= 2.31.0
- tqdm >= 4.66.0
- pyarrow >= 12.0.0
- pyyaml >= 6.0

## Contributing

When adding new data fetcher modules:

1. **Follow the pattern**: Use CensusDataFetcher as a template
2. **Include caching**: Save data to appropriate subdirectory
3. **Save metadata**: Track source, timestamp, API endpoint
4. **Error handling**: Implement retry logic and logging
5. **Documentation**: Include comprehensive docstrings
6. **Tests**: Add unit tests in `tests/test_data/`
7. **Update __init__.py**: Export new classes

## Examples

See `/examples/fetch_census_data.py` for detailed usage examples.

## Related Documentation

- [Census API Usage Guide](../../../docs/census_api_usage.md)
- [Project Configuration](../../../config/projection_config.yaml)
- [Data Processing Pipeline](../process/README.md)
