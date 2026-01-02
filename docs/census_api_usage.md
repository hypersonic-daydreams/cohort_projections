# Census API Data Fetcher Documentation

## Overview

The `CensusDataFetcher` class provides a comprehensive interface for retrieving demographic data from the U.S. Census Bureau APIs for North Dakota. It supports both the Population Estimates Program (PEP) and American Community Survey (ACS) data sources.

## Installation

Ensure required packages are installed:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 2.0.0
- requests >= 2.31.0
- tqdm >= 4.66.0
- pyarrow >= 12.0.0

## Quick Start

```python
from cohort_projections.data.fetch.census_api import CensusDataFetcher

# Initialize fetcher
fetcher = CensusDataFetcher()

# Fetch state-level PEP data
state_df = fetcher.fetch_pep_state_data(vintage=2024)

# Fetch county-level PEP data
county_df = fetcher.fetch_pep_county_data(vintage=2024)

# Fetch ACS place data
places_df = fetcher.fetch_acs_place_data(year=2023)
```

## Data Sources

### Population Estimates Program (PEP)

The PEP provides annual population estimates by detailed demographics:

- **Vintage**: Most recent is 2024 (estimates for 2020-2024)
- **Geography**: State and county levels
- **Demographics**: Age, sex, race, Hispanic origin
- **Frequency**: Annual
- **API**: https://api.census.gov/data/{vintage}/pep/charagegroups

**Available Variables:**
- `POP`: Population estimate
- `AGE`: Age group (single-year or grouped)
- `SEX`: Sex (0=Total, 1=Male, 2=Female)
- `RACE`: Race category
- `HISP`: Hispanic origin (1=Non-Hispanic, 2=Hispanic)
- `DATE_CODE`: Date code for the estimate

### American Community Survey (ACS)

The ACS provides detailed demographic and socioeconomic data:

- **Dataset**: 5-year estimates (e.g., 2019-2023)
- **Geography**: Places (incorporated cities and CDPs)
- **Demographics**: Detailed age/sex distributions, race, ethnicity
- **Frequency**: Annual releases
- **API**: https://api.census.gov/data/{year}/acs/acs5

**Key Variable Tables:**
- `B01001`: Sex by Age (detailed age groups)
- `B02001`: Race
- `B03003`: Hispanic or Latino Origin

## Class Reference

### CensusDataFetcher

#### Initialization

```python
CensusDataFetcher(
    cache_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: int = 5
)
```

**Parameters:**
- `cache_dir`: Directory for caching downloaded data (default: `data/raw/census`)
- `api_key`: Census API key for higher rate limits (optional)
- `max_retries`: Maximum retry attempts for failed requests
- `retry_delay`: Delay in seconds between retries

**Class Attributes:**
- `STATE_FIPS`: "38" (North Dakota FIPS code)
- `STATE_NAME`: "North Dakota"

#### Methods

##### fetch_pep_state_data()

Fetch state-level PEP data with demographic characteristics.

```python
fetch_pep_state_data(
    vintage: int = 2024,
    variables: Optional[List[str]] = None
) -> pd.DataFrame
```

**Returns:** DataFrame with columns:
- `state`: State FIPS code
- `AGE`: Age group
- `SEX`: Sex code
- `RACE`: Race code
- `HISP`: Hispanic origin code
- `POP`: Population estimate
- `DATE_CODE`: Date code
- `DATE_DESC`: Date description

**Example:**
```python
df = fetcher.fetch_pep_state_data(vintage=2024)
print(f"Records: {len(df)}")
print(df.head())
```

##### fetch_pep_county_data()

Fetch county-level PEP data for all 53 North Dakota counties.

```python
fetch_pep_county_data(
    vintage: int = 2024,
    variables: Optional[List[str]] = None
) -> pd.DataFrame
```

**Returns:** DataFrame with columns:
- `state`: State FIPS code
- `county`: County FIPS code
- `AGE`, `SEX`, `RACE`, `HISP`, `POP`: Demographic variables
- `DATE_CODE`, `DATE_DESC`: Date information

**Example:**
```python
df = fetcher.fetch_pep_county_data(vintage=2024)
print(f"Counties: {df['county'].nunique()}")
```

##### fetch_acs_place_data()

Fetch ACS place-level data for North Dakota cities and CDPs.

```python
fetch_acs_place_data(
    year: int = 2023,
    dataset: str = 'acs5'
) -> pd.DataFrame
```

**Parameters:**
- `year`: ACS year (e.g., 2023 for 2019-2023 estimates)
- `dataset`: 'acs5' (5-year) or 'acs1' (1-year)

**Returns:** DataFrame with columns:
- `NAME`: Place name
- `state`: State FIPS code
- `place`: Place FIPS code
- `B01001_001E`: Total population
- `B01001_002E` - `B01001_049E`: Age/sex distributions
- `B02001_*E`: Race variables
- `B03003_*E`: Hispanic origin variables
- `is_cdp`: Boolean flag for Census-Designated Places
- `place_type`: "Incorporated Place" or "Census-Designated Place"

**Example:**
```python
df = fetcher.fetch_acs_place_data(year=2023)
cdps = df[df['is_cdp']]
cities = df[~df['is_cdp']]
print(f"Cities: {len(cities)}, CDPs: {len(cdps)}")
```

##### fetch_pep_by_file()

Fetch PEP data via direct file download (alternative to API).

```python
fetch_pep_by_file(
    vintage: int = 2024,
    geography: str = 'state'
) -> pd.DataFrame
```

**Parameters:**
- `vintage`: PEP vintage year
- `geography`: 'state' or 'county'

**Returns:** DataFrame with all available PEP variables from the file.

**Note:** This method is useful when API endpoints are unavailable or for retrieving additional variables not available via API.

##### fetch_all_pep_data()

Convenience method to fetch all PEP data (state and county).

```python
fetch_all_pep_data(
    vintage: int = 2024,
    use_file_method: bool = True
) -> Dict[str, pd.DataFrame]
```

**Returns:** Dictionary with keys:
- `'state'`: State-level DataFrame
- `'county'`: County-level DataFrame

**Example:**
```python
data = fetcher.fetch_all_pep_data(vintage=2024)
state_df = data['state']
county_df = data['county']
```

##### fetch_all_acs_data()

Convenience method to fetch all ACS data.

```python
fetch_all_acs_data(
    year: int = 2023,
    dataset: str = 'acs5'
) -> pd.DataFrame
```

**Returns:** DataFrame with place-level ACS data.

##### get_cached_data()

Retrieve cached data if available.

```python
get_cached_data(
    source: str,
    geography: str,
    vintage_or_year: int
) -> Optional[pd.DataFrame]
```

**Parameters:**
- `source`: 'pep' or 'acs'
- `geography`: 'state', 'county', or 'place'
- `vintage_or_year`: Year of the data

**Returns:** DataFrame if cached, None otherwise.

**Example:**
```python
df = fetcher.get_cached_data('pep', 'state', 2024)
if df is not None:
    print("Using cached data")
```

##### list_cached_files()

List all cached data files.

```python
list_cached_files() -> Dict[str, List[Path]]
```

**Returns:** Dictionary mapping source types to lists of cached file paths.

## Caching

All fetched data is automatically cached to avoid redundant API calls:

**Cache Structure:**
```
data/raw/census/
├── pep/
│   ├── pep_state_2024.parquet
│   ├── pep_state_2024_metadata.json
│   ├── pep_county_2024.parquet
│   └── pep_county_2024_metadata.json
├── acs/
│   ├── acs5_place_2023.parquet
│   └── acs5_place_2023_metadata.json
└── decennial/
```

**Metadata Files:**
Each cached data file has an accompanying JSON metadata file containing:
- Source type (PEP, ACS)
- Vintage/year
- State information
- API URL used
- Download timestamp
- Record count

## Demographics Coding

### PEP Codes

**SEX:**
- `0`: Total
- `1`: Male
- `2`: Female

**RACE (simplified):**
- `1`: White alone
- `2`: Black or African American alone
- `3`: American Indian and Alaska Native alone
- `4`: Asian alone
- `5`: Native Hawaiian and Other Pacific Islander alone
- `6`: Two or more races

**HISP:**
- `1`: Not Hispanic or Latino
- `2`: Hispanic or Latino

**AGE:**
- `0-99`: Single-year ages
- `999`: All ages combined

### ACS Variables

**B01001 (Sex by Age):**
- `B01001_001E`: Total population
- `B01001_002E`: Male total
- `B01001_003E`: Male under 5 years
- ... (continues through detailed age groups)
- `B01001_026E`: Female total
- `B01001_027E`: Female under 5 years
- ... (continues through detailed age groups)

**B02001 (Race):**
- `B02001_001E`: Total
- `B02001_002E`: White alone
- `B02001_003E`: Black or African American alone
- `B02001_004E`: American Indian and Alaska Native alone
- `B02001_005E`: Asian alone
- `B02001_006E`: Native Hawaiian and Other Pacific Islander alone
- `B02001_007E`: Some other race alone
- `B02001_008E`: Two or more races

**B03003 (Hispanic Origin):**
- `B03003_001E`: Total
- `B03003_002E`: Not Hispanic or Latino
- `B03003_003E`: Hispanic or Latino

## Error Handling

The fetcher includes comprehensive error handling:

1. **Automatic retries**: Failed requests are retried up to `max_retries` times
2. **Exponential backoff**: Configurable delay between retries
3. **Detailed logging**: All operations are logged for debugging
4. **Graceful degradation**: Falls back to cached data when possible

**Example with error handling:**
```python
try:
    df = fetcher.fetch_pep_state_data(vintage=2024)
except requests.RequestException as e:
    logger.error(f"Failed to fetch data: {e}")
    # Try cached data
    df = fetcher.get_cached_data('pep', 'state', 2024)
```

## API Key Setup

While not required, a Census API key provides higher rate limits.

**Get an API key:**
1. Visit: https://api.census.gov/data/key_signup.html
2. Register for a free key
3. Set environment variable:

```bash
export CENSUS_API_KEY="your_key_here"
```

**Use in code:**
```python
import os
api_key = os.environ.get('CENSUS_API_KEY')
fetcher = CensusDataFetcher(api_key=api_key)
```

## Integration with Cohort Projections

### Data Pipeline Integration

```python
from cohort_projections.data.fetch.census_api import CensusDataFetcher
from cohort_projections.utils import ConfigLoader

# Load configuration
config = ConfigLoader()
base_year = config.get_parameter('project', 'base_year')

# Fetch demographic data
fetcher = CensusDataFetcher()
pep_data = fetcher.fetch_all_pep_data(vintage=2024)

# Use in projections
state_pop = pep_data['state']
county_pop = pep_data['county']
```

### Processing Demographics

```python
# Convert PEP data to projection format
def process_pep_demographics(df):
    """Convert PEP codes to readable categories."""
    df_processed = df.copy()

    # Map sex codes
    sex_map = {0: 'Total', 1: 'Male', 2: 'Female'}
    df_processed['sex_label'] = df_processed['SEX'].map(sex_map)

    # Map Hispanic origin
    hisp_map = {1: 'Non-Hispanic', 2: 'Hispanic'}
    df_processed['hispanic_label'] = df_processed['HISP'].map(hisp_map)

    return df_processed

# Apply processing
processed = process_pep_demographics(state_df)
```

## Best Practices

1. **Use caching**: Check for cached data before making API calls
2. **Handle rate limits**: Use an API key for production workloads
3. **Validate data**: Check record counts and demographics after fetch
4. **Save metadata**: Keep track of data sources and vintages
5. **Error logging**: Monitor logs for API issues

## Examples

See `/examples/fetch_census_data.py` for comprehensive usage examples.

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure dependencies are installed
pip install -r requirements.txt
```

**API rate limits:**
```python
# Use API key and increase retry delay
fetcher = CensusDataFetcher(
    api_key=your_key,
    retry_delay=10  # Increase delay
)
```

**Missing data:**
```python
# Check if data exists for vintage/year
try:
    df = fetcher.fetch_pep_state_data(vintage=2024)
except requests.HTTPError as e:
    if e.response.status_code == 404:
        print("Data not available for this vintage")
```

## References

- Census API Documentation: https://www.census.gov/data/developers/data-sets.html
- PEP Technical Documentation: https://www.census.gov/programs-surveys/popest/technical-documentation.html
- ACS Documentation: https://www.census.gov/programs-surveys/acs/technical-documentation.html
- Census API Discovery Tool: https://api.census.gov/data.html
