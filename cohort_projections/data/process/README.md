# Data Processing Module

## Overview

The `data.process` module processes raw demographic data from various sources (Census, SEER, NVSS, IRS) into standardized formats required by the cohort component projection engine.

### Modules

1. **`base_population.py`**: Process Census data into cohort matrices
2. **`fertility_rates.py`**: Process SEER/NVSS fertility rates
3. **`mortality_rates.py`**: Process SEER/CDC life tables (planned)
4. **`migration_rates.py`**: Process IRS/ACS migration data (planned)

---

# Base Population Processor

## Overview

The `base_population.py` module processes raw Census data (PEP/ACS) into structured cohort matrices for population projections. It creates Age × Sex × Race/Ethnicity matrices at state, county, and place geographic levels.

## Features

### 1. Race/Ethnicity Harmonization
- Maps various Census race codes to a standardized 6-category system:
  - White alone, Non-Hispanic
  - Black alone, Non-Hispanic
  - AIAN alone, Non-Hispanic (American Indian/Alaska Native)
  - Asian/PI alone, Non-Hispanic (Asian/Pacific Islander)
  - Two or more races, Non-Hispanic
  - Hispanic (any race)

### 2. Cohort Matrix Creation
- Creates complete cohort matrices with all demographic dimensions
- Ensures all combinations exist (fills with 0 for missing cohorts)
- Handles single-year age groups (0-90+)
- Standardizes sex categories (Male, Female)

### 3. Multi-Level Geographic Processing
- **State-level**: North Dakota aggregate
- **County-level**: All 53 counties
- **Place-level**: Cities and incorporated places

### 4. Data Validation
- Checks for missing cohorts
- Validates presence of all expected counties
- Detects negative populations
- Flags unusual sex ratios
- Ensures totals are reasonable

### 5. Metadata and Output
- Adds base year and processing date to all outputs
- Saves compressed Parquet files
- Includes summary statistics

## Usage

### Basic Usage

```python
from cohort_projections.data.process.base_population import (
    process_state_population,
    process_county_population,
    process_place_population
)

# Process state-level population
state_matrix = process_state_population(raw_state_data)

# Process all counties
county_matrix = process_county_population(raw_county_data)

# Process all places
place_matrix = process_place_population(raw_place_data)
```

### Input Data Format

#### State-Level Data
Required columns:
- `age`: Integer (0-90+)
- `sex`: String ("Male", "Female", or variants)
- `race_ethnicity`: String (Census race code)
- `population`: Numeric (count)

#### County-Level Data
Required columns (in addition to state-level):
- `county`: County FIPS code (or `COUNTY`, `county_fips`, `COUNTYFP`, `geo_id`)

#### Place-Level Data
Required columns (in addition to state-level):
- `place`: Place FIPS code (or `PLACE`, `place_fips`, `PLACEFP`, `geo_id`)

### Example with Sample Data

```python
import pandas as pd
from cohort_projections.data.process.base_population import (
    harmonize_race_categories,
    create_cohort_matrix,
    get_cohort_summary
)

# Create sample data
sample_data = pd.DataFrame({
    'age': [25, 25, 30, 30, 35, 35],
    'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'race_ethnicity': ['WA_NH', 'WA_NH', 'H', 'H', 'BA_NH', 'BA_NH'],
    'population': [1000, 950, 800, 825, 150, 145]
})

# Harmonize race categories
harmonized = harmonize_race_categories(sample_data)

# Create cohort matrix
cohort_matrix = create_cohort_matrix(
    harmonized,
    geography_level='state',
    geography_id='38'
)

# Get summary statistics
summary = get_cohort_summary(cohort_matrix)
print(summary)
```

### Advanced Usage: Custom Processing

```python
from cohort_projections.data.process.base_population import (
    harmonize_race_categories,
    create_cohort_matrix,
    validate_cohort_matrix
)

# Load raw data
raw_data = pd.read_csv('census_data.csv')

# Step 1: Harmonize race categories
harmonized = harmonize_race_categories(raw_data)

# Step 2: Create cohort matrix
cohort_matrix = create_cohort_matrix(
    harmonized,
    geography_level='county',
    geography_id='38001'  # Adams County
)

# Step 3: Validate
validation_results = validate_cohort_matrix(
    cohort_matrix,
    geography_level='county'
)

if validation_results['valid']:
    print("Validation passed!")
else:
    print("Errors:", validation_results['errors'])

if validation_results['warnings']:
    print("Warnings:", validation_results['warnings'])

# Step 4: Save
cohort_matrix.to_parquet(
    'output/county_cohort_matrix.parquet',
    compression='gzip'
)
```

## Functions Reference

### `harmonize_race_categories(df: pd.DataFrame) -> pd.DataFrame`
Maps Census race codes to standardized 6-category system.

**Parameters:**
- `df`: DataFrame with race/ethnicity column

**Returns:**
- DataFrame with harmonized 'race_ethnicity' column

**Raises:**
- `ValueError`: If race column not found or contains unmapped categories

---

### `create_cohort_matrix(df: pd.DataFrame, geography_level: str, geography_id: Optional[str] = None) -> pd.DataFrame`
Creates age × sex × race cohort matrix.

**Parameters:**
- `df`: DataFrame with age, sex, race_ethnicity, population columns
- `geography_level`: One of 'state', 'county', 'place'
- `geography_id`: Geographic identifier (FIPS code)

**Returns:**
- DataFrame with complete cohort matrix

**Raises:**
- `ValueError`: If required columns missing

---

### `validate_cohort_matrix(df: pd.DataFrame, geography_level: str, expected_counties: Optional[int] = None) -> Dict`
Validates cohort matrix for completeness and plausibility.

**Parameters:**
- `df`: Cohort matrix DataFrame
- `geography_level`: Geographic level being validated
- `expected_counties`: Number of counties expected (for county data)

**Returns:**
- Dictionary with validation results:
  - `valid`: Boolean
  - `warnings`: List of warning messages
  - `errors`: List of error messages

---

### `process_state_population(raw_data: pd.DataFrame, output_dir: Optional[Path] = None) -> pd.DataFrame`
Processes state-level base population.

**Parameters:**
- `raw_data`: Raw Census DataFrame
- `output_dir`: Output directory (default: data/processed/base_population)

**Returns:**
- Processed cohort matrix DataFrame

**Raises:**
- `ValueError`: If validation fails

---

### `process_county_population(raw_data: pd.DataFrame, output_dir: Optional[Path] = None) -> pd.DataFrame`
Processes all 53 North Dakota counties.

**Parameters:**
- `raw_data`: Raw Census DataFrame with county identifier
- `output_dir`: Output directory

**Returns:**
- Processed cohort matrix DataFrame for all counties

**Raises:**
- `ValueError`: If validation fails or counties missing

---

### `process_place_population(raw_data: pd.DataFrame, output_dir: Optional[Path] = None) -> pd.DataFrame`
Processes cities and incorporated places.

**Parameters:**
- `raw_data`: Raw Census DataFrame with place identifier
- `output_dir`: Output directory

**Returns:**
- Processed cohort matrix DataFrame for all places

**Raises:**
- `ValueError`: If validation fails

---

### `get_cohort_summary(cohort_matrix: pd.DataFrame) -> pd.DataFrame`
Generates summary statistics for cohort matrix.

**Parameters:**
- `cohort_matrix`: Processed cohort matrix

**Returns:**
- DataFrame with summary statistics by demographic group

## Output Files

All processed data is saved to `data/processed/base_population/`:

- `state_base_population.parquet`: State-level cohort matrix
- `county_base_population.parquet`: All 53 counties
- `place_base_population.parquet`: All places

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age (0-90) |
| sex | str | Male or Female |
| race_ethnicity | str | 6-category race/ethnicity |
| population | float | Population count |
| geography_level | str | state, county, or place |
| geography_id | str | FIPS code (if applicable) |
| base_year | int | Base year (2025) |
| processing_date | str | Date processed (YYYY-MM-DD) |

## Validation Checks

The module performs the following validation checks:

1. **Completeness**: All age × sex × race combinations present
2. **County Coverage**: All 53 ND counties present (for county data)
3. **Non-negative**: No negative population values
4. **Sex Ratios**: Flags unusual sex ratios (>2.0 or <0.5)
5. **Total Population**: Ensures total is reasonable (non-zero)

## Configuration

The module reads configuration from `config/projection_config.yaml`:

```yaml
demographics:
  age_groups:
    type: "single_year"
    min_age: 0
    max_age: 90
  sex:
    - "Male"
    - "Female"
  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"
```

## Logging

All processing steps are logged using the project's logging configuration. Logs include:
- Processing progress
- Validation results
- Warnings for unusual data patterns
- Summary statistics

## Error Handling

The module provides clear error messages for common issues:
- Missing required columns
- Unmapped race categories
- Validation failures
- File I/O errors

## Dependencies

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0
- pyyaml >= 6.0

## Testing

To test the module:

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic import test
python -c "from cohort_projections.data.process.base_population import *; print('Success!')"

# Run with sample data (see examples above)
```

## Notes

- **Single-year ages**: Age 90 represents "90 and over" (open-ended)
- **Missing cohorts**: All cohorts are included, even if population is 0
- **Compression**: Uses gzip compression for Parquet files (configurable)
- **North Dakota specific**: Expects 53 counties (North Dakota county count)

## Related Modules

- `cohort_projections.utils.logger`: Logging utilities
- `cohort_projections.utils.config_loader`: Configuration management
- `cohort_projections.data.fetch`: Data acquisition from Census API
- `cohort_projections.data.validate`: Additional validation tools

---

# Fertility Rates Processor

## Overview

The `fertility_rates.py` module processes raw SEER (Surveillance, Epidemiology, and End Results Program) or NVSS (National Vital Statistics System) fertility data into age-specific fertility rates (ASFR) by race/ethnicity for the cohort component projection engine.

## Features

### 1. Multi-Format Data Loading
- Supports CSV, TXT (tab-delimited), Excel, and Parquet formats
- Flexible column name recognition
- Optional year range filtering
- Automatic format detection

### 2. Race/Ethnicity Harmonization
- Maps SEER race codes to standardized 6-category system
- Handles numeric and text-based race codes
- Explicit mapping with warnings for unmapped categories

### 3. Multi-Year Averaging
- Weighted averaging based on female population size
- Configurable averaging period (default: 5 years)
- Smooths annual fluctuations
- Handles missing data gracefully

### 4. Complete Rate Table Creation
- Ensures all age (15-49) × race (6 categories) combinations present
- Fills missing combinations with 0 (conservative approach)
- Validates plausibility of rates

### 5. Comprehensive Validation
- Checks for negative rates
- Validates age range coverage
- Ensures all race categories present
- Calculates Total Fertility Rate (TFR) for quality assurance
- Flags implausibly high rates

### 6. Metadata and Provenance
- Saves processing metadata (JSON)
- Includes TFR by race/ethnicity
- Documents source files and parameters
- Timestamped processing records

## Usage

### Basic Usage - Complete Pipeline

```python
from cohort_projections.data.process.fertility_rates import process_fertility_rates

# Process SEER fertility data with single function call
fertility_rates = process_fertility_rates(
    input_path='data/raw/fertility/seer_asfr_2018_2022.csv',
    year_range=(2018, 2022),
    averaging_period=5
)

# Output files created in data/processed/fertility/:
#   - fertility_rates.parquet
#   - fertility_rates.csv
#   - fertility_rates_metadata.json
```

### Step-by-Step Processing

```python
from cohort_projections.data.process.fertility_rates import (
    load_seer_fertility_data,
    harmonize_fertility_race_categories,
    calculate_average_fertility_rates,
    create_fertility_rate_table,
    validate_fertility_rates
)

# Step 1: Load raw data
raw_df = load_seer_fertility_data(
    'data/raw/fertility/seer_data.csv',
    year_range=(2018, 2022)
)

# Step 2: Harmonize race categories
harmonized_df = harmonize_fertility_race_categories(raw_df)

# Step 3: Calculate multi-year averages
averaged_df = calculate_average_fertility_rates(
    harmonized_df,
    averaging_period=5
)

# Step 4: Create complete fertility table
fertility_table = create_fertility_rate_table(
    averaged_df,
    validate=True
)

# Step 5: Validate rates
validation = validate_fertility_rates(fertility_table)
print(f"Valid: {validation['valid']}")
print(f"TFR by race: {validation['tfr_by_race']}")
```

### Input Data Format

SEER/NVSS fertility data should contain:

**Required columns** (flexible naming):
- `year`: Year of data (int)
- `age` (or `age_group`, `age_of_mother`): Age (int, 15-49)
- `race` (or `race_ethnicity`, `race_code`): Race/ethnicity code
- `fertility_rate`: Age-specific fertility rate (births per woman)

**Optional columns** (for weighted averaging):
- `population`: Female population in age-race group
- `births`: Number of births (for metadata)

**Example CSV format:**
```csv
year,age,race,fertility_rate,population,births
2018,15,White NH,0.0050,10000,50
2018,16,White NH,0.0075,9800,74
2018,17,White NH,0.0120,9600,115
...
2022,48,Hispanic,0.0040,8500,34
2022,49,Hispanic,0.0015,8300,12
```

### Output Data Format

Processed fertility rates ready for projection engine:

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age (15-49) |
| race_ethnicity | str | Standardized 6-category race/ethnicity |
| fertility_rate | float | Births per woman per year (0.0-0.15) |
| processing_date | str | Date processed (YYYY-MM-DD) |

**Example output:**
```
   age                    race_ethnicity  fertility_rate processing_date
0   15  White alone, Non-Hispanic         0.0051        2025-12-18
1   16  White alone, Non-Hispanic         0.0078        2025-12-18
2   17  White alone, Non-Hispanic         0.0125        2025-12-18
...
208  48  Hispanic (any race)              0.0042        2025-12-18
209  49  Hispanic (any race)              0.0016        2025-12-18
```

## Functions Reference

### `load_seer_fertility_data(file_path, year_range=None)`
Load raw SEER fertility data from file.

**Parameters:**
- `file_path`: Path to fertility data file (CSV, TXT, Excel, Parquet)
- `year_range`: Optional tuple (min_year, max_year) to filter data

**Returns:**
- DataFrame with raw SEER fertility data

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format not supported

---

### `harmonize_fertility_race_categories(df)`
Map SEER race codes to standard 6 categories.

**Parameters:**
- `df`: DataFrame with SEER race/ethnicity column

**Returns:**
- DataFrame with harmonized 'race_ethnicity' column

**Raises:**
- `ValueError`: If race column not found

---

### `calculate_average_fertility_rates(df, averaging_period=5)`
Average fertility rates over multiple years.

**Parameters:**
- `df`: DataFrame with fertility rates by year, age, and race
- `averaging_period`: Number of years to average (default: 5)

**Returns:**
- DataFrame with averaged fertility rates

**Notes:**
- Uses weighted average if `population` column exists
- Otherwise uses simple mean

---

### `create_fertility_rate_table(df, validate=True, config=None)`
Create final fertility rate table for projection.

**Parameters:**
- `df`: DataFrame with averaged fertility rates
- `validate`: Whether to validate the table (default: True)
- `config`: Optional configuration dictionary

**Returns:**
- Complete fertility rate table (35 ages × 6 races = 210 rows)

**Raises:**
- `ValueError`: If validation fails and validate=True

---

### `validate_fertility_rates(df, config=None)`
Validate fertility rates for plausibility.

**Parameters:**
- `df`: DataFrame with fertility rates
- `config`: Optional configuration dictionary

**Returns:**
- Dictionary with validation results:
  - `valid`: Boolean
  - `errors`: List of error messages
  - `warnings`: List of warning messages
  - `tfr_by_race`: Dict of TFR by race/ethnicity
  - `overall_tfr`: Float (mean TFR)

**Validation Checks:**
- All ages 15-49 present
- All 6 race categories present
- No negative rates
- Rates ≤ 0.15 (error) and ≤ 0.13 (warning)
- TFR in typical range (1.3-2.5)

---

### `process_fertility_rates(input_path, output_dir=None, config=None, year_range=None, averaging_period=5)`
Main processing function - complete pipeline.

**Parameters:**
- `input_path`: Path to raw SEER/NVSS fertility data
- `output_dir`: Output directory (default: data/processed/fertility)
- `config`: Optional configuration dictionary
- `year_range`: Optional (min_year, max_year) tuple
- `averaging_period`: Years to average (default: 5)

**Returns:**
- Processed fertility rate DataFrame

**Output Files:**
- `fertility_rates.parquet`: Primary output (compressed)
- `fertility_rates.csv`: Human-readable backup
- `fertility_rates_metadata.json`: Processing metadata

**Pipeline Steps:**
1. Load raw data
2. Harmonize race categories
3. Calculate multi-year averages
4. Create complete fertility table
5. Validate rates
6. Save outputs and metadata

## Output Files

Processed data saved to `data/processed/fertility/`:

- **`fertility_rates.parquet`**: Primary output (Parquet format, gzip compressed)
- **`fertility_rates.csv`**: Human-readable CSV backup
- **`fertility_rates_metadata.json`**: Processing metadata and provenance

### Metadata Schema

```json
{
  "processing_date": "2025-12-18T14:30:00",
  "source_file": "data/raw/fertility/seer_asfr_2018_2022.csv",
  "year_range": [2018, 2022],
  "averaging_period": 5,
  "total_records": 210,
  "age_range": [15, 49],
  "race_categories": ["White alone, Non-Hispanic", ...],
  "tfr_by_race": {
    "White alone, Non-Hispanic": 1.75,
    "Black alone, Non-Hispanic": 1.65,
    "Hispanic (any race)": 2.05,
    ...
  },
  "overall_tfr": 1.82,
  "validation_warnings": [],
  "config_used": {...}
}
```

## Validation and Quality Checks

### Error Conditions (Fail Validation)
1. **Negative rates**: Any fertility_rate < 0
2. **Missing ages**: Any age 15-49 not present
3. **Missing races**: Any of 6 standard categories not present
4. **Extreme rates**: Any rate > 0.15 (biological implausibility)

### Warning Conditions (Pass with Flags)
1. **High rates**: Rates > 0.13 (unusual but possible)
2. **Low TFR**: TFR < 1.0 (very low fertility)
3. **High TFR**: TFR > 3.0 (very high fertility)
4. **Zero-fill**: Any age-race combinations filled with 0

### Total Fertility Rate (TFR)

TFR = Sum of age-specific fertility rates across all ages

**Interpretation:**
- TFR represents average number of children per woman
- Typical U.S. range: 1.3-2.5
- Replacement level: ~2.1
- Used for quality assurance of processed rates

**TFR by Race/Ethnicity (typical U.S. patterns):**
- White alone, Non-Hispanic: 1.6-1.8
- Black alone, Non-Hispanic: 1.7-1.9
- Hispanic (any race): 2.0-2.3
- AIAN alone, Non-Hispanic: 1.7-2.0
- Asian/PI alone, Non-Hispanic: 1.4-1.7
- Two or more races, Non-Hispanic: 1.6-1.9

## Configuration

Reads from `config/projection_config.yaml`:

```yaml
demographics:
  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"

rates:
  fertility:
    source: "SEER"
    averaging_period: 5
    apply_to_ages: [15, 49]

output:
  compression: "gzip"
```

## Example: Integration with Projection Engine

```python
# Process fertility rates
from cohort_projections.data.process import process_fertility_rates

fertility_rates = process_fertility_rates(
    input_path='data/raw/fertility/seer_data.csv',
    year_range=(2018, 2022)
)

# Use in projection
from cohort_projections.core import CohortComponentProjection

projection = CohortComponentProjection(
    base_population=base_pop_df,
    fertility_rates=fertility_rates,  # <-- Use processed rates here
    survival_rates=survival_df,
    migration_rates=migration_df
)

results = projection.run_projection(
    start_year=2025,
    end_year=2045
)
```

## SEER Race Code Mapping

The module maps SEER race codes to 6 standard categories:

| SEER Code | Standard Category |
|-----------|-------------------|
| White NH, NH White, WNH, 1 | White alone, Non-Hispanic |
| Black NH, NH Black, BNH, 2 | Black alone, Non-Hispanic |
| AIAN NH, NH AIAN, 3 | AIAN alone, Non-Hispanic |
| Asian NH, API NH, 4 | Asian/PI alone, Non-Hispanic |
| Two+ Races NH, 5 | Two or more races, Non-Hispanic |
| Hispanic, Hisp, 6 | Hispanic (any race) |

See `SEER_RACE_ETHNICITY_MAP` in `fertility_rates.py` for complete mapping.

## Architectural Decisions

See **ADR-001: Fertility Rate Processing Methodology** (`docs/adr/001-fertility-rate-processing.md`) for detailed documentation of design decisions:

1. **Multi-year averaging with population weighting**
2. **Zero fill for missing combinations** (not imputation)
3. **Explicit race code mapping with strict validation**
4. **Plausibility thresholds** (errors vs warnings)
5. **Multi-format support** with flexible column naming
6. **Metadata generation** for reproducibility

## Logging

All processing steps are logged:
- Data loading progress
- Race category harmonization
- Averaging calculations
- Validation results
- TFR by race/ethnicity
- Warnings for unusual patterns

Example log output:
```
2025-12-18 14:30:00 - INFO - Loading SEER fertility data from sample_data.csv
2025-12-18 14:30:01 - INFO - Loaded 6300 records from sample_data.csv
2025-12-18 14:30:01 - INFO - Harmonizing fertility race/ethnicity categories
2025-12-18 14:30:02 - INFO - Calculating average fertility rates over 5 years
2025-12-18 14:30:03 - INFO - Creating fertility rate table for projection
2025-12-18 14:30:04 - INFO - Fertility rates validated successfully. Overall TFR: 1.82
```

## Dependencies

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0
- pyyaml >= 6.0

## Testing

### Example Script

See `examples/process_fertility_example.py` for complete working examples:

```bash
python examples/process_fertility_example.py
```

### Unit Tests (Recommended)

```python
# Test race code mapping
from cohort_projections.data.process.fertility_rates import harmonize_fertility_race_categories
import pandas as pd

test_df = pd.DataFrame({
    'age': [25, 30],
    'race': ['White NH', 'Hispanic'],
    'fertility_rate': [0.08, 0.09]
})

harmonized = harmonize_fertility_race_categories(test_df)
assert 'race_ethnicity' in harmonized.columns
assert harmonized['race_ethnicity'].iloc[0] == 'White alone, Non-Hispanic'
```

## Data Sources

### SEER (Recommended)
- **Source**: NCI SEER Program (https://seer.cancer.gov/popdata/)
- **Coverage**: U.S. states and counties
- **Format**: ASCII text files (tab-delimited)
- **Frequency**: Annual
- **Age detail**: Single-year ages

### NVSS
- **Source**: CDC National Vital Statistics System
- **Coverage**: National and state-level
- **Format**: Various (CSV, ASCII)
- **Frequency**: Annual
- **Age detail**: 5-year age groups or single-year

### State Vital Statistics
- **Source**: State health departments
- **Coverage**: State-specific
- **Format**: Varies by state
- **Frequency**: Annual

## Notes

- **Reproductive ages**: Ages 15-49 following demographic convention
- **Missing data strategy**: Conservative zero-fill (see ADR-001)
- **TFR calculation**: Sum of ASFRs (not cohort TFR)
- **Zero rates**: May indicate truly zero fertility or missing data
- **Validation warnings**: Review but don't necessarily indicate errors

## Related Modules

- `cohort_projections.core.fertility`: Uses processed fertility rates
- `cohort_projections.utils.logger`: Logging utilities
- `cohort_projections.utils.config_loader`: Configuration management
- `cohort_projections.data.process.base_population`: Population processing (similar pattern)

---

# Survival Rates Processor

## Overview

The `survival_rates.py` module processes raw SEER (Surveillance, Epidemiology, and End Results Program) or CDC life tables into age-specific survival rates by sex and race/ethnicity for the cohort component projection engine.

## Features

### 1. Multi-Format Life Table Loading
- Supports CSV, TXT (tab-delimited), Excel, and Parquet formats
- Flexible column name recognition (lx, qx, Lx, Tx)
- Optional year filtering for multi-year life tables
- Automatic format detection

### 2. Multiple Conversion Methods
- **lx method** (preferred): S(x) = l(x+1) / l(x)
- **qx method**: S(x) = 1 - q(x)
- **Lx method**: S(x) = L(x+1) / L(x)
- Automatic method selection based on available columns

### 3. Special Age 90+ Handling
- Open-ended age group requires special calculation
- Uses Tx-based formula when available: S(90+) = T(91) / (T(90) + L(90)/2)
- Graceful fallback to default values if Tx/Lx not available

### 4. Mortality Improvement
- Lee-Carter style linear improvement over time
- Configurable annual improvement factor (default: 0.5%)
- Projects future survival rates from base year life table

### 5. Complete Rate Table Creation
- Ensures all age (0-90) × sex (2) × race (6 categories) combinations present
- Fills missing combinations with age-appropriate defaults
- 91 ages × 2 sexes × 6 races = 1,092 rows

### 6. Comprehensive Validation
- Age-specific plausibility checks
- Life expectancy calculation for quality assurance
- Validates presence of all required combinations
- Flags unusual patterns

### 7. Metadata and Provenance
- Saves processing metadata (JSON)
- Includes life expectancy by sex and race
- Documents calculation method and parameters
- Timestamped processing records

## Usage

### Basic Usage - Complete Pipeline

```python
from cohort_projections.data.process.survival_rates import process_survival_rates

# Process SEER life table with single function call
survival_rates = process_survival_rates(
    input_path='data/raw/mortality/seer_lifetables_2020.csv',
    base_year=2020,
    improvement_factor=0.005  # 0.5% annual improvement
)

# Output files created in data/processed/mortality/:
#   - survival_rates.parquet
#   - survival_rates.csv
#   - survival_rates_metadata.json
```

### Step-by-Step Processing

```python
from cohort_projections.data.process.survival_rates import (
    load_life_table_data,
    harmonize_mortality_race_categories,
    calculate_survival_rates_from_life_table,
    apply_mortality_improvement,
    create_survival_rate_table,
    validate_survival_rates,
    calculate_life_expectancy
)

# Step 1: Load raw life table
raw_df = load_life_table_data(
    'data/raw/mortality/seer_lifetable.csv',
    year=2020
)

# Step 2: Harmonize race categories
harmonized_df = harmonize_mortality_race_categories(raw_df)

# Step 3: Calculate survival rates
survival_df = calculate_survival_rates_from_life_table(
    harmonized_df,
    method='lx'  # or 'qx' or 'Lx'
)

# Step 4: Apply mortality improvement (optional)
improved_df = apply_mortality_improvement(
    survival_df,
    base_year=2020,
    projection_year=2025,
    improvement_factor=0.005
)

# Step 5: Create complete survival table
survival_table = create_survival_rate_table(
    improved_df,
    validate=True
)

# Step 6: Calculate life expectancy for QA
life_exp = calculate_life_expectancy(survival_table)
print(f"Life expectancy by sex-race: {life_exp}")

# Step 7: Validate rates
validation = validate_survival_rates(survival_table)
print(f"Valid: {validation['valid']}")
print(f"Life expectancy: {validation['life_expectancy']}")
```

### Input Data Format

SEER/CDC life table data should contain:

**Required columns** (flexible naming):
- `age` (or `age_group`, `agegrp`): Age (int, 0-90)
- `sex` (or `gender`): Sex (Male, Female)
- `race` (or `race_ethnicity`, `race_code`): Race/ethnicity code

**Life table columns** (need at least one):
- `lx`: Survivorship (number surviving to age x, radix typically 100,000)
- `qx`: Death probability (probability of dying between age x and x+1)
- `Lx`: Person-years lived in age interval
- `Tx`: Total person-years lived above age x

**Optional columns**:
- `year`: Year of life table (for filtering)
- `dx`: Deaths between age x and x+1
- `ex`: Life expectancy at age x

**Example CSV format**:
```csv
age,sex,race,lx,qx,Lx,Tx,ex
0,Male,White NH,100000,0.00600,99415,7650000,76.50
1,Male,White NH,99400,0.00040,99380,7550585,75.97
2,Male,White NH,99360,0.00030,99345,7451205,75.01
...
89,Male,White NH,15420,0.15000,14268,142680,9.25
90,Male,White NH,13107,0.35000,8519,85190,6.50
```

### Output Data Format

Processed survival rates ready for projection engine:

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age (0-90) |
| sex | str | Male or Female |
| race_ethnicity | str | Standardized 6-category race/ethnicity |
| survival_rate | float | Probability of surviving to next age (0.0-1.0) |
| processing_date | str | Date processed (YYYY-MM-DD) |

**Example output**:
```
   age    sex                race_ethnicity  survival_rate processing_date
0    0   Male  White alone, Non-Hispanic         0.9940        2025-12-18
1    1   Male  White alone, Non-Hispanic         0.9996        2025-12-18
2    2   Male  White alone, Non-Hispanic         0.9997        2025-12-18
...
1089 88 Female  Hispanic (any race)              0.8800        2025-12-18
1090 89 Female  Hispanic (any race)              0.8500        2025-12-18
1091 90 Female  Hispanic (any race)              0.6500        2025-12-18
```

## Functions Reference

### `load_life_table_data(file_path, year=None)`
Load raw SEER/CDC life table from file.

**Parameters:**
- `file_path`: Path to life table file (CSV, TXT, Excel, Parquet)
- `year`: Optional year to filter (for multi-year life tables)

**Returns:**
- DataFrame with raw life table data

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format not supported

---

### `harmonize_mortality_race_categories(df)`
Map SEER/CDC race codes to standard 6 categories.

**Parameters:**
- `df`: DataFrame with SEER/CDC race/ethnicity column

**Returns:**
- DataFrame with harmonized 'race_ethnicity' column

**Raises:**
- `ValueError`: If race column not found

---

### `calculate_survival_rates_from_life_table(df, method='lx')`
Convert life table to survival rates.

**Parameters:**
- `df`: DataFrame with life table data
- `method`: Calculation method - 'lx', 'qx', or 'Lx'

**Returns:**
- DataFrame with survival_rate column

**Methods:**
- `lx`: S(x) = l(x+1) / l(x) [preferred, most accurate]
- `qx`: S(x) = 1 - q(x) [simple, direct]
- `Lx`: S(x) = L(x+1) / L(x) [accounts for within-interval deaths]

**Special handling:**
- Age 90+ uses formula: S(90+) = T(91) / (T(90) + L(90)/2)
- Falls back to default 0.65 if Tx/Lx not available

---

### `apply_mortality_improvement(df, base_year, projection_year, improvement_factor=0.005)`
Apply mortality improvement trends over time.

**Parameters:**
- `df`: DataFrame with survival rates
- `base_year`: Base year of life table
- `projection_year`: Target projection year
- `improvement_factor`: Annual improvement rate (default: 0.005 = 0.5%)

**Returns:**
- DataFrame with improved survival rates

**Formula:**
```
q(x, t) = q(x, base) × (1 - improvement_factor)^(t - base)
S(x, t) = 1 - q(x, t)
```

**Notes:**
- Typical improvement factors: 0.25%-1.0% annually
- Capped at S = 1.0 (cannot exceed 100% survival)
- No improvement if projection_year <= base_year

---

### `create_survival_rate_table(df, validate=True, config=None)`
Create final survival rate table for projection.

**Parameters:**
- `df`: DataFrame with survival rates
- `validate`: Whether to validate the table (default: True)
- `config`: Optional configuration dictionary

**Returns:**
- Complete survival rate table (91 ages × 2 sexes × 6 races = 1,092 rows)

**Fills missing values with age-appropriate defaults:**
- Age 0 (infant): 0.994
- Ages 1-14 (children): 0.9995
- Ages 15-64 (adults): 0.997
- Ages 65-89 (elderly): 0.95
- Age 90+: 0.65

**Raises:**
- `ValueError`: If validation fails and validate=True

---

### `validate_survival_rates(df, config=None)`
Validate survival rates for plausibility.

**Parameters:**
- `df`: DataFrame with survival rates
- `config`: Optional configuration dictionary

**Returns:**
- Dictionary with validation results:
  - `valid`: Boolean
  - `errors`: List of error messages
  - `warnings`: List of warning messages
  - `life_expectancy`: Dict of e0 by sex-race

**Validation Checks:**
- All ages 0-90 present
- All sex and race categories present
- Rates in range [0, 1]
- Age-specific plausibility:
  - Infant (age 0): 0.993-0.995
  - Children (1-14): > 0.9995
  - Elderly (65-84): 0.93-0.98
  - Age 90+: 0.6-0.7
- Life expectancy in typical range (75-87 years)

---

### `calculate_life_expectancy(df)`
Calculate life expectancy at birth (e0) from survival rates.

**Parameters:**
- `df`: DataFrame with survival rates

**Returns:**
- Dictionary of life expectancy by sex-race combination
- Format: {"{sex}_{race}": e0, ...}

**Notes:**
- Simplified approximation: e0 ≈ sum(cumulative survival)
- Sufficient for quality assurance validation
- Typical U.S. values: 75-87 years

**Example output:**
```python
{
    "Male_White alone, Non-Hispanic": 76.5,
    "Female_White alone, Non-Hispanic": 81.2,
    "Male_Black alone, Non-Hispanic": 72.1,
    "Female_Black alone, Non-Hispanic": 78.0,
    ...
}
```

---

### `process_survival_rates(input_path, output_dir=None, config=None, base_year=None, improvement_factor=None)`
Main processing function - complete pipeline.

**Parameters:**
- `input_path`: Path to raw SEER/CDC life table
- `output_dir`: Output directory (default: data/processed/mortality)
- `config`: Optional configuration dictionary
- `base_year`: Base year of life table (for metadata)
- `improvement_factor`: Annual mortality improvement (default: 0.005)

**Returns:**
- Processed survival rate DataFrame

**Output Files:**
- `survival_rates.parquet`: Primary output (compressed)
- `survival_rates.csv`: Human-readable backup
- `survival_rates_metadata.json`: Processing metadata

**Pipeline Steps:**
1. Load raw life table data
2. Harmonize race categories
3. Calculate survival rates (automatic method selection)
4. Create complete survival table
5. Validate rates
6. Save outputs and metadata

## Output Files

Processed data saved to `data/processed/mortality/`:

- **`survival_rates.parquet`**: Primary output (Parquet format, gzip compressed)
- **`survival_rates.csv`**: Human-readable CSV backup
- **`survival_rates_metadata.json`**: Processing metadata and provenance

### Metadata Schema

```json
{
  "processing_date": "2025-12-18T14:30:00",
  "source_file": "data/raw/mortality/seer_lifetables_2020.csv",
  "base_year": 2020,
  "improvement_factor": 0.005,
  "calculation_method": "lx",
  "total_records": 1092,
  "age_range": [0, 90],
  "sex_categories": ["Male", "Female"],
  "race_categories": ["White alone, Non-Hispanic", ...],
  "life_expectancy": {
    "Male_White alone, Non-Hispanic": 76.5,
    "Female_White alone, Non-Hispanic": 81.2,
    "Male_Black alone, Non-Hispanic": 72.1,
    "Female_Black alone, Non-Hispanic": 78.0,
    "Male_Hispanic (any race)": 79.0,
    "Female_Hispanic (any race)": 84.2,
    ...
  },
  "validation_warnings": [],
  "config_used": {...}
}
```

## Validation and Quality Checks

### Error Conditions (Fail Validation)
1. **Negative rates**: Any survival_rate < 0
2. **Rates > 1.0**: Any survival_rate > 1.0
3. **Missing ages**: Any age 0-90 not present
4. **Missing sexes/races**: Any required category not present

### Warning Conditions (Pass with Flags)
1. **Low infant survival**: < 0.990 (typical: 0.993-0.995)
2. **Low child survival**: < 0.9990 (typical: > 0.9995)
3. **Unusual elderly rates**: Outside 0.90-0.99 range
4. **Age 90+ outside range**: < 0.50 or > 0.80 (typical: 0.60-0.70)
5. **Life expectancy unusual**: < 70 or > 90 years

### Life Expectancy Reference Values

**United States (2020-2023)**:

| Sex-Race Group | Expected e0 |
|----------------|-------------|
| White NH Male | 76-78 years |
| White NH Female | 81-83 years |
| Black NH Male | 71-73 years |
| Black NH Female | 77-79 years |
| Hispanic Male | 78-80 years |
| Hispanic Female | 83-85 years |
| AIAN NH | 70-75 years |
| Asian/PI NH | 83-87 years |

## Configuration

Reads from `config/projection_config.yaml`:

```yaml
demographics:
  age_groups:
    type: "single_year"
    min_age: 0
    max_age: 90
  sex:
    - "Male"
    - "Female"
  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"

rates:
  mortality:
    source: "SEER"
    life_table_year: 2020
    improvement_factor: 0.005  # 0.5% annual improvement

output:
  compression: "gzip"
```

## Example: Integration with Projection Engine

```python
# Process survival rates
from cohort_projections.data.process import process_survival_rates

survival_rates = process_survival_rates(
    input_path='data/raw/mortality/seer_lifetables_2020.csv',
    base_year=2020,
    improvement_factor=0.005
)

# Use in projection
from cohort_projections.core import CohortComponentProjection

projection = CohortComponentProjection(
    base_population=base_pop_df,
    fertility_rates=fertility_df,
    survival_rates=survival_rates,  # <-- Use processed rates here
    migration_rates=migration_df
)

results = projection.run_projection(
    start_year=2025,
    end_year=2045
)
```

## SEER/CDC Race Code Mapping

The module maps SEER/CDC race codes to 6 standard categories:

| SEER Code | Standard Category |
|-----------|-------------------|
| White NH, NH White, WNH, 1 | White alone, Non-Hispanic |
| Black NH, NH Black, BNH, 2 | Black alone, Non-Hispanic |
| AIAN NH, NH AIAN, 3 | AIAN alone, Non-Hispanic |
| Asian NH, API NH, 4 | Asian/PI alone, Non-Hispanic |
| Two+ Races NH, 5 | Two or more races, Non-Hispanic |
| Hispanic, Hisp, 6 | Hispanic (any race) |

See `SEER_MORTALITY_RACE_MAP` in `survival_rates.py` for complete mapping.

## Life Table Format Reference

### Standard Life Table Columns

| Column | Definition | Typical Range |
|--------|------------|---------------|
| age | Age at start of interval | 0-90 |
| qx | Probability of death | 0.0005-0.35 |
| lx | Number surviving (radix 100,000) | 100,000 to ~10,000 |
| dx | Deaths in interval (dx = lx × qx) | 50 to 5,000 |
| Lx | Person-years lived | Varies by age |
| Tx | Total person-years above age x | Decreases with age |
| ex | Life expectancy at age x | 80 at birth to 5 at age 90 |

### Conversion Formulas

**Method 1 - From lx** (preferred):
```
S(x) = l(x+1) / l(x)
```

**Method 2 - From qx**:
```
S(x) = 1 - q(x)
```

**Method 3 - From Lx**:
```
S(x) = L(x+1) / L(x)
```

**Age 90+ (open-ended group)**:
```
S(90+) = T(91) / (T(90) + L(90)/2)

Where:
- T(91) = T(90) - L(90)
- Denominator = person-years at risk
```

## Architectural Decisions

See **ADR-002: Survival Rate Processing Methodology** (`docs/adr/002-survival-rate-processing.md`) for detailed documentation of design decisions:

1. **Multi-method life table conversion** with automatic selection
2. **Special handling for age 90+ open-ended group**
3. **Lee-Carter style mortality improvement**
4. **Age-specific plausibility thresholds** for validation
5. **Age-appropriate default values** for missing data
6. **Life expectancy calculation** for quality assurance

## Logging

All processing steps are logged:
- Life table data loading
- Race category harmonization
- Survival rate calculation method
- Mortality improvement application
- Validation results
- Life expectancy by sex-race
- Warnings for unusual patterns

Example log output:
```
2025-12-18 14:30:00 - INFO - Loading life table data from seer_lifetables_2020.csv
2025-12-18 14:30:01 - INFO - Loaded 1092 records from seer_lifetables_2020.csv
2025-12-18 14:30:01 - INFO - Harmonizing mortality race/ethnicity categories
2025-12-18 14:30:02 - INFO - Calculating survival rates using method: lx
2025-12-18 14:30:03 - INFO - Creating survival rate table for projection
2025-12-18 14:30:04 - INFO - Calculating life expectancy at birth (e0)
2025-12-18 14:30:04 - INFO - Life expectancy at birth (e0) by sex and race:
2025-12-18 14:30:04 - INFO -   Male_White alone, Non-Hispanic: 76.50 years
2025-12-18 14:30:04 - INFO -   Female_White alone, Non-Hispanic: 81.20 years
2025-12-18 14:30:05 - INFO - Survival rates validated successfully
```

## Dependencies

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0
- pyyaml >= 6.0

## Data Sources

### SEER (Recommended)
- **Source**: NCI SEER Program (https://seer.cancer.gov/popdata/)
- **Coverage**: U.S. states and counties
- **Format**: ASCII text files (tab-delimited)
- **Frequency**: Annual
- **Age detail**: Single-year ages 0-90+

### CDC WONDER
- **Source**: CDC WONDER Life Tables (https://wonder.cdc.gov/)
- **Coverage**: National and state-level
- **Format**: CSV, text
- **Frequency**: Annual
- **Age detail**: Single-year ages

### Social Security Administration (SSA)
- **Source**: SSA Actuarial Life Tables
- **Coverage**: National (by sex)
- **Format**: Text, Excel
- **Frequency**: Annual
- **Age detail**: Single-year ages 0-100+

## Notes

- **Single-year ages**: Age 90 represents "90 and over" (open-ended)
- **Survival rates**: Probability of surviving from age x to age x+1
- **Life expectancy**: Period life expectancy (not cohort)
- **Missing data strategy**: Age-appropriate defaults (see ADR-002)
- **Validation warnings**: Review but don't necessarily indicate errors
- **Mortality improvement**: Configurable; set to 0 for constant mortality

## Related Modules

- `cohort_projections.core.mortality`: Uses processed survival rates
- `cohort_projections.utils.logger`: Logging utilities
- `cohort_projections.utils.config_loader`: Configuration management
- `cohort_projections.data.process.fertility_rates`: Similar processing pattern

---

# Migration Rates Processor

## Overview

The `migration_rates.py` module processes raw IRS county-to-county migration flows and Census/ACS international migration data into age-specific, sex-specific, race-specific net migration rates or counts for the cohort component projection engine.

**The Migration Challenge**: Unlike fertility and mortality which come with demographic detail, IRS migration data provides only aggregate migrant counts with NO age/sex/race breakdown. This module solves the distribution problem using standard demographic age patterns and population-proportional allocation.

## Features

### 1. Multi-Source Data Loading
- IRS county-to-county migration flows (domestic migration)
- Census/ACS international migration estimates
- Supports CSV, TXT (tab-delimited), Excel, and Parquet formats
- Flexible column name recognition
- Optional year range and geography filtering

### 2. Net Migration Calculation
- Calculates net migration (in - out) by area
- Handles both positive (net in-migration) and negative (net out-migration)
- Combines domestic and international components
- Preserves totals through distribution

### 3. Distribution Algorithms
- **Age Distribution**: Standard demographic age pattern (peaks at 20-35)
  - Simplified method (default): Easy-to-understand age-group multipliers
  - Rogers-Castro model (optional): Established demographic migration model
- **Sex Distribution**: 50/50 split (default, configurable)
- **Race Distribution**: Proportional to population composition

### 4. Age Pattern Methods
- **Simplified Pattern**: Age-group multipliers based on demographic knowledge
- **Rogers-Castro Model**: Mathematical model from migration literature
- Both methods produce plausible age-specific migration patterns

### 5. Complete Rate Table Creation
- Ensures all age (0-90) × sex (2) × race (6) combinations present
- Total combinations: 91 ages × 2 sexes × 6 races = 1,092 rows
- Supports both absolute net migration and migration rates
- Missing cohorts filled with 0

### 6. Comprehensive Validation
- Age pattern plausibility (should peak at young adult ages)
- Extreme value detection (>20% of population)
- Negative population checks
- Total migration conservation validation

### 7. Metadata and Provenance
- Saves processing metadata (JSON)
- Includes in/out migration totals separately
- Documents distribution methodology
- Timestamped processing records

## Usage

### Basic Usage - Complete Pipeline

```python
from cohort_projections.data.process.migration_rates import process_migration_rates

# Process IRS and international migration with single function call
migration_rates = process_migration_rates(
    irs_path='data/raw/migration/irs_flows_2018_2022.csv',
    intl_path='data/raw/migration/international_2018_2022.csv',
    population_path='data/processed/base_population.parquet',
    year_range=(2018, 2022),
    target_county_fips='38'  # North Dakota
)

# Output files created in data/processed/migration/:
#   - migration_rates.parquet
#   - migration_rates.csv
#   - migration_rates_metadata.json
```

### Step-by-Step Processing

```python
from cohort_projections.data.process.migration_rates import (
    load_irs_migration_data,
    load_international_migration_data,
    get_standard_age_migration_pattern,
    distribute_migration_by_age,
    distribute_migration_by_sex,
    distribute_migration_by_race,
    calculate_net_migration,
    create_migration_rate_table,
    validate_migration_data
)

# Step 1: Load IRS county-to-county flows
irs_df = load_irs_migration_data(
    'data/raw/migration/irs_flows.csv',
    year_range=(2018, 2022),
    target_county_fips='38'
)

# Step 2: Load international migration
intl_df = load_international_migration_data(
    'data/raw/migration/international.csv',
    year_range=(2018, 2022),
    target_county_fips='38'
)

# Step 3: Calculate net domestic migration
in_migration = irs_df[irs_df['to_county_fips'].str.startswith('38')]
out_migration = irs_df[irs_df['from_county_fips'].str.startswith('38')]
net_domestic = in_migration['migrants'].sum() - out_migration['migrants'].sum()

# Step 4: Get international migration total
net_international = intl_df['international_migrants'].sum()

# Step 5: Total net migration
total_net = net_domestic + net_international

# Step 6: Get standard age pattern
age_pattern = get_standard_age_migration_pattern(
    peak_age=25,
    method='simplified'  # or 'rogers_castro'
)

# Step 7: Distribute to ages
age_migration = distribute_migration_by_age(total_net, age_pattern)

# Step 8: Distribute to sex
age_sex_migration = distribute_migration_by_sex(
    age_migration,
    sex_ratio=0.5  # 50/50 split
)

# Step 9: Load base population for race distribution
population_df = pd.read_parquet('data/processed/base_population.parquet')

# Step 10: Distribute to race/ethnicity
age_sex_race_migration = distribute_migration_by_race(
    age_sex_migration,
    population_df
)

# Step 11: Create complete migration table
migration_table = create_migration_rate_table(
    age_sex_race_migration,
    population_df=population_df,
    as_rates=False,  # Use absolute numbers (default)
    validate=True
)

# Step 12: Validate
validation = validate_migration_data(migration_table, population_df)
print(f"Valid: {validation['valid']}")
print(f"Total net migration: {validation['total_net_migration']:,.0f}")
```

### Input Data Formats

#### IRS County-to-County Flows

**Required columns** (flexible naming):
- `from_county_fips` (or `from_fips`, `origin_fips`): Origin county FIPS code
- `to_county_fips` (or `to_fips`, `dest_fips`): Destination county FIPS code
- `migrants` (or `migration`, `count`): Number of migrants (aggregate)
- `year`: Year of migration data

**Example CSV format**:
```csv
from_county_fips,to_county_fips,migrants,year
38001,38003,45,2018
38003,38001,32,2018
38001,27053,67,2018
27053,38001,54,2018
...
```

#### International Migration

**Required columns** (flexible naming):
- `county_fips` (or `fips`, `geoid`): County or state FIPS code
- `international_migrants` (or `net_international`): Net international migration
- `year`: Year of estimate

**Example CSV format**:
```csv
county_fips,international_migrants,year
38,523,2018
38,541,2019
38,498,2020
...
```

#### Base Population (Required for Distribution)

**Required columns**:
- `age`: Age (0-90)
- `sex`: Male or Female
- `race_ethnicity`: Standardized 6-category race/ethnicity
- `population`: Population count

This is the output from `base_population.py` processor.

### Output Data Format

Processed migration data ready for projection engine:

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age (0-90) |
| sex | str | Male or Female |
| race_ethnicity | str | Standardized 6-category race/ethnicity |
| net_migration | float | Net migration (can be positive or negative) |
| processing_date | str | Date processed (YYYY-MM-DD) |

OR (if `as_rates=True`):

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age (0-90) |
| sex | str | Male or Female |
| race_ethnicity | str | Standardized 6-category race/ethnicity |
| migration_rate | float | Net migration / population |
| processing_date | str | Date processed (YYYY-MM-DD) |

**Example output**:
```
   age    sex                race_ethnicity  net_migration processing_date
0    0   Male  White alone, Non-Hispanic           2.4        2025-12-18
1    0 Female  White alone, Non-Hispanic           2.3        2025-12-18
2    1   Male  White alone, Non-Hispanic           2.4        2025-12-18
...
1089 89 Female  Hispanic (any race)                0.8        2025-12-18
1090 90   Male  Hispanic (any race)                0.4        2025-12-18
1091 90 Female  Hispanic (any race)                0.4        2025-12-18
```

**Note**: Net migration can be negative (net out-migration) which is normal.

## Functions Reference

### `load_irs_migration_data(file_path, year_range=None, target_county_fips=None)`
Load IRS county-to-county migration flows.

**Parameters:**
- `file_path`: Path to IRS data file (CSV, TXT, Excel, Parquet)
- `year_range`: Optional tuple (min_year, max_year) to filter data
- `target_county_fips`: Optional county/state FIPS to filter (e.g., '38' for ND)

**Returns:**
- DataFrame with from_county_fips, to_county_fips, migrants, year

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format not supported or required columns missing

---

### `load_international_migration_data(file_path, year_range=None, target_county_fips=None)`
Load Census/ACS international migration estimates.

**Parameters:**
- `file_path`: Path to international migration data
- `year_range`: Optional tuple (min_year, max_year)
- `target_county_fips`: Optional county/state FIPS

**Returns:**
- DataFrame with county_fips, international_migrants, year

---

### `get_standard_age_migration_pattern(peak_age=25, method='simplified')`
Get standard migration age pattern.

**Parameters:**
- `peak_age`: Age at which migration peaks (default: 25)
- `method`: Pattern method - 'simplified' or 'rogers_castro'

**Returns:**
- DataFrame with age (0-90) and migration_propensity (sums to 1.0)

**Notes:**
- Simplified: Age-group multipliers (easy to understand)
- Rogers-Castro: Demographic model from literature (more sophisticated)
- Both produce plausible patterns with peak at young adult ages

---

### `distribute_migration_by_age(total_migration, age_pattern)`
Distribute aggregate migration to ages using pattern.

**Parameters:**
- `total_migration`: Total net migration (can be positive or negative)
- `age_pattern`: DataFrame with age and migration_propensity

**Returns:**
- DataFrame with age and migrants

**Notes:**
- Works with both positive and negative net migration
- Propensities should sum to 1.0

---

### `distribute_migration_by_sex(age_migration, sex_ratio=0.5)`
Distribute age-specific migration to sex.

**Parameters:**
- `age_migration`: DataFrame with age and migrants
- `sex_ratio`: Proportion going to males (default: 0.5 for 50/50)

**Returns:**
- DataFrame with age, sex ('Male'/'Female'), migrants

**Notes:**
- sex_ratio = 0.5 means equal split
- sex_ratio = 0.52 would be 52% male, 48% female

---

### `distribute_migration_by_race(age_sex_migration, population_df)`
Distribute to race/ethnicity proportional to population.

**Parameters:**
- `age_sex_migration`: DataFrame with age, sex, migrants
- `population_df`: Base population with age, sex, race_ethnicity, population

**Returns:**
- DataFrame with age, sex, race_ethnicity, migrants

**Notes:**
- Uses population proportions within each age-sex group
- If age-sex group has zero population, distributes equally across races
- Preserves total migration

---

### `calculate_net_migration(in_migration, out_migration)`
Calculate net migration (in - out) by cohort.

**Parameters:**
- `in_migration`: DataFrame with in-migration by age, sex, race
- `out_migration`: DataFrame with out-migration by age, sex, race

**Returns:**
- DataFrame with net_migration (positive or negative)

**Notes:**
- Negative net migration is valid (out > in)
- Cohorts present in only one DataFrame get that value as net

---

### `combine_domestic_international_migration(domestic_df, international_df)`
Combine domestic and international migration.

**Parameters:**
- `domestic_df`: Domestic net migration DataFrame
- `international_df`: International net migration DataFrame

**Returns:**
- Combined net migration DataFrame

**Notes:**
- Sums both components for each cohort
- Handles cohorts present in only one component

---

### `create_migration_rate_table(df, population_df=None, as_rates=False, validate=True, config=None)`
Create final migration table for projection.

**Parameters:**
- `df`: DataFrame with net_migration by age, sex, race
- `population_df`: Optional base population (required if as_rates=True)
- `as_rates`: Whether to express as migration rates vs absolute numbers
- `validate`: Whether to validate the table (default: True)
- `config`: Optional configuration dictionary

**Returns:**
- Complete migration table (1,092 rows) ready for projection engine

**Raises:**
- `ValueError`: If validation fails and validate=True

**Notes:**
- Ensures all 1,092 cohorts present (fills with 0)
- Can output absolute numbers or rates
- Both formats work in projection engine

---

### `validate_migration_data(df, population_df=None, config=None)`
Validate migration data for plausibility.

**Parameters:**
- `df`: Migration DataFrame to validate
- `population_df`: Optional base population for rate validation
- `config`: Optional configuration dictionary

**Returns:**
- Dictionary with validation results:
  - `valid`: Boolean
  - `errors`: List of error messages
  - `warnings`: List of warning messages
  - `total_net_migration`: Float
  - `net_by_direction`: Dict with in/out statistics

**Validation Checks:**
- All age-sex-race combinations present
- Age pattern plausible (should peak at 20-35)
- Migration values not extreme (>20% of population)
- Won't cause negative populations

---

### `process_migration_rates(irs_path, intl_path=None, population_path=None, output_dir=None, config=None, year_range=None, target_county_fips=None, as_rates=False)`
Main processing function - complete pipeline.

**Parameters:**
- `irs_path`: Path to IRS county-to-county flows (required)
- `intl_path`: Optional path to international migration data
- `population_path`: Path to base population (required for distribution)
- `output_dir`: Output directory (default: data/processed/migration)
- `config`: Optional configuration dictionary
- `year_range`: Optional (min_year, max_year) tuple
- `target_county_fips`: Optional county/state FIPS (e.g., '38')
- `as_rates`: Whether to output rates vs absolute numbers

**Returns:**
- Processed migration DataFrame

**Output Files:**
- `migration_rates.parquet`: Primary output (compressed)
- `migration_rates.csv`: Human-readable backup
- `migration_rates_metadata.json`: Processing metadata

**Pipeline Steps:**
1. Load base population
2. Load IRS migration flows
3. Calculate net domestic migration
4. Load international migration (if provided)
5. Distribute to age groups
6. Distribute by sex
7. Distribute by race/ethnicity
8. Create complete migration table
9. Save outputs and metadata

## Output Files

Processed data saved to `data/processed/migration/`:

- **`migration_rates.parquet`**: Primary output (Parquet format, gzip compressed)
- **`migration_rates.csv`**: Human-readable CSV backup
- **`migration_rates_metadata.json`**: Processing metadata and provenance

### Metadata Schema

```json
{
  "processing_date": "2025-12-18T14:30:00",
  "source_files": {
    "irs_data": "data/raw/migration/irs_flows_2018_2022.csv",
    "international_data": "data/raw/migration/international_2018_2022.csv",
    "population_data": "data/processed/base_population.parquet"
  },
  "year_range": [2018, 2022],
  "target_area": "38",
  "output_format": "net_migration",
  "migration_summary": {
    "total_net_migration": 5234,
    "net_domestic": 4123,
    "net_international": 1111,
    "in_migration": 45678,
    "out_migration": 41555
  },
  "distribution_method": {
    "age_pattern": "simplified",
    "sex_distribution": "50/50",
    "race_distribution": "proportional_to_population"
  },
  "validation_summary": {
    "net_in_migration": 5234,
    "net_out_migration": 0,
    "cohorts_positive": 812,
    "cohorts_negative": 280,
    "cohorts_zero": 0
  }
}
```

## Validation and Quality Checks

### Error Conditions (Fail Validation)
1. **Missing cohorts**: Not all 1,092 age-sex-race combinations present
2. **Missing columns**: Required columns (age, sex, race_ethnicity, net_migration) missing

### Warning Conditions (Pass with Flags)
1. **Extreme values**: Net migration >20% of population for cohort
2. **Unusual age pattern**: Peak migration not in ages 20-35
3. **Very large values**: Absolute migration >10,000 for single cohort
4. **Negative population risk**: Migration would cause negative population

### Age Pattern Validation

Migration should peak at young adult ages (20-35):
- Ages 0-17: Lower migration (children with parents)
- Ages 18-35: Peak migration (education, career, family formation)
- Ages 36-64: Moderate to low migration (established careers/families)
- Ages 65+: Lowest migration (settled, retired)

If peak is outside 15-45 age range, validation warns (possible data error).

## Configuration

Reads from `config/projection_config.yaml`:

```yaml
demographics:
  age_groups:
    type: "single_year"
    min_age: 0
    max_age: 90
  sex:
    - "Male"
    - "Female"
  race_ethnicity:
    categories:
      - "White alone, Non-Hispanic"
      - "Black alone, Non-Hispanic"
      - "AIAN alone, Non-Hispanic"
      - "Asian/PI alone, Non-Hispanic"
      - "Two or more races, Non-Hispanic"
      - "Hispanic (any race)"

rates:
  migration:
    domestic:
      method: "IRS_county_flows"
      averaging_period: 5
      smooth_extreme_outliers: false
    international:
      method: "ACS_foreign_born"
      allocation: "proportional"

output:
  compression: "gzip"
```

## Example: Integration with Projection Engine

```python
# Process migration rates
from cohort_projections.data.process import process_migration_rates

migration_rates = process_migration_rates(
    irs_path='data/raw/migration/irs_flows_2018_2022.csv',
    intl_path='data/raw/migration/international_2018_2022.csv',
    population_path='data/processed/base_population.parquet',
    year_range=(2018, 2022),
    target_county_fips='38'
)

# Use in projection
from cohort_projections.core import CohortComponentProjection

projection = CohortComponentProjection(
    base_population=base_pop_df,
    fertility_rates=fertility_df,
    survival_rates=survival_df,
    migration_rates=migration_rates  # <-- Use processed rates here
)

results = projection.run_projection(
    start_year=2025,
    end_year=2045
)
```

## Age Pattern Methods

### Simplified Method (Default)

Age-group multipliers based on demographic knowledge:

| Age Group | Multiplier | Interpretation |
|-----------|------------|----------------|
| 0-9 | 0.30 | Children migrate with parents |
| 10-17 | 0.50 | Teenagers |
| 18-19 | 0.80 | Leaving home |
| 20-29 | 1.00 | Peak migration (education, career) |
| 30-39 | 0.75 | Still mobile |
| 40-49 | 0.45 | Less mobile |
| 50-64 | 0.25 | Settled |
| 65-74 | 0.20 | Early retirement |
| 75+ | 0.10 | Least mobile |

**Advantages**:
- Easy to understand and explain
- Transparent and adjustable
- Matches empirical patterns

### Rogers-Castro Method (Optional)

Mathematical model from demographic literature:

```
M(x) = a1 × exp(-α1 × x) + a2 × exp(-α2(x - μ2) - exp(-λ2(x - μ2))) + c

Components:
- Childhood (decreasing): a1 × exp(-α1 × x)
- Labor force peak: a2 × exp(-α2(x - μ2) - exp(-λ2(x - μ2)))
- Constant baseline: c
```

**Parameters**:
- a1 = 0.02, α1 = 0.08 (childhood migration)
- a2 = 0.06, μ2 = 25, α2 = 0.5, λ2 = 0.4 (young adult peak)
- c = 0.001 (baseline)

**Advantages**:
- Published methodology
- Smooth age pattern
- Widely used in demographic research

## Distribution Algorithm Flow

```
1. Calculate Total Net Migration
   ├─ In-migration: sum(IRS flows TO target area)
   ├─ Out-migration: sum(IRS flows FROM target area)
   ├─ Net domestic: in - out
   └─ Add international migration

2. Distribute to Ages (91 ages)
   └─ Use age pattern (simplified or Rogers-Castro)

3. Distribute to Sex (× 2 = 182 combinations)
   └─ Split by sex ratio (default 50/50)

4. Distribute to Race (× 6 = 1,092 cohorts)
   └─ Proportional to population composition

5. Create Complete Table
   └─ Fill missing cohorts with 0
```

## Architectural Decisions

See **ADR-003: Migration Rate Processing Methodology** (`docs/adr/003-migration-rate-processing.md`) for detailed documentation of design decisions:

1. **Simplified age pattern over Rogers-Castro** (default)
2. **50/50 sex distribution** (configurable)
3. **Population-proportional race distribution**
4. **Net migration calculation** (in - out, allows negatives)
5. **Domestic + international combination** (separate then sum)
6. **Migration rates vs absolute numbers** (absolute as default)
7. **Outlier smoothing methodology** (off by default)
8. **Missing data handling** (zero fill)

## Logging

All processing steps are logged:
- IRS and international data loading
- Net migration calculation (in/out/net)
- Age pattern generation
- Distribution to age, sex, race
- Validation results
- Total net migration by component

Example log output:
```
2025-12-18 14:30:00 - INFO - Loading IRS migration data from irs_flows_2018_2022.csv
2025-12-18 14:30:01 - INFO - Domestic migration: in=45,678, out=41,555, net=+4,123
2025-12-18 14:30:02 - INFO - International migration: +1,111
2025-12-18 14:30:02 - INFO - Total net migration: +5,234
2025-12-18 14:30:03 - INFO - Distributing migration to age groups
2025-12-18 14:30:04 - INFO - Distributing migration by sex
2025-12-18 14:30:05 - INFO - Distributing migration by race/ethnicity
2025-12-18 14:30:06 - INFO - Migration data validated successfully
```

## Dependencies

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- pyarrow >= 12.0.0
- pyyaml >= 6.0

## Testing

### Example Script

See `examples/process_migration_example.py` for complete working examples:

```bash
python examples/process_migration_example.py
```

### Unit Tests (Recommended)

```python
# Test age pattern generation
from cohort_projections.data.process.migration_rates import get_standard_age_migration_pattern

pattern = get_standard_age_migration_pattern(peak_age=25, method='simplified')
assert len(pattern) == 91  # All ages 0-90
assert pattern['migration_propensity'].sum() == 1.0  # Normalized
peak_idx = pattern['migration_propensity'].idxmax()
peak_age = pattern.loc[peak_idx, 'age']
assert 20 <= peak_age <= 35  # Peak in young adult range

# Test distribution
from cohort_projections.data.process.migration_rates import distribute_migration_by_age

age_dist = distribute_migration_by_age(1000, pattern)
assert abs(age_dist['migrants'].sum() - 1000) < 0.01  # Preserves total
```

## Data Sources

### IRS Migration Data
- **Source**: IRS Statistics of Income (SOI) Migration Data (https://www.irs.gov/statistics/soi-tax-stats-migration-data)
- **Coverage**: County-to-county flows for all U.S. counties
- **Format**: CSV (downloadable by year)
- **Frequency**: Annual
- **Detail**: Aggregate counts only (no demographics)

### Census International Migration
- **Source**: Census Bureau Population Estimates Program
- **Coverage**: State and county-level
- **Format**: API or downloadable files
- **Frequency**: Annual estimates
- **Detail**: Net international migration totals

## Notes

- **Aggregate Data**: IRS/Census provide no age/sex/race detail; distribution is necessary
- **Distribution Assumptions**: Age pattern, sex ratio, race proportions are assumptions
- **Negative Migration**: Net out-migration is valid and common for declining areas
- **Validation Critical**: Review age pattern and total migration plausibility
- **Population-Dependent**: Race distribution requires accurate base population

## Related Modules

- `cohort_projections.core.migration`: Uses processed migration rates
- `cohort_projections.utils.logger`: Logging utilities
- `cohort_projections.utils.config_loader`: Configuration management
- `cohort_projections.data.process.fertility_rates`: Similar processing pattern
- `cohort_projections.data.process.survival_rates`: Similar processing pattern
