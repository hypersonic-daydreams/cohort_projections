# Core Module Architecture

## Module Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                    CohortComponentProjection                     │
│                   (cohort_component.py)                          │
│                                                                   │
│  Main orchestration class that coordinates all components        │
│  - Manages projection workflow                                   │
│  - Handles multi-year iterations                                 │
│  - Generates summary statistics                                  │
│  - Exports results                                               │
└───────────────┬─────────────┬─────────────┬─────────────────────┘
                │             │             │
                ▼             ▼             ▼
    ┌───────────────┐ ┌─────────────┐ ┌──────────────┐
    │  Fertility    │ │  Mortality  │ │  Migration   │
    │ (fertility.py)│ │(mortality.py)│ │(migration.py)│
    └───────────────┘ └─────────────┘ └──────────────┘
         │                   │                │
         │                   │                │
         ▼                   ▼                ▼
    ┌────────────────────────────────────────────┐
    │         pandas DataFrame Operations         │
    │    (vectorized, no explicit loops)          │
    └────────────────────────────────────────────┘
                        │
                        ▼
    ┌────────────────────────────────────────────┐
    │          Configuration & Logging            │
    │      (utils.config_loader, utils.logger)    │
    └────────────────────────────────────────────┘
```

## Data Flow

### Single Year Projection

```
Year t Population
        │
        ▼
┌───────────────────┐
│ Apply Survival    │ ← survival_rates
│ (mortality.py)    │
└────────┬──────────┘
         │ Aged population (now year t+1)
         ▼
┌───────────────────┐
│ Calculate Births  │ ← fertility_rates (applied to females)
│ (fertility.py)    │
└────────┬──────────┘
         │ Newborns (age 0)
         ▼
┌───────────────────┐
│ Apply Migration   │ ← migration_rates
│ (migration.py)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Combine           │
│ Aged + Births     │
│ + Migration       │
└────────┬──────────┘
         │
         ▼
Year t+1 Population
```

## Component Details

### 1. Fertility Module (`fertility.py`)

**Input:**
- Female population (year t)
- Age-specific fertility rates (ASFR)

**Process:**
```
For each age a in [15, 49]:
    births(a, race) = females(a, race) × fertility_rate(a, race)

Total_births(race) = Σ births(a, race) for all a

Male_births(race) = Total_births(race) × 0.51
Female_births(race) = Total_births(race) × 0.49
```

**Output:**
- Newborn population by sex and race (age 0, year t+1)

**Functions:**
- `calculate_births()` - Main calculation
- `apply_fertility_scenario()` - Scenario adjustments
- `validate_fertility_rates()` - Input validation

---

### 2. Mortality Module (`mortality.py`)

**Input:**
- Population (year t, all ages/sexes/races)
- Survival rates by age/sex/race

**Process:**
```
For ages 0 to 89:
    pop(a+1, s, r, t+1) = pop(a, s, r, t) × survival_rate(a, s, r)

For age 90+ (open-ended group):
    pop(90+, s, r, t+1) = pop(90+, s, r, t) × survival_rate(90+, s, r)
                         + pop(89, s, r, t) × survival_rate(89, s, r)
```

**Output:**
- Survived and aged population (year t+1)

**Functions:**
- `apply_survival_rates()` - Main calculation
- `apply_mortality_improvement()` - Trend adjustment
- `validate_survival_rates()` - Input validation
- `calculate_life_expectancy()` - Utility function

---

### 3. Migration Module (`migration.py`)

**Input:**
- Population (year t+1, after survival)
- Net migration by age/sex/race

**Process:**
```
For each cohort (a, s, r):
    pop_final(a, s, r, t+1) = pop_survived(a, s, r, t+1) + net_migration(a, s, r)

    If pop_final < 0:
        pop_final = 0  (with warning)
```

**Output:**
- Population with migration (year t+1)

**Functions:**
- `apply_migration()` - Main calculation
- `apply_migration_scenario()` - Scenario adjustments
- `validate_migration_data()` - Input validation
- `distribute_international_migration()` - Helper for allocation
- `combine_domestic_international()` - Combine migration types

---

### 4. Main Engine (`cohort_component.py`)

**Purpose:** Orchestrate the full projection

**Key Methods:**

#### `__init__()`
- Load configuration
- Store input data
- Validate all inputs
- Initialize results storage

#### `project_single_year(population, year, scenario)`
- Apply scenario adjustments to rates
- Call survival module → aged population
- Call fertility module → births
- Call migration module → final population
- Validate results (no negative populations)
- Return year t+1 population

#### `run_projection(start_year, end_year, scenario)`
```python
results = [base_population]

for year in range(start_year, end_year):
    next_pop = project_single_year(current_pop, year, scenario)
    results.append(next_pop)
    current_pop = next_pop

return concatenate(results)
```

#### Analysis Methods
- `get_projection_summary()` - Annual statistics
- `get_population_by_year()` - Extract specific year
- `get_cohort_trajectory()` - Track birth cohort

#### Export Methods
- `export_results()` - Full detailed data
- `export_summary()` - Summary statistics

## Configuration Integration

All modules read from `config/projection_config.yaml`:

```python
# In each module
from ..utils.logger import get_logger_from_config
from ..utils.config_loader import ConfigLoader

logger = get_logger_from_config(__name__)

# Access configuration
config_loader = ConfigLoader()
config = config_loader.get_projection_config()

# Get specific parameters
base_year = config['project']['base_year']
max_age = config['demographics']['age_groups']['max_age']
```

## Error Handling Strategy

### Input Validation
```python
# Check required columns
if 'age' not in df.columns:
    raise ValueError("Missing required column: 'age'")

# Check for negative values
if (df['population'] < 0).any():
    raise ValueError("Negative population values found")
```

### Runtime Warnings
```python
# Warn about missing data
if missing_rates.any():
    logger.warning(f"Missing rates for {missing_rates.sum()} cohorts, using defaults")

# Warn about adjustments
if negative_pops.any():
    logger.warning(f"Capping {negative_pops.sum()} negative populations at 0")
```

### Logging
```python
# Info: Normal progress
logger.info(f"Year {year}: Total population = {total:,.0f}")

# Debug: Detailed information
logger.debug(f"Applying fertility rates to {len(females)} female cohorts")

# Warning: Potential issues
logger.warning(f"Unusual age pattern detected")

# Error: Critical failures
logger.error(f"Projection failed at year {year}: {error}")
```

## Performance Optimization

### Vectorization
```python
# Good: Vectorized operation
births_df['births'] = pop_df['females'] * rates_df['fertility']

# Bad: Explicit loop (avoid)
for i, row in df.iterrows():
    births.append(row['females'] * row['fertility'])
```

### Memory Efficiency
```python
# Use in-place operations where appropriate
df['new_col'] = df['old_col'] * 2  # Creates new column

# For large data, process in chunks if needed
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### Type Efficiency
```python
# Use appropriate dtypes
df['age'] = df['age'].astype('int8')  # 0-90 fits in int8
df['population'] = df['population'].astype('float32')  # Reduce precision
```

## Testing Hooks

Each module provides validation functions for testing:

```python
# Test fertility rates
is_valid, issues = validate_fertility_rates(test_rates, config)
assert is_valid, f"Validation failed: {issues}"

# Test survival rates
is_valid, issues = validate_survival_rates(test_rates, config)
assert is_valid, f"Validation failed: {issues}"

# Test migration data
is_valid, issues = validate_migration_data(test_mig, test_pop, config)
assert is_valid, f"Validation failed: {issues}"
```

## Extension Points

### Adding New Scenarios

1. Define in `config/projection_config.yaml`:
```yaml
scenarios:
  custom_scenario:
    name: "Custom Scenario"
    fertility: "custom_trend"
    mortality: "accelerated_improvement"
    migration: "+50_percent"
```

2. Implement adjustments in each module's scenario function:
```python
def apply_fertility_scenario(rates, scenario, year, base_year):
    if scenario == 'custom_trend':
        # Custom logic here
        pass
```

### Adding New Output Statistics

Extend `_create_annual_summary()`:
```python
def _create_annual_summary(self, population, year):
    summary = {
        'year': year,
        'total_population': population['population'].sum(),
        # Add new statistics
        'your_custom_metric': calculate_custom_metric(population)
    }
    return summary
```

## Module Independence

Each component module (`fertility.py`, `mortality.py`, `migration.py`) can be used independently:

```python
# Use fertility module standalone
from cohort_projections.core.fertility import calculate_births

births = calculate_births(
    female_population=females_df,
    fertility_rates=rates_df,
    year=2025
)

# Use mortality module standalone
from cohort_projections.core.mortality import apply_survival_rates

survived = apply_survival_rates(
    population=pop_df,
    survival_rates=rates_df,
    year=2025
)
```

This modular design enables:
- Unit testing of individual components
- Reuse in other projection systems
- Easy modification of individual components
- Clear separation of concerns
