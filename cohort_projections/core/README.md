# Cohort Component Projection Engine

This directory contains the mathematical core of the population projection system, implementing the standard demographic **cohort-component method**.

## Overview

The cohort-component method is the gold standard for population projections, used by the U.S. Census Bureau, state demographic centers, and international agencies. It projects population forward by tracking cohorts (groups defined by age, sex, and race/ethnicity) through time.

### The Method

For each projection year, the method:

1. **Aging & Survival**: Apply age-specific survival rates to advance cohorts by 1 year
2. **Fertility**: Apply age-specific fertility rates to females → births by sex and race
3. **Migration**: Add net migration (domestic + international) by cohort
4. **Sum**: Total population at t+1

Mathematically, for cohort (a, s, r) at time t:

```
P(a+1, s, r, t+1) = P(a, s, r, t) × S(a, s, r, t) + M(a+1, s, r, t)
```

Where:
- P = Population
- S = Survival rate
- M = Net migration
- a = age, s = sex, r = race

Births are calculated separately:

```
B(0, s, r, t+1) = Σ[P(a, Female, r, t) × F(a, r, t)] × SRB(s)
```

Where:
- F = Fertility rate (births per woman)
- SRB = Sex ratio at birth (typically 51% male)

## Module Structure

### `cohort_component.py` - Main Engine

The `CohortComponentProjection` class orchestrates the full projection:

```python
from cohort_projections.core import CohortComponentProjection

# Initialize
projection = CohortComponentProjection(
    base_population=pop_df,
    fertility_rates=fert_df,
    survival_rates=surv_df,
    migration_rates=mig_df,
    config=config_dict
)

# Run projection
results = projection.run_projection(
    start_year=2025,
    end_year=2045,
    scenario='baseline'
)

# Export
projection.export_results('output/projection.parquet')
projection.export_summary('output/summary.csv')
```

**Key Methods:**
- `project_single_year()` - Project one year forward
- `run_projection()` - Run multi-year projection
- `get_projection_summary()` - Get annual statistics
- `get_population_by_year()` - Extract specific year
- `get_cohort_trajectory()` - Track birth cohort over time

### `fertility.py` - Fertility Component

Calculates births from age-specific fertility rates (ASFR):

```python
from cohort_projections.core import calculate_births

births = calculate_births(
    female_population=female_pop_df,
    fertility_rates=fertility_df,
    year=2025,
    config=config
)
```

**Key Functions:**
- `calculate_births()` - Apply fertility rates to females → births
- `apply_fertility_scenario()` - Adjust rates by scenario
- `validate_fertility_rates()` - Check rate plausibility

**Notes:**
- Fertility rates typically apply to ages 15-49
- Standard sex ratio at birth: 51% male, 49% female
- Mother's race/ethnicity assigned to child
- Rates are births per woman per year

### `mortality.py` - Mortality/Survival Component

Applies survival rates for aging and mortality:

```python
from cohort_projections.core import apply_survival_rates

survived_pop = apply_survival_rates(
    population=pop_df,
    survival_rates=survival_df,
    year=2025,
    config=config
)
```

**Key Functions:**
- `apply_survival_rates()` - Age population with mortality
- `apply_mortality_improvement()` - Trend mortality over time
- `validate_survival_rates()` - Check rate plausibility
- `calculate_life_expectancy()` - Compute life expectancy

**Notes:**
- Survival rates are probabilities (0-1) of surviving from age t to t+1
- Age 90+ is an open-ended group (special handling)
- Mortality improvement: death rates decline ~0.5% per year
- Derived from life tables (SEER, CDC, SSA)

### `migration.py` - Migration Component

Applies net migration (in-migration minus out-migration):

```python
from cohort_projections.core import apply_migration

pop_with_migration = apply_migration(
    population=pop_df,
    migration_rates=mig_df,
    year=2025,
    config=config
)
```

**Key Functions:**
- `apply_migration()` - Apply net migration by cohort
- `apply_migration_scenario()` - Adjust by scenario
- `validate_migration_data()` - Check plausibility
- `distribute_international_migration()` - Allocate international migrants
- `combine_domestic_international()` - Combine migration types

**Notes:**
- Net migration can be positive (in) or negative (out)
- Can be specified as absolute numbers or rates
- Young adults (20-35) most mobile
- Domestic: IRS county flows
- International: Census/ACS estimates

## Data Requirements

### Base Population
**Format:** DataFrame with columns `[year, age, sex, race, population]`

```python
   year  age     sex                        race  population
0  2025    0    Male  White alone, Non-Hispanic       5234.0
1  2025    0  Female  White alone, Non-Hispanic       4987.0
...
```

**Sources:**
- Census Decennial (2020)
- Census Population Estimates Program (PEP)
- American Community Survey (ACS)

### Fertility Rates
**Format:** DataFrame with columns `[age, race, fertility_rate]`

```python
   age                        race  fertility_rate
0   15  White alone, Non-Hispanic          0.0081
1   16  White alone, Non-Hispanic          0.0153
...
```

**Sources:**
- SEER age-specific fertility rates
- National Vital Statistics System (NVSS)
- State vital statistics

**Typical values:**
- Ages 15-19: 0.01-0.03
- Ages 20-29: 0.08-0.12 (peak)
- Ages 30-39: 0.05-0.10
- Ages 40-49: 0.01-0.03

### Survival Rates
**Format:** DataFrame with columns `[age, sex, race, survival_rate]`

```python
   age     sex                        race  survival_rate
0    0    Male  White alone, Non-Hispanic         0.9935
1    0  Female  White alone, Non-Hispanic         0.9945
...
```

**Sources:**
- SEER life tables by race/ethnicity
- CDC life tables
- Social Security Administration (SSA) actuarial tables

**Typical values:**
- Age 0 (infants): 0.993-0.995
- Ages 1-14: 0.9995+
- Ages 15-44: 0.999+
- Ages 45-64: 0.995-0.999
- Ages 65-84: 0.93-0.98
- Age 90+: 0.6-0.7

### Migration Rates
**Format:** DataFrame with columns `[age, sex, race, net_migration]` or `migration_rate`

```python
   age     sex                        race  net_migration
0    0    Male  White alone, Non-Hispanic            2.5
1    0  Female  White alone, Non-Hispanic            2.3
...
```

**Sources:**
- IRS county-to-county migration flows (domestic)
- Census Population Estimates (international)
- ACS migration/mobility data

## Scenarios

The projection engine supports multiple scenarios for sensitivity analysis:

### Baseline
- Recent trends continuation
- Constant fertility
- Improving mortality (0.5% annual)
- Recent average migration

### High Growth
- Fertility +10%
- Constant mortality
- Migration +25%

### Low Growth
- Fertility -10%
- Constant mortality
- Migration -25%

### Zero Migration
- Natural increase only
- Constant fertility
- Improving mortality
- Zero net migration

Configure in `config/projection_config.yaml`:

```yaml
scenarios:
  baseline:
    fertility: "constant"
    mortality: "improving"
    migration: "recent_average"
    active: true
```

## Validation

All modules include validation functions:

```python
from cohort_projections.core import (
    validate_fertility_rates,
    validate_survival_rates,
    validate_migration_data
)

# Check inputs before projection
is_valid, issues = validate_fertility_rates(fertility_df, config)
if not is_valid:
    print(f"Issues found: {issues}")
```

**Validation checks:**
- Required columns present
- No negative populations
- Rates within plausible bounds
- No missing age-sex-race combinations
- Age patterns make demographic sense

## Performance Considerations

### Vectorization
All calculations use pandas/numpy vectorized operations for performance:

```python
# Vectorized calculation (fast)
births['births'] = population['female'] * fertility_rates['rate']

# Avoid loops where possible
# for i, row in df.iterrows():  # Slow!
```

### Memory Management
For large projections:
- Use single-year age groups (not individual ages beyond 90)
- Process counties separately if needed
- Use parquet format with compression
- Consider chunking for very large geographies

### Typical Performance
On a modern laptop:
- State-level (91 ages × 2 sexes × 6 races = 1,092 cohorts)
  - 20-year projection: < 1 second
- All 53 North Dakota counties
  - 20-year projection: < 30 seconds
- All 406 places
  - 20-year projection: ~2 minutes

## Example Usage

See `/examples/run_basic_projection.py` for a complete working example.

Quick start:

```python
from cohort_projections.core import CohortComponentProjection
from cohort_projections.utils.config_loader import load_projection_config

# Load configuration
config = load_projection_config()

# Load your data (from data pipeline)
base_pop = pd.read_parquet('data/processed/base_population_2025.parquet')
fertility = pd.read_parquet('data/processed/fertility_rates.parquet')
survival = pd.read_parquet('data/processed/survival_rates.parquet')
migration = pd.read_parquet('data/processed/migration_rates.parquet')

# Initialize and run
projection = CohortComponentProjection(
    base_population=base_pop,
    fertility_rates=fertility,
    survival_rates=survival,
    migration_rates=migration,
    config=config
)

results = projection.run_projection(
    start_year=2025,
    end_year=2045,
    scenario='baseline'
)

# Analyze
summary = projection.get_projection_summary()
print(summary)

# Export
projection.export_results('output/nd_projection_2025_2045.parquet')
```

## Mathematical Details

### Survival Rate Calculation
Survival rate from life table:

```
S(x) = L(x+1) / L(x)
```

Where L(x) is person-years lived between ages x and x+1.

For open-ended age group (90+):

```
S(90+) = T(91) / (T(90) + L(90)/2)
```

### Total Fertility Rate (TFR)
Sum of age-specific rates:

```
TFR = Σ F(a)  for a = 15 to 49
```

Typical U.S. TFR: 1.6-1.8 births per woman

### Net Reproduction Rate (NRR)
Accounts for mortality:

```
NRR = Σ[F(a) × S(a)] × SRB(female)
```

NRR < 1.0 means population decline without migration

### Migration Timing
Applied at mid-year (July 1) after aging and fertility but integrated into annual change.

## References

- **Smith, S.K., Tayman, J., & Swanson, D.A.** (2013). *A Practitioner's Guide to State and Local Population Projections*. Springer.
- **Swanson, D.A. & Siegel, J.S.** (2004). *The Methods and Materials of Demography* (2nd ed.). Elsevier.
- **U.S. Census Bureau** - Population Projections Methodology: https://www.census.gov/programs-surveys/popproj/technical-documentation/methodology.html
- **National Center for Health Statistics** - Life Tables: https://www.cdc.gov/nchs/products/life_tables.htm

## Support

For questions or issues with the projection engine:
1. Check this README and docstrings
2. Review `/examples/run_basic_projection.py`
3. See `/tests/test_core/` for test cases
4. Consult `/docs/methodology.md` for detailed methods
