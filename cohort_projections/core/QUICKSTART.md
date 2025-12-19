# Quick Start Guide - Cohort Component Projection

## 5-Minute Quick Start

### 1. Import the module

```python
from cohort_projections.core import CohortComponentProjection
import pandas as pd
```

### 2. Prepare your data

You need 4 DataFrames:

```python
# Base population (starting point)
base_pop = pd.DataFrame({
    'year': [2025] * 1092,  # Repeated for each cohort
    'age': [...],           # 0-90
    'sex': [...],           # 'Male' or 'Female'
    'race': [...],          # Race/ethnicity category
    'population': [...]     # Population count
})

# Fertility rates (births per woman, ages 15-49)
fertility = pd.DataFrame({
    'age': [...],           # 15-49
    'race': [...],          # Must match base_pop races
    'fertility_rate': [...] # e.g., 0.08 = 80 births per 1000 women
})

# Survival rates (probability of surviving to next age)
survival = pd.DataFrame({
    'age': [...],           # 0-90
    'sex': [...],           # Must match base_pop
    'race': [...],          # Must match base_pop
    'survival_rate': [...]  # 0.0 to 1.0
})

# Migration (net in-migration minus out-migration)
migration = pd.DataFrame({
    'age': [...],           # 0-90
    'sex': [...],           # Must match base_pop
    'race': [...],          # Must match base_pop
    'net_migration': [...]  # Can be positive or negative
})
```

### 3. Run projection

```python
# Initialize
projection = CohortComponentProjection(
    base_population=base_pop,
    fertility_rates=fertility,
    survival_rates=survival,
    migration_rates=migration
)

# Run 20-year projection
results = projection.run_projection(
    start_year=2025,
    end_year=2045,
    scenario='baseline'
)

# View summary
summary = projection.get_projection_summary()
print(summary[['year', 'total_population']])
```

### 4. Export results

```python
# Full detailed results
projection.export_results('output/projection.parquet')

# Summary statistics
projection.export_summary('output/summary.csv')
```

## Common Tasks

### Get population for specific year

```python
pop_2030 = projection.get_population_by_year(2030)
print(f"Total 2030: {pop_2030['population'].sum():,.0f}")
```

### Track a birth cohort over time

```python
# Follow 2025 white males born in 2025
cohort = projection.get_cohort_trajectory(
    birth_year=2025,
    sex='Male',
    race='White alone, Non-Hispanic'
)
print(cohort)
```

### Calculate summary statistics

```python
pop_2045 = projection.get_population_by_year(2045)

# By age group
under_18 = pop_2045[pop_2045['age'] < 18]['population'].sum()
working_age = pop_2045[(pop_2045['age'] >= 18) & (pop_2045['age'] < 65)]['population'].sum()
seniors = pop_2045[pop_2045['age'] >= 65]['population'].sum()

print(f"Under 18: {under_18:,.0f}")
print(f"Working age (18-64): {working_age:,.0f}")
print(f"Seniors (65+): {seniors:,.0f}")

# By sex
by_sex = pop_2045.groupby('sex')['population'].sum()
print(by_sex)

# By race
by_race = pop_2045.groupby('race')['population'].sum()
print(by_race)
```

### Run multiple scenarios

```python
scenarios = ['baseline', 'high_growth', 'low_growth', 'zero_migration']
results_dict = {}

for scenario in scenarios:
    projection = CohortComponentProjection(
        base_population=base_pop,
        fertility_rates=fertility,
        survival_rates=survival,
        migration_rates=migration
    )

    results = projection.run_projection(
        start_year=2025,
        end_year=2045,
        scenario=scenario
    )

    results_dict[scenario] = results

# Compare 2045 totals
for scenario, results in results_dict.items():
    total_2045 = results[results['year'] == 2045]['population'].sum()
    print(f"{scenario}: {total_2045:,.0f}")
```

## Data Format Requirements

### Critical: All DataFrames must have matching categories

If your base population has these races:
```python
['White alone, Non-Hispanic', 'Black alone, Non-Hispanic', 'Hispanic (any race)']
```

Then your fertility, survival, and migration DataFrames MUST have the same race categories.

### Age ranges

- Base population: Ages 0-90 (90+ is open-ended group)
- Fertility: Ages 15-49 only (other ages ignored)
- Survival: All ages 0-90
- Migration: All ages 0-90

### Sex values

Must be exactly:
- `'Male'`
- `'Female'`

(Case-sensitive! Not 'M'/'F' or 'male'/'female')

## Typical Parameter Values

### Fertility Rates (births per woman per year)

| Age | Typical Rate | Range |
|-----|--------------|-------|
| 15-19 | 0.015 | 0.005-0.030 |
| 20-24 | 0.075 | 0.050-0.100 |
| 25-29 | 0.110 | 0.090-0.130 |
| 30-34 | 0.100 | 0.080-0.120 |
| 35-39 | 0.050 | 0.030-0.070 |
| 40-44 | 0.015 | 0.008-0.025 |
| 45-49 | 0.003 | 0.001-0.005 |

**Total Fertility Rate (TFR)** = sum of all rates ≈ 1.6-2.1 births per woman

### Survival Rates (probability)

| Age | Typical Rate | Note |
|-----|--------------|------|
| 0 (infant) | 0.9935 | ~6.5 deaths per 1000 births |
| 1-14 | 0.9995+ | Very low child mortality |
| 15-44 | 0.999+ | Peak health |
| 45-64 | 0.995-0.999 | Middle age |
| 65-74 | 0.980-0.990 | Early seniors |
| 75-84 | 0.930-0.970 | Late seniors |
| 85-89 | 0.850-0.920 | Advanced age |
| 90+ | 0.600-0.750 | Open-ended group |

**Note:** Females typically have 1-3% higher survival rates than males

### Migration (net migrants per year)

Highly variable by location. For a small state like North Dakota:

| Age | Typical Pattern |
|-----|-----------------|
| 0-17 | Low, moves with parents |
| 18-24 | HIGH (college, first jobs) |
| 25-34 | HIGH (young professionals) |
| 35-44 | Moderate |
| 45-64 | Low |
| 65+ | Low to negative (retirees) |

**Young adults (20-35) are most mobile** - this is the most important age group for migration patterns.

## Troubleshooting

### Error: "Missing required columns"

Check your column names exactly:
- `year`, `age`, `sex`, `race`, `population` (base_population)
- `age`, `race`, `fertility_rate` (fertility_rates)
- `age`, `sex`, `race`, `survival_rate` (survival_rates)
- `age`, `sex`, `race`, `net_migration` or `migration_rate` (migration_rates)

### Error: "Missing age-sex-race combinations"

You need ALL combinations. For example, if you have:
- Ages: 0-90 (91 ages)
- Sexes: Male, Female (2)
- Races: 6 categories

You need: 91 × 2 × 6 = 1,092 rows in survival_rates

### Warning: "Negative population"

This usually means migration is too negative. Check that:
```python
population + net_migration >= 0
```

for all cohorts.

### Results seem wrong

Common issues:
1. **Fertility rates too high/low** - Check TFR = sum of rates should be 1.5-2.5
2. **Survival rates inverted** - Should be close to 1.0, not close to 0.0
3. **Migration too extreme** - Check for data errors (e.g., wrong decimal place)
4. **Race categories don't match** - Verify all DataFrames use identical race strings

## Next Steps

1. See `/examples/run_basic_projection.py` for complete working example
2. Read `/cohort_projections/core/README.md` for detailed methodology
3. Check `/docs/data_requirements.md` for data sources
4. Review `/config/projection_config.yaml` for configuration options

## Quick Data Validation

Before running projection:

```python
from cohort_projections.core import (
    validate_fertility_rates,
    validate_survival_rates,
    validate_migration_data
)

# Validate each component
fert_valid, fert_issues = validate_fertility_rates(fertility, config)
surv_valid, surv_issues = validate_survival_rates(survival, config)
mig_valid, mig_issues = validate_migration_data(migration, base_pop, config)

if not fert_valid:
    print("Fertility issues:", fert_issues)
if not surv_valid:
    print("Survival issues:", surv_issues)
if not mig_valid:
    print("Migration issues:", mig_issues)
```

This will catch most common data problems before you run the projection.
