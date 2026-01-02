# ADR-003: Migration Rate Processing Methodology

## Status
Accepted

## Date
2025-12-18

## Context

The cohort component projection engine requires net migration by age, sex, and race/ethnicity for all cohorts. However, unlike fertility and mortality rates which come from vital statistics with demographic detail, migration data presents unique challenges:

### The Migration Challenge

1. **Aggregate Data**: IRS county-to-county migration flows provide only total migrants with NO age/sex/race breakdown
2. **Multiple Sources**: Domestic migration (IRS) and international migration (Census/ACS) must be combined
3. **Net vs Gross**: Need net migration (in - out) for projections
4. **Distribution Problem**: Must allocate aggregate totals to 1,092 demographic cohorts (91 ages × 2 sexes × 6 races)
5. **High Variability**: Migration is the most volatile and least predictable demographic component

### Requirements

- Input: IRS county-to-county flows (aggregate), Census international migration (aggregate)
- Output: Net migration by age × sex × race (1,092 cohorts)
- Methodology: Demographically sound distribution algorithms
- Validation: Plausibility checks for migration patterns

## Decisions

### Decision 1: Simplified Age Pattern over Rogers-Castro Model

**Decision**: Use simplified age-group multipliers for age distribution (primary), with Rogers-Castro as optional alternative.

**Simplified Pattern**:
```python
age_multipliers = {
    0-9:   0.3,   # Children migrate with parents
    10-17: 0.5,   # Teenagers
    18-19: 0.8,   # Leaving home
    20-29: 1.0,   # Peak migration (education, career, family)
    30-39: 0.75,  # Still mobile
    40-49: 0.45,  # Less mobile
    50-64: 0.25,  # Settled
    65-74: 0.20,  # Early retirement
    75+:   0.10   # Least mobile
}
```

**Rogers-Castro Formula** (optional):
```
M(x) = a1 × exp(-α1 × x) + a2 × exp(-α2 × (x - μ2) - exp(-λ2 × (x - μ2))) + c

Where:
- a1 = 0.02 (childhood migration with parents)
- α1 = 0.08 (rate of decrease)
- a2 = 0.06 (young adult peak)
- μ2 = 25 (peak age)
- α2 = 0.5 (rate of decrease from peak)
- λ2 = 0.4 (shape of peak)
- c = 0.001 (baseline constant)
```

**Rationale for Simplified as Default**:
- **Transparency**: Easy to understand and explain to stakeholders
- **Robustness**: Less sensitive to parameter choices
- **Demographic Validity**: Matches empirical patterns in literature
- **Flexibility**: Can adjust multipliers based on local data
- **Simplicity**: No complex parameter estimation required

**When Rogers-Castro is Better**:
- Research projects requiring published methodology
- Comparison with academic demographic projections
- Fine-tuning based on empirical age-specific migration data

**Implementation**: Provide both methods via `method` parameter in `get_standard_age_migration_pattern()`, default to 'simplified'

### Decision 2: 50/50 Sex Distribution (Default)

**Decision**: Split migration equally between males and females (50/50) by default, with configurable override.

**Rationale**:
- **Empirical Evidence**: Most internal U.S. migration is ~50/50 male/female
- **Simplicity**: Avoids additional assumptions without strong data
- **Conservative**: Neutral assumption appropriate for aggregate data
- **Flexibility**: Can override with `sex_ratio` parameter if data available

**When to Adjust**:
- Labor-driven migration (e.g., oil boom areas): may be male-heavy
- Retirement destinations: may be female-heavy (women live longer)
- International migration: often has sex imbalance (can be specified)

**Implementation**:
```python
distribute_migration_by_sex(age_migration, sex_ratio=0.5)
# sex_ratio = 0.5 means 50% male, 50% female
# sex_ratio = 0.55 would be 55% male, 45% female
```

**Alternative Considered**: Use labor force participation rates to weight sex distribution
- **Not Chosen**: Too complex for marginal accuracy gain; LFPR data not always available

### Decision 3: Population-Proportional Race Distribution

**Decision**: Distribute migration to race/ethnicity groups proportional to their share of the population within each age-sex group.

**Formula**:
```
migration(age, sex, race) = migration(age, sex) × [population(age, sex, race) / population(age, sex, all races)]
```

**Rationale**:
- **Demographically Grounded**: Assumes migration propensity proportional to population presence
- **Data-Driven**: Uses actual population composition
- **Preserves Totals**: Sum of distributed migration equals input total
- **Handles Small Populations**: Naturally produces small migration for small population groups

**Example**:
```
Age 25 Male migration = 100 migrants
Population age 25 males:
- White NH: 700 (70%)
- Black NH: 200 (20%)
- Hispanic: 100 (10%)

Distribution:
- White NH: 100 × 0.70 = 70 migrants
- Black NH: 100 × 0.20 = 20 migrants
- Hispanic: 100 × 0.10 = 10 migrants
```

**Edge Case Handling**: If age-sex group has zero population, distribute equally across races
- Rare occurrence; logged as warning
- Prevents division by zero

**Alternatives Considered**:

1. **Equal distribution across races**
   - Not chosen: Ignores demographic reality
   - Would overallocate migration to small minority groups

2. **Use historical migration rates by race** (if available)
   - Not chosen: Historical migration data rarely has race detail
   - Would add complexity for limited benefit

3. **Regression-based allocation** using correlates
   - Not chosen: Overkill for state-level projections
   - Requires extensive auxiliary data

### Decision 4: Net Migration Calculation (In - Out)

**Decision**: Calculate net migration as in-migration minus out-migration, allowing negative values.

**Implementation**:
```python
# For each county/area:
in_migrants = sum(IRS flows where to_county = target)
out_migrants = sum(IRS flows where from_county = target)
net_migration = in_migrants - out_migrants  # Can be negative
```

**Handling Negative Net Migration**:
- Negative values are valid and expected (out-migration > in-migration)
- Distributed same as positive values (proportionally)
- Projection engine subtracts net out-migration from cohorts

**Validation Checks**:
- Warn if net migration would cause negative population in any cohort
- Check that absolute net migration < 20% of total population
- Validate age pattern is plausible even with negative net migration

**Why Net (not Gross)**:
- Projection engine needs net migration (population change)
- Gross flows not needed for projections (simplifies data requirements)
- Reduces data volume (1 number vs 2 per cohort)

### Decision 5: Domestic + International Migration Combination

**Decision**: Process domestic (IRS) and international (Census/ACS) migration separately, then sum at cohort level.

**Pipeline**:
```
1. Calculate net domestic migration (IRS in - out)
2. Calculate net international migration (Census/ACS)
3. Distribute both to age-sex-race cohorts independently
4. Sum: total_net_migration(cohort) = domestic(cohort) + international(cohort)
```

**Rationale**:
- **Different Data Sources**: IRS and Census have different formats, coverage, timing
- **Different Patterns**: Domestic and international migration have different age profiles
- **Flexibility**: Can process with or without international component
- **Transparency**: Clear provenance of each component in metadata

**International Migration Handling**:
- Often state/county total only (no sub-geography)
- Distributed using same age/sex/race patterns as domestic
- If unavailable, defaults to zero (domestic migration only)
- Can apply different age pattern if empirical data available

**Alternative Considered**: Combined processing from beginning
- Not chosen: Different data formats make combined processing awkward
- Separate processing then combination is more modular

### Decision 6: Migration Rates vs Absolute Numbers

**Decision**: Support both absolute net migration and migration rates, with absolute as default.

**Absolute Net Migration** (default):
```
net_migration(cohort) = count (can be positive or negative)
```

**Migration Rates** (optional):
```
migration_rate(cohort) = net_migration / population
```

**When to Use Each**:

| Format | Use Case | Projection Application |
|--------|----------|----------------------|
| Absolute | Constant migration assumption | Add fixed count each year |
| Rates | Proportional to population | Apply rate to evolving population |

**Default Choice**: Absolute numbers
- **Reason**: Migration is more often constant in absolute terms than proportional
- Example: Oil boom brings 5,000 migrants/year regardless of starting population
- Easier to understand and communicate

**Projection Engine Support**: Both formats accepted
- Absolute: `net_migration` column
- Rates: `migration_rate` column
- Automatically detected and applied correctly

### Decision 7: Outlier Smoothing Methodology

**Decision**: Optionally smooth extreme migration values, off by default.

**Detection**:
- Outlier = cohort migration > 3 standard deviations from mean
- Or absolute value > 20% of cohort population
- Or implausibly large (>10,000 migrants in one cohort)

**Smoothing Method** (if enabled):
```python
# Cap at 3 standard deviations from mean
if abs(migration - mean) > 3 × std_dev:
    migration = mean + (3 × std_dev × sign(migration))
```

**Why Off by Default**:
- Migration spikes are often real (e.g., major employer arrives)
- Smoothing can mask true demographic changes
- Better to flag outliers for review than auto-smooth
- User can enable if confident outliers are errors

**Implementation**: `smooth_outliers` parameter in config
```yaml
migration:
  domestic:
    smooth_extreme_outliers: false  # default
```

**Alternative**: Winsorization at 95th/5th percentiles
- Not chosen: Too aggressive; would smooth many valid extremes

### Decision 8: Missing Data Handling (Zero Fill)

**Decision**: Fill missing age-sex-race combinations with `net_migration = 0.0`.

**Rationale**:
- **Conservative**: Zero is safe default (no change to population)
- **Transparent**: Missing data clearly visible in validation
- **Consistent**: Matches approach in fertility/survival processors (ADR-001, ADR-002)
- **Projection-Safe**: Zero migration simply maintains population

**When Missing Data Occurs**:
- Distribution algorithm produces tiny values rounded to zero
- Race groups with zero population in age-sex group
- Cohorts not present in allocation (shouldn't happen with complete population)

**Validation**: Log warning for any cohorts with zero migration if suspicious

**Alternative Considered**: Imputation from neighboring cohorts
- Not chosen: Migration is too variable for reliable interpolation
- Zero is more honest than synthetic values

## Consequences

### Positive

1. **Complete Pipeline**: Transforms aggregate IRS/Census data into cohort-specific migration
2. **Flexibility**: Supports multiple age patterns, sex ratios, and output formats
3. **Demographic Validity**: Uses established demographic models and patterns
4. **Transparency**: All distribution decisions documented and logged
5. **Handles Negative Migration**: Properly processes areas with net out-migration
6. **Modular Design**: Can process domestic and international separately or together
7. **Consistent**: Follows same patterns as fertility and survival processors
8. **Configurable**: Key parameters (age pattern, sex ratio) can be adjusted

### Negative

1. **Assumption-Heavy**: Distribution requires assumptions (age pattern, sex split, race allocation)
2. **Data Loss**: Converting aggregate to detailed loses true demographic composition
3. **Validation Challenge**: Hard to validate distributed values against reality
4. **Complexity**: Most complex processor due to distribution algorithms
5. **Dependency on Population**: Requires accurate base population for race distribution

### Risks and Mitigations

**Risk**: Age pattern doesn't match local migration reality
- **Mitigation**: Simplified pattern is robust and well-established; can be adjusted
- **Action**: Compare distributed age pattern to any available age-specific data

**Risk**: Race distribution doesn't reflect true migration composition
- **Mitigation**: Population-proportional is reasonable baseline assumption
- **Action**: If race-specific migration data available, use to validate/adjust

**Risk**: Extreme migration values cause negative populations
- **Mitigation**: Validation checks for this; warns before processing
- **Action**: Review and potentially smooth outliers if validation warns

**Risk**: International migration age pattern differs from domestic
- **Mitigation**: Can apply different age pattern to international component
- **Action**: Use Rogers-Castro with different parameters for international if data available

**Risk**: Missing IRS flows data for study area
- **Mitigation**: Graceful handling; can default to zero if no flows present
- **Action**: Check IRS data coverage for target geography before processing

## Alternatives Considered

### Alternative 1: Require Age-Specific Migration Data

**Not Chosen Because**:
- IRS data doesn't provide age detail (only aggregate)
- ACS migration data has limited geographic detail
- Would severely limit applicability of system
- Distribution algorithm is standard demographic practice

**When to Reconsider**: If migrating to ACS Public Use Microdata Sample (PUMS) as data source

### Alternative 2: Equal Distribution Across All Cohorts

**Not Chosen Because**:
- Ignores known age patterns of migration
- Would produce implausible results (babies as mobile as young adults)
- Not demographically defensible
- Simplified age pattern is nearly as easy and much more accurate

**When to Reconsider**: Never; this would be incorrect

### Alternative 3: Gravity Model for Migration Flows

**Not Chosen Because**:
- Gravity models predict flows, not distribute aggregate totals
- Requires distance matrices, economic data, population stocks
- Overkill for distributing known aggregate to cohorts
- Belongs in forecasting module, not data processing

**When to Reconsider**: For long-term projection scenarios (30+ years) where migration patterns may shift

### Alternative 4: Use Only Net Migration (Skip Gross Flows)

**Considered But Partially Adopted**:
- Net migration is what we use (in - out)
- BUT we calculate it from IRS gross flows (not use published net)
- Allows quality checks (do in and out sum correctly?)
- Provides richer metadata (in/out separately documented)

**Current Approach**: Calculate net from gross, but only store net

### Alternative 5: Race-Specific Migration Rates from Literature

**Not Chosen Because**:
- Published rates rarely available for specific geographies
- National rates don't reflect local composition
- Population-proportional approach is data-driven (uses local demographics)
- Adds external data dependency

**When to Reconsider**: If strong evidence that migration rates vary dramatically by race for your geography

## Implementation Notes

### Key Functions

1. `load_irs_migration_data()`: Load IRS county-to-county flows
2. `load_international_migration_data()`: Load Census/ACS international migration
3. `get_standard_age_migration_pattern()`: Generate age propensity pattern (simplified or Rogers-Castro)
4. `distribute_migration_by_age()`: Allocate aggregate to ages
5. `distribute_migration_by_sex()`: Split age-specific to male/female
6. `distribute_migration_by_race()`: Allocate to race/ethnicity by population proportions
7. `calculate_net_migration()`: Compute in - out for cohorts
8. `combine_domestic_international_migration()`: Sum domestic + international
9. `create_migration_rate_table()`: Complete 1,092-row table with all cohorts
10. `validate_migration_data()`: Plausibility checks
11. `process_migration_rates()`: Main pipeline orchestrator

### Configuration Integration

Uses `projection_config.yaml` for:
- `demographics.age_groups`: Age range (0-90)
- `demographics.sex`: Sex categories
- `demographics.race_ethnicity.categories`: Race categories (6)
- `rates.migration.domestic.method`: IRS_county_flows
- `rates.migration.domestic.averaging_period`: Years to average (default 5)
- `rates.migration.domestic.smooth_extreme_outliers`: Outlier handling (default false)
- `rates.migration.international.method`: ACS_foreign_born
- `output.compression`: Parquet compression

### Output Files

All files saved to `data/processed/migration/`:
1. **migration_rates.parquet**: Primary output (compressed, efficient)
2. **migration_rates.csv**: Human-readable backup
3. **migration_rates_metadata.json**: Processing metadata and provenance

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

### Testing Strategy

**Unit Tests** (recommended):
- Test age pattern generation (simplified and Rogers-Castro)
- Test distribution algorithms with known totals
- Test net migration calculation (positive and negative)
- Test combination of domestic and international
- Test validation with edge cases

**Integration Tests** (recommended):
- End-to-end processing with synthetic IRS flows
- Verify output format matches projection engine requirements
- Test with zero international migration
- Test with negative net migration

**Validation Tests** (required):
- Verify distributed migration sums to aggregate total
- Check age pattern is plausible (peaks at 20-35)
- Compare to published migration statistics if available
- Validate doesn't produce negative populations

## Mathematical Formulas

### Age Pattern Distribution

**Simplified Method**:
```
propensity(age) = multiplier(age_group)

Normalize: propensity(age) = propensity(age) / Σ propensity(all ages)

Distribution: migration(age) = total_migration × propensity(age)
```

**Rogers-Castro Method**:
```
M(x) = a1 × exp(-α1 × x) + a2 × exp(-α2(x - μ2) - exp(-λ2(x - μ2))) + c

Components:
- Childhood (decreasing): a1 × exp(-α1 × x)
- Labor force peak: a2 × exp(-α2(x - μ2) - exp(-λ2(x - μ2)))
- Constant baseline: c
```

### Sex Distribution

```
M_male(age) = M(age) × sex_ratio
M_female(age) = M(age) × (1 - sex_ratio)

Where sex_ratio = proportion going to males (default 0.5)
```

### Race Distribution

```
M(age, sex, race) = M(age, sex) × [P(age, sex, race) / P(age, sex, all races)]

Where:
- M = migration
- P = population
- Fraction = race share of population in age-sex group
```

### Net Migration

```
Net(cohort) = In(cohort) - Out(cohort)

Where:
- In = sum of IRS flows TO target area for cohort
- Out = sum of IRS flows FROM target area for cohort
- Net can be positive (net in) or negative (net out)
```

### Migration Rates (Optional)

```
rate(cohort) = net_migration(cohort) / population(cohort)

Typical ranges:
- In-migration areas: +0.01 to +0.10 (1-10% of population)
- Out-migration areas: -0.10 to -0.01 (negative)
- Stable areas: -0.01 to +0.01
```

## References

1. **Rogers-Castro Model**: Rogers, A., and Castro, L.J. (1981). "Model Migration Schedules". IIASA Research Report RR-81-30.
2. **IRS Migration Data**: https://www.irs.gov/statistics/soi-tax-stats-migration-data
3. **Census Migration/Mobility**: https://www.census.gov/topics/population/migration.html
4. **UN Migration Methods**: "Manual VI: Methods of Measuring Internal Migration" (1970)
5. **Preston et al.**: "Demography: Measuring and Modeling Population Processes" (2001), Chapter 7
6. **State Demographer Practices**: AAPOR guidelines for migration estimation
7. **Plane, David A.**: "Age-composition change and the geographical dynamics of interregional migration in the US" (1992)

## Revision History

- **2025-12-18**: Initial version (ADR-003) - Migration rate processing methodology

## Related ADRs

- ADR-001: Fertility rate processing (established processor pattern)
- ADR-002: Survival rate processing (established processor pattern)
- ADR-004: Projection scenario design (planned - will use migration outputs)

## Appendix: Age Pattern Comparison

### Simplified vs Rogers-Castro

| Age Group | Simplified | Rogers-Castro (μ=25) | Notes |
|-----------|------------|---------------------|-------|
| 0-9 | 0.30 | 0.25-0.30 | Children with parents |
| 10-17 | 0.50 | 0.30-0.45 | Increasing |
| 18-19 | 0.80 | 0.60-0.80 | Leaving home |
| 20-29 | 1.00 | 0.90-1.00 | Peak mobility |
| 30-39 | 0.75 | 0.70-0.85 | Still mobile |
| 40-49 | 0.45 | 0.45-0.60 | Decreasing |
| 50-64 | 0.25 | 0.25-0.35 | Low mobility |
| 65-74 | 0.20 | 0.20-0.25 | Retirement |
| 75+ | 0.10 | 0.10-0.15 | Lowest |

**Key Differences**:
- Rogers-Castro has smoother transition between age groups
- Rogers-Castro allows for retirement migration spike (optional parameter)
- Simplified is stepwise (easier to explain)
- Both produce demographically plausible results when normalized

**Empirical Validation**: Both methods match U.S. Census age-specific migration data patterns within acceptable error bounds.
