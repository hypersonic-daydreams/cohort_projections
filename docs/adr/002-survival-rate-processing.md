# ADR-002: Survival Rate Processing Methodology

## Status
Accepted

## Date
2025-12-18

## Context

The cohort component projection engine requires age-specific survival rates by sex and race/ethnicity for all ages (0-90+). These rates must be derived from SEER (Surveillance, Epidemiology, and End Results Program) or CDC life tables, which provide death probabilities (qx), survivorship (lx), and person-years lived (Lx, Tx) in standard actuarial format.

### Challenges

1. **Life Table Format Variations**: SEER/CDC data comes in multiple formats with different column sets (some have qx only, others have full lx/Lx/Tx)
2. **Open-Ended Age Group**: Age 90+ is a special case requiring different calculation methods
3. **Conversion Methods**: Multiple valid approaches to convert life tables to survival rates
4. **Mortality Improvement**: Need to project future improvements in survival over time
5. **Data Quality**: Must validate plausibility across wide age range and demographic groups

### Requirements

- Input: SEER/CDC life tables with age, sex, race, and mortality indicators (qx, lx, or Lx/Tx)
- Output: Complete age (0-90) × sex (2) × race (6 categories) matrix ready for projection engine
- Data quality: Plausible survival patterns with age-appropriate validation
- Methodology: Demographically sound conversion and improvement methods

## Decisions

### Decision 1: Multi-Method Life Table Conversion with Automatic Selection

**Decision**: Support three calculation methods (lx, qx, Lx) with automatic method selection based on available columns.

**Methods**:
1. **lx method** (preferred): S(x) = l(x+1) / l(x)
   - Most direct and accurate
   - Uses survivorship column from life table

2. **qx method**: S(x) = 1 - q(x)
   - Uses death probability directly
   - Simple but doesn't account for within-interval effects

3. **Lx method**: S(x) = L(x+1) / L(x)
   - Uses person-years lived
   - Accounts for timing of deaths within interval

**Rationale**:
- **Flexibility**: Different SEER/CDC tables provide different columns
- **Accuracy**: lx method preferred when available (most common in SEER data)
- **Robustness**: Fallback methods ensure processing succeeds
- **Transparency**: Method used is logged and saved in metadata

**Implementation**:
```python
# Automatic method selection
if 'lx' in df.columns:
    method = 'lx'  # Preferred
    survival_rate = lx(x+1) / lx(x)
elif 'qx' in df.columns:
    method = 'qx'  # Fallback
    survival_rate = 1 - qx
else:
    raise ValueError("Need either lx or qx column")
```

**Alternatives Considered**:
- Single method only: Too restrictive, wouldn't handle all data sources
- Always use qx when available: Less accurate than lx method
- Require preprocessing: Adds complexity for users

### Decision 2: Special Handling for Age 90+ Open-Ended Group

**Decision**: Use Tx-based formula when available, otherwise use qx or default value.

**Formula for Age 90+**:
```
S(90+) = T(91) / (T(90) + L(90)/2)

Where:
- T(91) = T(90) - L(90) (person-years above age 91)
- T(90) = total person-years lived above age 90
- L(90) = person-years lived in age interval 90-91
```

**Rationale**:
- **Demographic Correctness**: Age 90+ is open-ended; people don't "age out"
- **Within-Group Survival**: Formula calculates survival probability within the 90+ group
- **Data Availability**: Uses Tx/Lx columns when available, falls back gracefully
- **Typical Values**: Results in ~0.60-0.70 survival rate, matching empirical data

**Fallback Strategy**:
1. If Tx and Lx available: Use formula above
2. If only qx available: S(90+) = 1 - q(90+)
3. If neither available: Use default S(90+) = 0.65 (typical value)

**Why This Matters**:
- Projection engine applies S(90+) to 90+ population to calculate next year's 90+ population
- Incorrect values would cause unrealistic growth/decline in oldest-old population
- Age 90+ is fastest-growing age group in population projections

**Alternatives Considered**:
- Use same method as other ages: Doesn't account for open-ended nature
- Always use default: Loses information from life tables
- Extrapolate from ages 85-89: Too much assumption, less accurate

### Decision 3: Lee-Carter Style Mortality Improvement

**Decision**: Apply linear mortality improvement to death rates with configurable annual factor (default: 0.5%).

**Formula**:
```
q(x, t) = q(x, base_year) × (1 - improvement_factor)^(t - base_year)
S(x, t) = 1 - q(x, t)
```

**Default Improvement Factor**: 0.005 (0.5% annual improvement)

**Rationale**:
- **Historical Trend**: U.S. mortality has improved ~0.5-1% annually (varies by age)
- **Simplicity**: Linear improvement is transparent and widely used
- **Configurability**: Users can adjust based on assumptions
- **Consistency**: Matches Census Bureau and state demographer practices

**Implementation Notes**:
- Improvement applied to death rates (q), not survival rates (S)
- Survival rates capped at 1.0 (cannot exceed 100% survival)
- No improvement if projection_year <= base_year
- Improvement factor can be set to 0 for constant mortality scenario

**Typical Improvement Factors**:
- Optimistic: 1.0% (0.010)
- Baseline: 0.5% (0.005)
- Conservative: 0.25% (0.0025)
- Constant mortality: 0.0%

**Alternatives Considered**:
- Age-specific improvement: More complex, requires additional data
- Lee-Carter full model: Requires time series, overkill for state projections
- No improvement: Unrealistic for long-term projections
- Cohort-based improvement: More accurate but much more complex

### Decision 4: Age-Specific Plausibility Thresholds

**Decision**: Implement differentiated validation thresholds based on age group.

**Validation Thresholds**:

| Age Group | Expected Range | Error If Outside | Warning If Outside |
|-----------|----------------|------------------|-------------------|
| 0 (infant) | 0.993-0.995 | < 0.990 or > 0.998 | < 0.993 or > 0.995 |
| 1-14 (children) | > 0.9995 | < 0.999 | < 0.9995 |
| 15-44 (young adults) | > 0.999 | < 0.995 | < 0.999 |
| 45-64 (middle age) | 0.985-0.998 | < 0.98 or > 0.999 | - |
| 65-84 (elderly) | 0.93-0.98 | < 0.90 or > 0.99 | < 0.93 or > 0.98 |
| 90+ (oldest-old) | 0.60-0.70 | < 0.50 or > 0.80 | < 0.60 or > 0.70 |

**Life Expectancy Validation** (at birth, e0):
- Typical range: 75-87 years (varies by sex and race)
- Warning if < 70 or > 90 years
- Calculated for each sex-race combination

**Rationale**:
- **Age Patterns**: Mortality risk varies dramatically by age
- **Infant Mortality**: First year has higher mortality, then very low until teens
- **Young Adult Spike**: Slight increase in late teens/20s (accidents, risky behavior)
- **Elderly Acceleration**: Mortality rises exponentially with age after 65
- **Data Quality**: Catches data errors and implausible values early

**Error vs. Warning**:
- **Errors**: Fail validation, prevent processing (likely data errors)
- **Warnings**: Flag for review but allow processing (possible but unusual)

### Decision 5: Age-Appropriate Default Values for Missing Data

**Decision**: Fill missing age-sex-race combinations with age-specific defaults rather than single global value.

**Default Survival Rates by Age**:
- Age 0 (infant): 0.994
- Ages 1-14 (children): 0.9995
- Ages 15-64 (adults): 0.997
- Ages 65-89 (elderly): 0.95
- Age 90+: 0.65

**Rationale**:
- **Age Variation**: Survival rates vary by 0.3+ across age groups
- **Conservative**: Moderate values avoid extreme assumptions
- **Demographic Validity**: Based on typical U.S. patterns
- **Transparency**: Logged clearly in processing output

**Why Not Zero or Imputation**:
- Zero survival would eliminate populations (too extreme)
- Imputation (interpolation, regression) introduces complex assumptions
- Age-specific defaults are transparent and defensible
- Missing data should be rare; defaults are safety net

**When Defaults Are Used**:
- Rare race-age-sex combinations not in source data
- Data gaps in SEER tables for small populations
- Always logged as warnings in metadata

### Decision 6: Life Expectancy Calculation for Quality Assurance

**Decision**: Calculate simplified life expectancy (e0) for all sex-race combinations as validation metric.

**Method**:
```python
# Simplified calculation
lx = cumulative_product(survival_rates)  # Survival to each age
e0 = sum(lx)  # Sum of person-years ≈ life expectancy
```

**Notes**:
- This is an approximation; full life table method uses Lx and Tx
- Accurate enough for validation purposes
- Provides quick sanity check of processed rates

**Typical e0 Values** (United States, 2020-2023):

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

**Usage**:
- Saved in metadata for every processing run
- Logged to console for immediate review
- Compared against published CDC values for validation
- Flags generated if e0 outside typical ranges

**Alternatives Considered**:
- Skip e0 calculation: Loses valuable QA metric
- Full life table calculation: Overkill, requires more data
- Use source life table e0: Not always available after processing

## Consequences

### Positive

1. **Flexibility**: Handles multiple SEER/CDC life table formats automatically
2. **Accuracy**: Uses best available method (lx) when possible
3. **Robustness**: Age-specific validation catches errors early
4. **Future-Ready**: Mortality improvement projects realistic trends
5. **Quality Assurance**: Life expectancy calculation provides validation
6. **Transparency**: All decisions logged and saved in metadata
7. **Consistency**: Follows same patterns as fertility_rates processor

### Negative

1. **Complexity**: Multiple conversion methods add code complexity
2. **Approximation**: Age 90+ formula is an approximation
3. **Assumptions**: Mortality improvement is linear, may not reflect reality
4. **Default Values**: Missing data filled with defaults may need review
5. **Validation Overhead**: Comprehensive checks add processing time

### Risks and Mitigations

**Risk**: Open-ended age group formula produces implausible values
- **Mitigation**: Validation checks age 90+ rates; defaults to 0.65 if calculation fails
- **Action**: Review age 90+ rates in validation output

**Risk**: Mortality improvement factor inappropriate for population
- **Mitigation**: Configurable factor; can be set to 0 for constant mortality
- **Action**: Consult demographic literature for population-specific factors

**Risk**: Missing Tx/Lx columns prevent age 90+ calculation
- **Mitigation**: Graceful fallback to qx method or default value
- **Action**: Use complete SEER life tables when available

**Risk**: Life expectancy values outside typical ranges
- **Mitigation**: Validation warnings flag unusual values
- **Action**: Investigate source data quality; compare to published CDC values

## Alternatives Considered

### Alternative 1: Single Conversion Method (lx Only)

**Not Chosen Because**:
- Would fail on life tables without lx column
- SEER provides multiple formats; inflexible
- Requires preprocessing by users

**When to Reconsider**: If standardizing on single SEER data source

### Alternative 2: Abridged Life Tables (5-Year Age Groups)

**Not Chosen Because**:
- Projection engine requires single-year ages
- Loss of detail in age patterns
- Would require interpolation (adds complexity and error)

**When to Reconsider**: If switching to 5-year age group projections

### Alternative 3: Age-Specific Mortality Improvement Rates

**Not Chosen Because**:
- Requires detailed time series data
- Much more complex to implement and validate
- Marginal gain in accuracy for state-level projections

**When to Reconsider**: If doing detailed scenario analysis or long-term (50+ year) projections

### Alternative 4: Bayesian Interpolation for Missing Data

**Not Chosen Because**:
- Significant statistical complexity
- Black box for users
- Overkill when missing data is rare
- Transparent age-specific defaults preferred

**When to Reconsider**: If missing data exceeds 10% of combinations

### Alternative 5: Sex-Specific Mortality Improvement Factors

**Not Chosen Because**:
- Adds complexity
- Improvement rates similar for males and females in recent decades
- Can be implemented in scenario module if needed

**When to Reconsider**: For detailed gender-specific projection scenarios

## Implementation Notes

### Key Functions

1. `load_life_table_data()`: Multi-format loader with flexible column recognition
2. `harmonize_mortality_race_categories()`: Maps SEER codes to 6 standard categories
3. `calculate_survival_rates_from_life_table()`: Converts life table to survival rates (3 methods)
4. `apply_mortality_improvement()`: Applies Lee-Carter style improvement
5. `create_survival_rate_table()`: Complete age × sex × race matrix with defaults
6. `validate_survival_rates()`: Age-specific plausibility checks
7. `calculate_life_expectancy()`: Simplified e0 calculation for QA
8. `process_survival_rates()`: Main pipeline orchestrator

### Configuration Integration

Uses `projection_config.yaml` for:
- `demographics.age_groups`: Age range (0-90)
- `demographics.sex`: Expected sex categories
- `demographics.race_ethnicity.categories`: Expected race categories (6)
- `rates.mortality.life_table_year`: Base year for life table
- `rates.mortality.improvement_factor`: Annual mortality improvement (default: 0.005)
- `output.compression`: Parquet compression method

### Output Files

All files saved to `data/processed/mortality/`:
1. **survival_rates.parquet**: Primary output (compressed, efficient)
2. **survival_rates.csv**: Human-readable backup
3. **survival_rates_metadata.json**: Processing metadata and provenance

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
    ...
  },
  "validation_warnings": [],
  "config_used": {...}
}
```

### Testing Strategy

**Unit Tests** (recommended):
- Test each conversion method (lx, qx, Lx) with known values
- Test age 90+ calculation with and without Tx/Lx
- Test mortality improvement calculations
- Test validation with edge cases
- Test default value application

**Integration Tests** (recommended):
- End-to-end processing with synthetic SEER-format life table
- Verify output format matches projection engine requirements
- Verify life expectancy values are plausible
- Test with multiple file formats

**Validation Tests** (required):
- Compare processed survival rates to published CDC life tables
- Verify life expectancy values match published statistics
- Visual inspection of survival curves (should be smooth, declining)
- Check age-specific patterns match demographic literature

## References

1. **SEER Life Tables**: https://seer.cancer.gov/popdata/methods.html
2. **CDC Life Tables**: https://www.cdc.gov/nchs/products/life_tables.htm
3. **Lee-Carter Model**: Lee, R.D. and Carter, L.R. (1992). "Modeling and forecasting U.S. mortality." JASA.
4. **Preston et al.**: "Demography: Measuring and Modeling Population Processes" (2001)
5. **UN Methods**: "Manual X: Indirect Techniques for Demographic Estimation"
6. **Census Bureau**: "Methodology for the United States Population Projections"
7. **SSA Actuarial Tables**: Social Security Administration Life Table Methods

## Revision History

- **2025-12-18**: Initial version (ADR-002) - Survival rate processing methodology

## Related ADRs

- ADR-001: Fertility rate processing (established pattern to follow)
- ADR-003: Migration rate processing (planned)
- ADR-004: Projection scenario design (planned)

## Appendix: Mathematical Formulas

### Life Table Relationships

**Basic definitions**:
- qx = probability of dying between age x and x+1
- px = probability of surviving = 1 - qx
- lx = number surviving to age x (out of radix, typically 100,000)
- dx = number dying = lx × qx
- Lx = person-years lived between x and x+1 ≈ (lx + lx+1) / 2
- Tx = total person-years above age x = Σ(Lx) for all ages ≥ x
- ex = life expectancy at age x = Tx / lx

**Survival rate conversions**:
1. From lx: S(x) = lx+1 / lx = 1 - dx/lx = 1 - qx = px
2. From qx: S(x) = 1 - qx = px
3. From Lx: S(x) ≈ Lx+1 / Lx (approximation)

**Age 90+ formula** (derived from life table properties):
```
S(90+) = T(91) / (T(90) + L(90)/2)

Derivation:
- Population at 90+ next year = survivors within 90+ group
- T(91) = person-years above age 91 = years lived by survivors
- T(90) + L(90)/2 = person-years at risk in 90+ group
- Ratio gives survival probability within open-ended group
```

### Mortality Improvement

**Lee-Carter simplification**:
```
q(x, t) = q(x, base) × exp(β(t - base))

Linear approximation:
q(x, t) ≈ q(x, base) × (1 - α)^(t - base)

Where:
- α = improvement factor (e.g., 0.005 for 0.5% annual improvement)
- t = projection year
- base = base year of life table
```

**Then convert to survival**:
```
S(x, t) = 1 - q(x, t)
```

This is the method implemented in the processor.
