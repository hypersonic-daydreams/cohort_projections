# ADR-011: Testing Strategy

## Status
Accepted

## Date
2025-12-18

## Context

A population projection system involves complex mathematical calculations, data transformations, and multi-stage processing. Testing is essential to ensure correctness, prevent regressions, and build confidence in results. However, comprehensive automated testing requires significant upfront investment.

### Requirements

1. **Correctness**: Verify mathematical calculations are accurate
2. **Regression Prevention**: Catch bugs before deployment
3. **Documentation**: Tests serve as executable documentation
4. **Confidence**: Stakeholders trust projection results
5. **Development Speed**: Tests shouldn't slow development excessively
6. **Maintainability**: Tests should be easy to update as code evolves

### Challenges

1. **Stochastic Uncertainty**: Some demographic processes have inherent variability
2. **Complex Workflows**: End-to-end projections involve many steps
3. **Data Dependencies**: Tests need input data (real or synthetic)
4. **Validation Difficulty**: Hard to know "correct" answer for novel projections
5. **Resource Constraints**: Limited time for test development
6. **Changing Requirements**: Specifications evolve during development

## Decision

### Decision 1: Pragmatic Testing Approach (Not Full TDD)

**Decision**: Use a pragmatic testing approach emphasizing built-in validation and example scripts rather than comprehensive unit test coverage.

**Testing Pyramid for This Project**:

```
        ┌─────────────────┐
        │  Manual Review  │  ← Compare to Census benchmarks
        │   & Validation  │     Visual inspection of outputs
        └─────────────────┘
             ▲
             │
        ┌─────────────────┐
        │   Integration   │  ← Example scripts (examples/)
        │     Examples    │     End-to-end demonstrations
        └─────────────────┘
             ▲
             │
        ┌─────────────────┐
        │   Built-In      │  ← Validation functions
        │   Validation    │     Plausibility checks
        └─────────────────┘     Defensive programming
             ▲
             │
        ┌─────────────────┐
        │   Unit Tests    │  ← Critical functions
        │   (Selective)   │     Mathematical operations
        └─────────────────┘
```

**Rationale**:

**Why Not Full TDD** (Test-Driven Development):
- Research/analytical codebase, not production SaaS
- Specifications evolved during development
- Correctness validated against demographic benchmarks (not pure logic)
- Resource constraints (solo developer, time limits)

**Why This Approach**:
- **Built-in Validation**: Catches most errors immediately
- **Examples as Tests**: Demonstrate usage and verify end-to-end
- **Selective Unit Tests**: Cover critical calculations
- **Manual Validation**: Ultimate check against known results

**Trade-off Acknowledged**: Lower automated test coverage, but appropriate for this project context.

### Decision 2: Built-In Validation Functions (Primary Quality Assurance)

**Decision**: Comprehensive validation functions in every module as the primary quality assurance mechanism.

**Validation Levels**:

**1. Input Validation** (at module entry):
```python
def validate_fertility_rates(df, config):
    """Validate fertility rate data structure and values."""
    issues = []

    # Structure validation
    required = ['age', 'race', 'fertility_rate']
    if not all(c in df.columns for c in required):
        issues.append("ERROR: Missing required columns")
        return False, issues

    # Value validation
    if (df['fertility_rate'] < 0).any():
        issues.append("ERROR: Negative fertility rates")
        return False, issues

    if (df['fertility_rate'] > 0.15).any():
        issues.append("ERROR: Implausible rate > 0.15")
        return False, issues

    # Plausibility checks (warnings)
    tfr = df.groupby('race')['fertility_rate'].sum()
    for race, value in tfr.items():
        if not 1.0 <= value <= 3.0:
            issues.append(f"WARNING: TFR {race} = {value:.2f}")

    return len([i for i in issues if 'ERROR' in i]) == 0, issues
```

**2. Runtime Validation** (during processing):
```python
def project_single_year(population, year):
    # Calculate births
    births = calculate_births(population, fertility_rates, year)

    # Validate births are reasonable
    if births['population'].sum() < 0:
        raise ValueError("Negative total births")

    expected_births = population[population['sex'] == 'Female'].sum() * 0.05
    if births['population'].sum() > expected_births * 2:
        logger.warning(f"Unusually high births: {births['population'].sum()}")

    # Continue processing...
```

**3. Output Validation** (before returning results):
```python
def run_projection(start_year, end_year):
    results = []
    for year in range(start_year, end_year + 1):
        pop = project_single_year(population, year)

        # Validate each year
        if (pop['population'] < 0).any():
            logger.error(f"Negative populations in year {year}")
            pop['population'] = pop['population'].clip(lower=0)

        results.append(pop)

    # Final validation
    validate_projection_results(results)
    return pd.concat(results)
```

**Rationale**:
- **Immediate Feedback**: Errors caught at source
- **Defensive Programming**: Assume inputs might be wrong
- **User-Friendly**: Clear error messages
- **Documentation**: Validation documents expected values
- **No Separate Test**: Validation runs every execution

### Decision 3: Example Scripts as Integration Tests

**Decision**: Comprehensive example scripts in `examples/` serve as integration tests and documentation.

**Example Scripts**:

**`examples/run_basic_projection.py`** (234 lines):
- Creates synthetic test data
- Runs full projection
- Validates results
- Demonstrates API usage

**Key Sections**:
```python
def create_sample_data():
    """Create realistic synthetic data for testing."""
    # Base population
    base_pop = create_population_matrix(ages, sexes, races)

    # Fertility rates
    fertility = create_fertility_schedule()

    # Survival rates
    survival = create_life_table()

    # Migration rates
    migration = create_migration_patterns()

    return base_pop, fertility, survival, migration

def run_projection_example():
    """Run projection and validate results."""
    pop, fert, surv, mig = create_sample_data()

    projection = CohortComponentProjection(pop, fert, surv, mig)
    results = projection.run_projection(2025, 2030)

    # Validate results
    assert len(results) == 6 * 91 * 2 * 3  # 6 years × cohorts
    assert results['population'].sum() > 0
    assert not (results['population'] < 0).any()

    # Check population balance
    initial = pop['population'].sum()
    final = results[results['year'] == 2030]['population'].sum()
    growth = (final - initial) / initial
    assert -0.1 < growth < 0.3  # Growth between -10% and +30%

    print("✓ All validation checks passed")
```

**Usage**:
```bash
# Run as integration test
python examples/run_basic_projection.py

# Expected output:
# Processing complete
# ✓ All validation checks passed
```

**Rationale**:
- **Executable Documentation**: Shows how to use system
- **End-to-End Validation**: Tests complete workflows
- **Regression Detection**: Breaking changes caught
- **User Testing**: Users can verify their environment

### Decision 4: Selective Unit Tests for Critical Functions

**Decision**: Write unit tests for critical mathematical operations and edge cases, not comprehensive coverage.

**Priority for Unit Testing**:

**High Priority** (should have tests):
1. **Mathematical Functions**:
   - Survival rate calculations
   - Birth calculations
   - Migration application
   - Age advancement logic

2. **Data Transformations**:
   - Race category mapping
   - Column name harmonization
   - Missing value handling

3. **Edge Cases**:
   - Empty populations
   - Age 90+ handling
   - Negative migration
   - Zero fertility rates

**Low Priority** (can skip):
- Simple getter/setter methods
- Configuration loading (tested via usage)
- Logging functions
- I/O operations (file reading/writing)

**Example Unit Test**:
```python
# tests/test_core/test_fertility.py
import pytest
from cohort_projections.core import calculate_births

def test_calculate_births_basic():
    """Test basic birth calculation."""
    # Setup: 1000 women age 25, fertility rate 0.10
    female_pop = pd.DataFrame({
        'age': [25],
        'sex': ['Female'],
        'race': ['White alone, Non-Hispanic'],
        'population': [1000]
    })

    fertility_rates = pd.DataFrame({
        'age': [25],
        'race': ['White alone, Non-Hispanic'],
        'fertility_rate': [0.10]
    })

    # Execute
    births = calculate_births(female_pop, fertility_rates, 2025)

    # Validate
    expected_total = 1000 * 0.10  # 100 births
    assert abs(births['population'].sum() - expected_total) < 1.0

    # Check sex ratio (51% male, 49% female)
    male_births = births[births['sex'] == 'Male']['population'].sum()
    assert abs(male_births / expected_total - 0.51) < 0.01

def test_calculate_births_zero_fertility():
    """Test with zero fertility rates."""
    female_pop = pd.DataFrame({...})
    fertility_rates = pd.DataFrame({'fertility_rate': [0.0]})

    births = calculate_births(female_pop, fertility_rates, 2025)

    assert births['population'].sum() == 0.0

def test_calculate_births_empty_population():
    """Test with empty population."""
    empty_pop = pd.DataFrame(columns=['age', 'sex', 'race', 'population'])
    fertility_rates = pd.DataFrame({...})

    births = calculate_births(empty_pop, fertility_rates, 2025)

    assert len(births) == 0
```

**Running Tests**:
```bash
pytest tests/test_core/
```

**Rationale**:
- **Critical Functions**: Mathematical correctness essential
- **Edge Cases**: Rare scenarios hard to catch otherwise
- **Regression**: Prevent breaking working code
- **Confidence**: Tests increase trust

**Realistic Expectation**: ~50-60% unit test coverage (focused on critical paths).

### Decision 5: Manual Validation Against Known Benchmarks

**Decision**: Ultimate validation is comparison to published Census projections and demographic benchmarks.

**Validation Process**:

**1. Compare to Census Projections**:
```python
# Load Census projection for ND
census_projection = pd.read_csv('census_nd_projection_2020_2040.csv')

# Run our projection
our_projection = run_projection(2020, 2040)

# Compare totals
census_2030 = census_projection[census_projection['year'] == 2030]['population'].sum()
our_2030 = our_projection[our_projection['year'] == 2030]['population'].sum()

pct_diff = abs(census_2030 - our_2030) / census_2030 * 100

print(f"Difference from Census: {pct_diff:.1f}%")

# Should be within 5-10% (different assumptions okay)
assert pct_diff < 10.0
```

**2. Demographic Plausibility Checks**:
```python
def validate_demographic_plausibility(projection):
    """Check if projection results are demographically plausible."""
    issues = []

    # Total Fertility Rate (TFR)
    births = projection[projection['age'] == 0]
    females = projection[projection['sex'] == 'Female']
    implied_tfr = births.sum() / females.sum() * 35  # Approximate

    if not 1.0 < implied_tfr < 3.5:
        issues.append(f"Implausible TFR: {implied_tfr:.2f}")

    # Life Expectancy (rough estimate)
    survival_rate_avg = projection.mean_survival_rate()
    implied_e0 = -np.log(1 - survival_rate_avg)  # Approximation

    if not 70 < implied_e0 < 90:
        issues.append(f"Implausible life expectancy: {implied_e0:.1f}")

    # Growth rate
    initial = projection[projection['year'] == 2025]['population'].sum()
    final = projection[projection['year'] == 2045]['population'].sum()
    annual_growth = ((final / initial) ** (1/20) - 1) * 100

    if not -2.0 < annual_growth < 3.0:
        issues.append(f"Implausible growth rate: {annual_growth:.2f}%")

    return len(issues) == 0, issues
```

**3. Visual Inspection**:
- Age pyramids (should have smooth shapes)
- Population trends (should be continuous)
- Sex ratios (should be ~0.95-1.05)
- Race distributions (should match known trends)

**Rationale**:
- **Real-World Validation**: Ultimately, results must match reality
- **Complex System**: Unit tests can't capture all interactions
- **Demographic Expertise**: Require domain knowledge to assess
- **Stakeholder Trust**: Comparison to official projections builds confidence

### Decision 6: pytest as Unit Test Framework

**Decision**: Use pytest for unit testing when tests are written.

**Reasons for pytest**:
- **Simple**: Minimal boilerplate
- **Powerful**: Fixtures, parametrization, plugins
- **Standard**: Most popular Python test framework
- **Integration**: Works with CI/CD
- **Discovery**: Automatic test discovery

**Alternative Considered**: unittest (standard library) - more verbose, less convenient.

**Test Organization**:
```
tests/
  test_core/
    test_fertility.py
    test_mortality.py
    test_migration.py
    test_cohort_component.py
  test_data/
    test_census_api.py
    test_base_population.py
    test_fertility_rates.py
  test_utils/
    test_config_loader.py
    test_demographic_utils.py
```

### Decision 7: Synthetic Test Data Generation

**Decision**: Use synthetic (generated) test data rather than committing large real datasets to repository.

**Synthetic Data Generator**:
```python
def generate_test_population(num_ages=91, num_sexes=2, num_races=3):
    """Generate realistic synthetic population for testing."""
    data = []
    for age in range(num_ages):
        for sex in ['Male', 'Female']:
            for race in ['White NH', 'Black NH', 'Hispanic']:
                # Realistic age distribution
                if age < 18:
                    pop = 1000
                elif age < 65:
                    pop = 1500
                else:
                    pop = max(100, 1500 - (age - 65) * 50)

                # Vary by race
                if race == 'White NH':
                    pop *= 3.0

                data.append({
                    'year': 2025,
                    'age': age,
                    'sex': sex,
                    'race': race,
                    'population': pop
                })

    return pd.DataFrame(data)
```

**Rationale**:
- **No Large Files**: Don't commit MB of test data to git
- **Reproducible**: Generate on demand
- **Customizable**: Easy to modify for different test scenarios
- **Fast**: Generate instantly

**When to Use Real Data**:
- Integration testing with actual Census files (not committed to repo)
- Validation against real projections

## Consequences

### Positive

1. **Pragmatic**: Appropriate testing for research/analytical codebase
2. **Built-in Quality**: Validation functions catch most errors
3. **Documentation**: Example scripts demonstrate usage
4. **Maintainable**: Less test code to maintain
5. **Flexible**: Can add more tests as needed
6. **Real-World Validation**: Comparison to benchmarks is ultimate test
7. **Fast Development**: Don't spend excessive time on tests
8. **User-Friendly**: Examples help users understand system

### Negative

1. **Lower Coverage**: Not comprehensive automated test suite
2. **Manual Validation**: Requires domain expertise to assess
3. **Regression Risk**: Fewer automated checks for breaking changes
4. **Trust Issues**: Some stakeholders expect high test coverage
5. **Debugging**: Some bugs only found during usage, not tests

### Risks and Mitigations

**Risk**: Bugs slip through with limited testing
- **Mitigation**: Comprehensive built-in validation
- **Mitigation**: Example scripts as integration tests
- **Mitigation**: Manual validation against benchmarks
- **Mitigation**: Add unit tests for critical functions

**Risk**: Changes break existing functionality
- **Mitigation**: Run example scripts before commits
- **Mitigation**: Compare outputs to previous runs
- **Mitigation**: Gradual test suite expansion

**Risk**: Users don't trust results without test suite
- **Mitigation**: Document validation process
- **Mitigation**: Show comparison to Census benchmarks
- **Mitigation**: Emphasize built-in validation
- **Mitigation**: Can add more tests if requested

## Alternatives Considered

### Alternative 1: Comprehensive Unit Testing (80%+ Coverage)

**Description**: Write unit tests for all functions, aim for 80%+ coverage.

**Pros**:
- High confidence in code correctness
- Catch regressions automatically
- Professional standard

**Cons**:
- Significant time investment (weeks)
- Maintenance burden
- Specs evolved during development (tests would need constant updating)
- Diminishing returns for research code

**Why Rejected**:
- Resource constraints
- Pragmatic approach sufficient
- Can add tests incrementally

### Alternative 2: No Testing (Trust and Hope)

**Description**: No formal tests, just run and see if it works.

**Pros**:
- Zero testing overhead
- Fastest development

**Cons**:
- High error risk
- No regression detection
- Low stakeholder confidence
- Unprofessional

**Why Rejected**:
- Unacceptable risk
- Built-in validation minimum requirement

### Alternative 3: Property-Based Testing (Hypothesis)

**Description**: Use Hypothesis library for property-based testing.

**Pros**:
- Finds edge cases automatically
- Less test code

**Cons**:
- Learning curve
- Harder to debug
- Overkill for this project

**Why Rejected**:
- Traditional testing sufficient
- Added complexity not justified

### Alternative 4: Snapshot Testing

**Description**: Record outputs, compare future runs to snapshots.

**Pros**:
- Easy to create
- Detects any changes

**Cons**:
- Can't distinguish good/bad changes
- Brittle (small changes break tests)
- Large snapshot files

**Why Rejected**:
- Better to validate semantics, not exact outputs
- Manual benchmarks more meaningful

## Implementation Notes

### Current Test Coverage

**Implemented**:
- ✅ Built-in validation functions in all modules
- ✅ Example script (`examples/run_basic_projection.py`)
- ✅ Defensive programming throughout
- ⚠️ Some unit tests (`tests/test_data/test_census_api.py`)

**To Be Implemented** (future):
- ❌ Comprehensive unit test suite (can add incrementally)
- ❌ Continuous integration (CI) pipeline
- ❌ Automated benchmark comparisons

### Running Tests

**Run Example Script**:
```bash
python examples/run_basic_projection.py
```

**Run Unit Tests** (when written):
```bash
# All tests
pytest

# Specific module
pytest tests/test_core/test_fertility.py

# With coverage
pytest --cov=cohort_projections tests/
```

**Manual Validation**:
```bash
# Generate projection
python scripts/projections/run_state_projection.py

# Compare to Census benchmark
python scripts/validation/compare_to_census.py
```

## References

1. **Testing Best Practices**: "Effective Software Testing" - Maurício Aniche (2022)
2. **pytest Documentation**: https://docs.pytest.org/
3. **Demographic Validation**: Smith, Tayman & Swanson, Chapter 9: "Evaluation of Projections"
4. **Python Testing**: "Python Testing with pytest" - Brian Okken (2022)

## Revision History

- **2025-12-18**: Initial version (ADR-011) - Testing strategy

## Related ADRs

- ADR-006: Data pipeline (validation at each stage)
- ADR-009: Logging and error handling (validation logging)
- All ADRs: Built-in validation mentioned throughout
