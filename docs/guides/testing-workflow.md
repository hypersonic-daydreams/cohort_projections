# Testing Workflow Guide

Detailed testing procedures for AI agents and developers working on this codebase.

**Related**: [AGENTS.md](../../AGENTS.md) (Section 5 - Quality Standards)

---

## Quick Reference

```bash
pytest tests/ -v                    # All tests, verbose
pytest tests/ -q                    # Quick/quiet mode
pytest tests/ -x                    # Stop on first failure
pytest tests/ --tb=long             # Detailed tracebacks
scripts/testing/run_relevant_tests.sh      # Relevant tests for current changes (fast)
```

---

## When Modifying Production Code

1. **Before changing code**: Run `pytest tests/ -v` to establish baseline
2. **After changing code**: Run tests again - failures indicate breaking changes
3. **If tests fail**: Either fix the code OR update the tests (if behavior change is intentional)
4. **Pre-commit enforces this**: Tests run automatically when committing changes to `cohort_projections/`

---

## When to Update Tests

| Change Type | Test Action |
|-------------|-------------|
| Bug fix | Add test that reproduces the bug, then fix |
| New function | Add tests for the new function |
| Changed signature | Update all tests that call the function |
| Changed behavior | Update tests to expect new behavior |
| Removed function | Remove tests for that function |

---

## Test Commands

### Fast Relevant Runner (Recommended for Edit Loops)

Use the helper script to run only test targets related to current changes.
It defaults to `--no-cov` for speed and falls back to the safe fast gate when
relevant mapping is ambiguous.

```bash
# Infer relevant tests from working tree changes
scripts/testing/run_relevant_tests.sh

# Infer from staged files only
scripts/testing/run_relevant_tests.sh --staged

# Run relevant tests, then run fast safety gate
scripts/testing/run_relevant_tests.sh --gate

# Provide files explicitly
scripts/testing/run_relevant_tests.sh cohort_projections/core/migration.py

# Show command/targets without running
scripts/testing/run_relevant_tests.sh --list-only
```

Other modes:

```bash
scripts/testing/run_relevant_tests.sh --fast   # pytest tests/ -x -q --ignore=tests/test_integration/ -m "not slow"
scripts/testing/run_relevant_tests.sh --full   # full suite (no-cov by default)
```

### Basic Usage

```bash
pytest tests/ -v                           # All tests
pytest tests/test_core/ -v                 # Just core module tests
pytest tests/ -k "test_fertility" -v       # Tests matching pattern
pytest tests/ -x                           # Stop on first failure
pytest tests/ --tb=long                    # Detailed tracebacks
```

### With Coverage

```bash
pytest --cov                               # With coverage report
pytest --cov=cohort_projections --cov-report=html  # HTML coverage report
```

### Markers

```bash
pytest -m slow                             # Only slow tests
pytest -m "not slow"                       # Exclude slow tests
pytest -m integration                      # Only integration tests
```

---

## Finding Related Tests

```bash
# Find tests for a specific function
grep -r "function_name" tests/

# Find tests for a module
ls tests/test_core/test_fertility.py      # Tests for core/fertility.py
```

---

## Test File Mapping

| Production Module | Test File |
|-------------------|-----------|
| `cohort_projections/core/cohort_component.py` | `tests/test_core/test_cohort_component.py` |
| `cohort_projections/core/fertility.py` | `tests/test_core/test_fertility.py` |
| `cohort_projections/core/mortality.py` | `tests/test_core/test_mortality.py` |
| `cohort_projections/core/migration.py` | `tests/test_core/test_migration.py` |
| `cohort_projections/data/process/base_population.py` | `tests/test_data/test_base_population.py` |
| `cohort_projections/output/writers.py` | `tests/test_output/test_writers.py` |

---

## Test Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_core/               # Core engine tests
│   ├── test_cohort_component.py
│   ├── test_fertility.py
│   ├── test_mortality.py
│   └── test_migration.py
├── test_data/               # Data processing tests
├── test_geographic/         # Geographic handling tests
├── test_integration/        # End-to-end tests
├── test_output/             # Output generation tests
├── test_statistical/        # Statistical analysis tests (B1, B2, B4)
└── test_tools/              # Tooling tests
```

---

## Writing Good Tests

### Fixture Usage

```python
import pytest

@pytest.fixture
def sample_population():
    """Create sample population DataFrame for testing."""
    return pd.DataFrame({
        "age": range(0, 90),
        "sex": ["M"] * 45 + ["F"] * 45,
        "population": [1000] * 90,
    })

def test_cohort_aging(sample_population):
    """Test that cohort ages correctly."""
    result = age_cohort(sample_population)
    assert result["age"].min() == 1
```

### Parameterized Tests

```python
@pytest.mark.parametrize("age,expected", [
    (0, "infant"),
    (17, "child"),
    (18, "adult"),
    (65, "elderly"),
])
def test_age_category(age, expected):
    assert categorize_age(age) == expected
```

### Testing Edge Cases

```python
def test_negative_population_raises():
    """Negative population should raise ValueError."""
    with pytest.raises(ValueError, match="negative"):
        validate_population(-100)
```

---

## Pre-commit Integration

The project's pre-commit configuration runs fast tests automatically on commits that modify `cohort_projections/`:

```yaml
# From .pre-commit-config.yaml
- id: pytest-check
  entry: pytest tests/ -x -q --ignore=tests/test_integration/ -m "not slow"
  files: ^cohort_projections/.*\.py$
```

To run the same tests manually:

```bash
pytest tests/ -x -q --ignore=tests/test_integration/ -m "not slow"
```

---

*Last Updated: 2026-01-01*
