# Agent B6: Test Suite and Validation Infrastructure

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B6 |
| Scope | Comprehensive test suite for Phase B implementation |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 Existing Test Infrastructure

Test directory structure:
```
tests/
|-- test_core/           # Demographic model tests
|-- test_data/           # Data loading/API tests
|-- test_geographic/     # Geographic data tests
|-- test_integration/    # End-to-end tests
|-- test_output/         # Output generation tests
|-- test_tools/          # Tooling tests
```

### 1.2 Testing Configuration

From `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",
    "--cov=cohort_projections",
    "--cov-report=term-missing",
]
markers = [
    "slow: marks tests as slow",
    "data_fetch: marks tests that fetch external data",
    "integration: marks integration tests",
]
```

### 1.3 Test Patterns Observed

| Pattern | Example |
|---------|---------|
| Class-based organization | `test_migration.py` |
| Fixtures | `@pytest.fixture` |
| Parameterized tests | `@pytest.mark.parametrize` |
| Skip conditions | `@pytest.mark.skipif` |
| Mocking | `unittest.mock.patch` |

### 1.4 Current Gaps

- No tests for statistical analysis modules (B1, B2, B4)
- No LaTeX validation tests (B3)
- No ADR cross-reference validation (B5)
- No shared `conftest.py`

---

## 2. Test Categories

| Category | Purpose | Scope |
|----------|---------|-------|
| **Unit** | Function correctness | B1, B2, B4 functions |
| **Integration** | Pipeline verification | Data pipelines |
| **Validation** | Output verification | B1 JSON, B3 article |
| **Smoke** | Basic functionality | B1-B4 main scripts |
| **Documentation** | Link validation | B5 ADR references |

---

## 3. Tests per Agent

### 3.1 B1: Regime-Aware Statistical Modeling

#### Unit Tests

| Function | Test Cases |
|----------|------------|
| `create_vintage_dummies()` | 2000-2024 data; single year; edge case (2010) |
| `estimate_piecewise_trend()` | Linear data; data with known break |
| `create_covid_intervention()` | 2020 included; 2020 excluded |
| `estimate_regime_variances()` | Three-regime; homogeneous |
| `estimate_wls_by_regime()` | High variance ratio |

#### Edge Cases

- Negative migration value (2003)
- Single observation per regime
- Missing regime

#### Parameterized Tests

```python
@pytest.mark.parametrize("se_type", ["HC0", "HC1", "HC2", "HC3", "HAC"])
def test_robust_se_types(se_type): ...
```

### 3.2 B2: Multi-State Placebo Analysis

#### Unit Tests

| Function | Test Cases |
|----------|------------|
| `load_vintage_2009_data()` | Real CSV, 50 states x 10 years |
| `combine_all_vintages()` | 50 states x 25 years with labels |
| `calculate_state_shift()` | ND; state with negative shift |
| `test_oil_state_hypothesis()` | Full 50-state data |

#### Integration Tests

- Full placebo analysis pipeline
- Data coverage verification
- No missing data check

### 3.3 B3: Journal Article Methodology

#### LaTeX Compilation Tests

| Test | Verification |
|------|--------------|
| `test_article_compiles()` | pdflatex exits 0 |
| `test_no_undefined_references()` | No "undefined reference" in log |

#### Numeric Claim Consistency

| Source | Target |
|--------|--------|
| B1 JSON | `03_results.tex` table |
| B2 percentile | `02_data_methods.tex` |

### 3.4 B4: Bayesian/Panel Extensions

#### Unit Tests

| Function | Test Cases |
|----------|------------|
| `construct_minnesota_prior()` | 2 vars x 1 lag; random walk structure |
| `estimate_bayesian_var()` | Synthetic data |
| `estimate_panel_var()` | 50-state panel |
| `compare_models()` | Classical vs Bayesian |

#### Smoke Tests (conditional on PyMC)

- `test_pymc_available()`
- `test_bayesian_var_runs()` (60s timeout)

### 3.5 B5: ADR Documentation

#### Link Validation

| Test | Verification |
|------|--------------|
| `test_adr_internal_links()` | All markdown links resolve |
| `test_artifact_links()` | Links to `020-reports/` valid |
| `test_readme_includes_adr()` | ADR-020 in README |

---

## 4. Files Inventory

### 4.1 New Test Files

| File | Priority |
|------|----------|
| `tests/conftest.py` | **HIGH** |
| `tests/test_statistical/__init__.py` | **HIGH** |
| `tests/test_statistical/test_regime_aware.py` | **HIGH** |
| `tests/test_statistical/test_multistate_placebo.py` | **HIGH** |
| `tests/test_statistical/test_sensitivity_suite.py` | **HIGH** |
| `tests/test_statistical/test_bayesian_var.py` | **MEDIUM** |
| `tests/test_article/test_latex_compilation.py` | **MEDIUM** |
| `tests/test_article/test_numeric_claims.py` | **HIGH** |
| `tests/test_docs/test_adr_links.py` | **MEDIUM** |

### 4.2 Test Fixtures Needed

| Fixture | Content |
|---------|---------|
| `sample_nd_migration_n25` | 25-row DataFrame with vintages |
| `sample_50_state_panel` | 50-state x 25-year DataFrame |
| `synthetic_regime_data` | Data with known structure |

### 4.3 Test Data Files

| File | Purpose |
|------|---------|
| `tests/fixtures/sample_nd_migration.csv` | B1 unit tests |
| `tests/fixtures/expected_regime_output.json` | Regression testing |

---

## 5. Code Structure

### 5.1 Test Module Organization

```
tests/
|-- conftest.py                    # Shared fixtures
|-- test_statistical/
|   |-- conftest.py                # Statistical fixtures
|   |-- test_regime_aware.py       # B1
|   |-- test_multistate_placebo.py # B2
|   |-- test_bayesian_var.py       # B4
|-- test_article/
|   |-- test_latex_compilation.py  # B3
|   |-- test_numeric_claims.py     # B3
|-- test_docs/
|   |-- test_adr_links.py          # B5
|-- fixtures/
    |-- sample_nd_migration.csv
    |-- expected_regime_output.json
```

### 5.2 Parameterized Test Examples

```python
@pytest.mark.parametrize("spec_name,expected_n", [
    ("n15_baseline", 15),
    ("n25_with_vintage", 25),
])
def test_sensitivity_spec_sample_sizes(spec_name, expected_n): ...

@pytest.mark.parametrize("state,expected_oil", [
    ("North Dakota", True),
    ("California", False),
])
def test_oil_state_classification(state, expected_oil): ...
```

---

## 6. Dependencies

### 6.1 Test Data Dependencies

| Test Module | Data Source | Status |
|-------------|-------------|--------|
| B1 tests | `agent2_nd_migration_data.csv` | Available |
| B2 tests | `NST-EST2009-ALLDATA.csv` | Available |
| B3 tests | B1, B2 JSON outputs | Generated |
| B4 tests | B2 panel | Generated |

### 6.2 Test Execution Order

1. B1 Unit Tests (no deps)
2. B2 Unit Tests (no deps)
3. B1 Integration Tests (generates outputs)
4. B2 Integration Tests (generates panel)
5. B3 Validation Tests (requires B1, B2)
6. B4 Tests (requires B2 panel)
7. B5 Tests (no deps)

### 6.3 Pytest Markers

```python
markers = [
    "requires_b1: requires B1 outputs",
    "requires_b2: requires B2 outputs",
    "requires_pymc: requires PyMC package",
]
```

---

## 7. Estimated Complexity

| Component | Complexity | Justification |
|-----------|------------|---------------|
| B1 Unit Tests | **LOW** | Standard function testing |
| B1 Integration | **MEDIUM** | Full pipeline |
| B2 Unit Tests | **LOW** | Standard function testing |
| B2 Integration | **MEDIUM** | Three data sources |
| B3 LaTeX Tests | **MEDIUM** | Requires subprocess |
| B3 Numeric Claims | **HIGH** | Parse LaTeX + JSON matching |
| B4 Unit Tests | **MEDIUM** | Bayesian validation |
| B5 Link Tests | **LOW** | File existence |
| **Overall** | **MEDIUM** |

---

## 8. Implementation Considerations

### 8.1 Test Data Strategy

**Hybrid approach:**
- Unit tests: Synthetic data with known properties
- Integration tests: Real data subsets
- Regression tests: Frozen expected outputs

### 8.2 PyMC Dependency Handling

```python
pytestmark = pytest.mark.skipif(
    not PYMC_AVAILABLE,
    reason="PyMC not installed"
)
```

### 8.3 LaTeX Testing

```python
import subprocess

def test_article_compiles():
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        cwd=ARTICLE_DIR,
        capture_output=True,
    )
    assert result.returncode == 0
```

---

## Summary

This plan provides:

1. **Current state assessment** of test infrastructure
2. **Five test categories** (unit, integration, validation, smoke, documentation)
3. **Test specifications** for B1-B5 agents
4. **16+ new test files** to create
5. **Fixture and utility design**
6. **Dependency mapping** for execution order

**Key Finding:** Existing test patterns are solid. Main gap is statistical analysis module coverage.

**Decision Required:** Approve to proceed with B6 implementation.
