# Statistical Concordance

A comprehensive reference mapping all statistical tests and equations from the journal article "Forecasting International Migration to North Dakota: A Multi-Method Analysis" to their LaTeX formulations, Python implementations, and explanatory documentation.

## Purpose

This concordance serves as a **machine-readable reference** for AI agents and human reviewers to:

1. **Understand statistical tests** - What each test does, its null/alternative hypotheses, and decision rules
2. **Decipher equations** - Symbol-by-symbol breakdown of all 17 numbered equations
3. **Find implementations** - Map equations and tests to their Python source files
4. **Cross-reference** - Link between LaTeX source, Python code, and paper sections

## File Structure

```
concordance/
├── README.md                    # This file
├── statistical_concordance.yaml # Main concordance data (YAML format)
├── concordance_parser.py        # Python parser for programmatic access
└── explanations/                # Detailed explanatory text (PENDING)
    ├── equations/               # One file per equation
    └── tests/                   # One file per statistical test
```

## Quick Reference

### Statistical Tests (18 total)

| Category | Tests |
|----------|-------|
| Unit Root | ADF, Phillips-Perron, KPSS, Zivot-Andrews |
| Residual Diagnostics | Ljung-Box, Shapiro-Wilk |
| Structural Breaks | Chow, CUSUM, Bai-Perron |
| Panel Data | Hausman, Breusch-Pagan LM |
| Causal Inference | Parallel Trends, Wild Cluster Bootstrap, Randomization Inference, Granger Causality |
| Survival Analysis | Log-Rank, Schoenfeld Residuals |

### Equations (17 total)

| # | Name | Category | Python Module |
|---|------|----------|---------------|
| 1 | Hodrick-Prescott Filter | Trend Decomposition | module_1_1 |
| 2 | Herfindahl-Hirschman Index | Concentration | module_1_1 |
| 3 | Location Quotient | Concentration | module_1_1 |
| 4 | ADF Regression | Unit Root | module_2_1_1 |
| 5 | AIC | Model Selection | module_2_1 |
| 6 | Bai-Perron | Structural Break | - |
| 7 | VAR | Multivariate TS | module_2_2 |
| 8 | Two-Way Fixed Effects | Panel | module_3_1 |
| 9 | PPML Gravity | Allocation | module_5 |
| 10 | Elastic Net | Regularization | module_6 |
| 11 | K-Means | Clustering | module_6 |
| 12 | Difference-in-Differences | Causal | module_7 |
| 13 | Interrupted Time Series | Causal | module_7 |
| 14 | Synthetic Control | Causal | module_7 |
| 15 | Shift-Share (Bartik) | IV | module_7 |
| 16 | Kaplan-Meier | Survival | module_8 |
| 17 | Cox Proportional Hazards | Survival | module_8 |

## Usage

### For AI Agents (Programmatic Access)

```python
from concordance_parser import Concordance

concordance = Concordance()

# Get equation by key
eq = concordance.get_equation('eq_did')
print(eq['latex'])
print(eq['python_implementation'])
print(eq['symbols'])

# Get equation by number
eq_12 = concordance.get_equation_by_number(12)

# Get statistical test
test = concordance.get_test('adf_test')
print(test['null_hypothesis'])
print(test['decision_rule'])
print(test['python_library'])

# Find implementing module
module = concordance.find_module_for_equation('eq_km')

# Search
results = concordance.search_equations('causal')
```

### For Human Reviewers (YAML Reference)

Open `statistical_concordance.yaml` and navigate to:

- `statistical_tests:` - All test definitions
- `equations:` - All equation definitions
- `python_scripts:` - Module index

Each equation entry contains:

```yaml
eq_did:
  number: 12
  name: "Difference-in-Differences"
  category: "causal_inference"
  paper_section: "2.7.1 Difference-in-Differences"
  latex: |
    \ln(y_{ct} + 1) = \alpha_c + \lambda_t + \delta \cdot ...
  symbols:
    y_ct: "Arrivals from country c in year t"
    delta: "Average treatment effect on the treated (ATT)"
    ...
  python_module: "module_7_causal_inference.py"
  python_implementation: |
    # Full working code example
    ...
```

Each test entry contains:

```yaml
adf_test:
  full_name: "Augmented Dickey-Fuller Test"
  category: "unit_root"
  null_hypothesis: "Unit root present (series is non-stationary)"
  alternative_hypothesis: "Series is stationary"
  decision_rule: "Reject H0 if p-value < 0.05"
  python_library: "statsmodels.tsa.stattools.adfuller"
  python_module: "module_2_1_1_unit_root_tests.py"
```

## Status

### Completed

- [x] Concordance structure (YAML)
- [x] All 17 equations extracted with LaTeX
- [x] All 18 statistical tests catalogued
- [x] Python implementation snippets
- [x] Symbol definitions
- [x] Python parser module
- [x] Script-to-method mapping

### Pending

- [ ] Detailed explanatory text for each equation
- [ ] Detailed explanatory text for each test
- [ ] Mathematical derivations
- [ ] Intuitive explanations for non-technical readers
- [ ] Visual diagrams/flowcharts

## Contributing

To add explanatory text for an equation:

1. Create a new file in `explanations/equations/eq_<name>.md`
2. Include:
   - Plain-English explanation of what the equation does
   - Step-by-step interpretation of each term
   - Worked example with real numbers
   - Common pitfalls/assumptions
   - References to related tests
3. Update the `# PLACEHOLDER: explanatory_text` in the YAML

## Version

- **Structure Version**: 0.1
- **Generated**: 2026-01-12
- **Source Article**: article-0.9-production

---

*This concordance is designed for AI agent consumption. All formulas are provided in both LaTeX (for rendering) and Python (for computation).*
