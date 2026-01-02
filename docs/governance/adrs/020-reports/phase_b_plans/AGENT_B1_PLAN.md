# Agent B1: Statistical Modeling Infrastructure

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B1 |
| Scope | Regime-aware statistical model implementation |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 Existing Statistical Analysis Code

The project has a well-established statistical analysis infrastructure located at:
`sdc_2024_replication/scripts/statistical_analysis/`

**Key existing modules:**

| Module | Purpose | Relevance to B1 |
|--------|---------|-----------------|
| `module_template.py` | Standard module structure with `ModuleResult` class | **HIGH** - Pattern to follow |
| `module_2_1_arima.py` | ARIMA time series modeling | **HIGH** - Existing time series infrastructure |
| `module_2_1_1_unit_root_tests.py` | Unit root tests (ADF, KPSS, etc.) | **MEDIUM** - Integration order dependency |
| `module_2_1_2_structural_breaks.py` | Structural break detection (Bai-Perron, CUSUM, Chow) | **HIGH** - Break detection already implemented |
| `module_4_regression_extensions.py` | Quantile regression, robust regression, GLM | **HIGH** - Robust inference patterns |
| `module_7_robustness.py` | Wild cluster bootstrap, DiD with clustered SE | **HIGH** - Heteroskedasticity-robust inference |

### 1.2 Available Packages

From `pyproject.toml` and `requirements_statistical.txt`:

| Package | Version | Purpose for B1 |
|---------|---------|----------------|
| `statsmodels` | >=0.14 | OLS, robust SE, HAC standard errors, GLM |
| `scipy` | >=1.10 | Statistical tests (t-test, F-test, Levene, etc.) |
| `numpy` | >=1.24 | Numerical operations |
| `pandas` | >=2.0 | Data manipulation |
| `pmdarima` | >=2.0 | Auto-ARIMA (already used) |
| `ruptures` | >=1.1 | Change point detection (already used) |
| `arch` | >=6.0 | GARCH, structural breaks (available) |
| `linearmodels` | >=5.0 | Panel fixed effects, IV (available) |
| `matplotlib` | >=3.7 | Visualization |
| `seaborn` | >=0.12 | Statistical visualization |

**Key finding**: All required packages for regime-aware modeling are already available.

### 1.3 Current Data Files

| File | Location | Content | Use |
|------|----------|---------|-----|
| `nd_migration_summary.csv` | `data/processed/immigration/analysis/` | n=15 observations (2010-2024) | Primary analysis series |
| `agent2_nd_migration_data.csv` | `docs/adr/020-reports/chatgpt_review_package/` | n=25 observations (2000-2024) with vintage labels | **Extended series for B1** |

**Critical observation**: The extended series (n=25) is available in the Phase A artifacts with structure:
- `year`, `intl_migration`, `vintage`, `vintage_period`
- Three vintages: 2009 (2000-2009), 2020 (2010-2019), 2024 (2020-2024)
- 2003 has a negative value (-545) - critical for log transformations

---

## 2. Files Inventory

### 2.1 Files to Modify

| File | Reason for Modification | Type of Change |
|------|------------------------|----------------|
| `sdc_2024_replication/scripts/statistical_analysis/SUBAGENT_COORDINATION.md` | Add B1 module documentation | Documentation update |

### 2.2 New Files to Create

| File | Purpose |
|------|---------|
| **Core Module** | |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/` | New module directory |
| `module_regime_aware/__init__.py` | Module initialization |
| `module_regime_aware/regime_definitions.py` | Regime/vintage boundary definitions |
| `module_regime_aware/vintage_dummies.py` | Vintage dummy variable creation |
| `module_regime_aware/piecewise_trends.py` | Piecewise linear trend estimation |
| `module_regime_aware/covid_intervention.py` | 2020 intervention/outlier modeling |
| `module_regime_aware/robust_inference.py` | Heteroskedasticity-robust and regime-specific variance |
| `module_regime_aware/sensitivity_suite.py` | Coordinated sensitivity analysis runner |
| **Main Analysis Scripts** | |
| `sdc_2024_replication/scripts/statistical_analysis/module_B1_regime_aware_models.py` | Primary analysis script |
| **Output Files** | |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_regime_aware_models.json` | JSON results |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_sensitivity_summary.csv` | CSV sensitivity results |
| `sdc_2024_replication/scripts/statistical_analysis/figures/module_B1_*.png/pdf` | Figures (multiple) |

### 2.3 Data Files to Use

| File | Source | Processing Required |
|------|--------|---------------------|
| `agent2_nd_migration_data.csv` | Phase A artifacts | Copy to analysis directory, validate |
| `nd_migration_summary.csv` | Standard data directory | For n=15 baseline comparison |

---

## 3. Implementation Plan

### 3.1 Vintage Dummies (2000s vs 2010s vs 2020s)

**Objective**: Create indicator variables that capture regime differences at vintage boundaries.

**Statistical Method**:
- Create dummy variables: `D_2010s` (1 for 2010-2019, 0 otherwise), `D_2020s` (1 for 2020-2024, 0 otherwise)
- Reference category: 2000s (Vintage 2009)
- Model: `y_t = alpha + beta_1 * D_2010s + beta_2 * D_2020s + gamma * t + epsilon_t`

**Python Implementation**:
```python
# In vintage_dummies.py
def create_vintage_dummies(df: pd.DataFrame, year_col: str = 'year') -> pd.DataFrame:
    """Create vintage/regime dummy variables."""
    df = df.copy()
    df['vintage_2010s'] = ((df[year_col] >= 2010) & (df[year_col] <= 2019)).astype(int)
    df['vintage_2020s'] = (df[year_col] >= 2020).astype(int)
    return df
```

**Output**:
- Coefficients for `D_2010s` and `D_2020s` with standard errors
- F-test for joint significance of vintage dummies
- Interpretation: Level shifts attributable to vintage transitions

### 3.2 Piecewise Trends (Allow Slopes to Differ by Regime)

**Objective**: Allow trend slope to vary across measurement regimes.

**Statistical Method**:
- Piecewise linear model with regime-specific slopes:
  ```
  y_t = alpha + beta_1 * t * D_2000s + beta_2 * t * D_2010s + beta_3 * t * D_2020s + epsilon_t
  ```
- Alternative: Spline-based approach with knots at 2010 and 2020

**Python Implementation**:
```python
# In piecewise_trends.py
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def estimate_piecewise_trend(df: pd.DataFrame, y_col: str, year_col: str = 'year') -> dict:
    """Estimate piecewise linear trend with regime-specific slopes."""
    df = df.copy()

    # Create centered time variable
    df['t'] = df[year_col] - df[year_col].min()

    # Regime indicators
    df['regime_2000s'] = (df[year_col] < 2010).astype(int)
    df['regime_2010s'] = ((df[year_col] >= 2010) & (df[year_col] < 2020)).astype(int)
    df['regime_2020s'] = (df[year_col] >= 2020).astype(int)

    # Regime-specific trend interactions
    df['trend_2000s'] = df['t'] * df['regime_2000s']
    df['trend_2010s'] = (df['t'] - 10) * df['regime_2010s']  # Reset at 2010
    df['trend_2020s'] = (df['t'] - 20) * df['regime_2020s']  # Reset at 2020

    # Design matrix
    X = df[['regime_2010s', 'regime_2020s', 'trend_2000s', 'trend_2010s', 'trend_2020s']]
    X = sm.add_constant(X)
    y = df[y_col]

    # Estimate with HAC standard errors
    model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 2})

    return {
        'model': model,
        'slopes': {
            '2000s': model.params['trend_2000s'],
            '2010s': model.params['trend_2010s'],
            '2020s': model.params['trend_2020s'],
        },
        'slope_equality_test': model.f_test('trend_2000s = trend_2010s = trend_2020s')
    }
```

**Output**:
- Regime-specific slope coefficients
- F-test for slope equality across regimes
- Visual: Segmented trend plot

### 3.3 Intervention/Outlier Term for 2020 (COVID Shock)

**Objective**: Model 2020 as an exceptional observation due to COVID-19.

**Statistical Method**:
Two approaches:
1. **Intervention dummy**: Add `D_2020 = 1` for year 2020 only
2. **Pulse intervention in ARIMA**: Use `exog` parameter in ARIMA with COVID indicator

**Python Implementation**:
```python
# In covid_intervention.py
def create_covid_intervention(df: pd.DataFrame, year_col: str = 'year') -> pd.DataFrame:
    """Create COVID-19 intervention indicators."""
    df = df.copy()

    # Pulse intervention (2020 only)
    df['covid_pulse'] = (df[year_col] == 2020).astype(int)

    # Step intervention (2020 onward) - for modeling sustained effects
    df['covid_step'] = (df[year_col] >= 2020).astype(int)

    # Recovery ramp (increasing from 2020)
    df['covid_recovery'] = np.maximum(0, df[year_col] - 2020)

    return df
```

**Output**:
- COVID intervention effect magnitude
- Comparison of models with/without intervention term
- AIC/BIC comparison

### 3.4 Heteroskedasticity-Robust Inference (Regime-Specific Error Variance)

**Objective**: Account for the 29:1 variance ratio across vintages identified in Phase A.

**Statistical Method**:
1. **Heteroskedasticity-consistent standard errors**: HC0, HC1, HC2, HC3
2. **HAC standard errors**: Newey-West for autocorrelation
3. **Regime-specific variance models**: Estimate separate sigma^2 per regime
4. **Weighted Least Squares (WLS)**: Weight inversely by regime variance

**Python Implementation**:
```python
# In robust_inference.py
from statsmodels.regression.linear_model import OLS, WLS
from scipy import stats

def estimate_regime_variances(df: pd.DataFrame, y_col: str,
                              regime_col: str = 'vintage') -> dict:
    """Estimate regime-specific error variances."""
    regimes = df[regime_col].unique()
    variances = {}

    for regime in regimes:
        subset = df[df[regime_col] == regime]
        y = subset[y_col].values
        t = np.arange(len(y))
        X = sm.add_constant(t)
        residuals = OLS(y, X).fit().resid
        variances[regime] = np.var(residuals, ddof=1)

    # Levene test
    groups = [df[df[regime_col] == r][y_col].values for r in regimes]
    levene_stat, levene_p = stats.levene(*groups)

    return {
        'variances': variances,
        'variance_ratio': max(variances.values()) / min(variances.values()),
        'levene_test': {'statistic': levene_stat, 'p_value': levene_p}
    }

def estimate_wls_by_regime(df: pd.DataFrame, y_col: str, X_cols: list,
                           regime_col: str = 'vintage') -> dict:
    """Estimate WLS with weights inverse to regime variance."""
    var_info = estimate_regime_variances(df, y_col, regime_col)

    df = df.copy()
    df['weight'] = df[regime_col].map(
        lambda r: 1.0 / var_info['variances'].get(r, 1.0)
    )

    X = sm.add_constant(df[X_cols])
    y = df[y_col]

    model = WLS(y, X, weights=df['weight']).fit()

    return {
        'model': model,
        'params': model.params.to_dict(),
        'bse': model.bse.to_dict(),
        'regime_weights': {r: 1.0/v for r, v in var_info['variances'].items()}
    }
```

**Output**:
- Comparison table: OLS SE vs HC3 SE vs HAC SE
- WLS estimates with regime weighting
- Regime-specific variance estimates

### 3.5 Sensitivity Analysis Framework

**Objective**: Run coordinated sensitivity analyses as recommended by external review.

**Specifications to run**:
1. **n=15 baseline**: 2010-2024 only (current standard)
2. **n=25 with vintage controls**: Full series with vintage dummies
3. **Excluding 2020**: Remove COVID shock year
4. **Excluding 2000-2009**: Post-methodology-change only

**Python Implementation**:
```python
# In sensitivity_suite.py
def run_sensitivity_suite(df_full: pd.DataFrame, y_col: str = 'intl_migration') -> dict:
    """Run complete sensitivity analysis suite."""

    specifications = {
        'n15_baseline': {
            'filter': lambda d: d[d['year'] >= 2010],
            'description': 'Primary window (2010-2024, n=15)',
            'vintage_controls': False
        },
        'n25_with_vintage': {
            'filter': lambda d: d,
            'description': 'Extended with vintage dummies (2000-2024, n=25)',
            'vintage_controls': True
        },
        'n24_excl_2020': {
            'filter': lambda d: d[d['year'] != 2020],
            'description': 'Excluding COVID year (n=24)',
            'vintage_controls': True
        },
        'n15_post_2010': {
            'filter': lambda d: d[d['year'] >= 2010],
            'description': 'Post-methodology-change only (n=15)',
            'vintage_controls': False
        },
    }

    results = {}
    for spec_name, spec in specifications.items():
        df_spec = spec['filter'](df_full.copy())

        if spec['vintage_controls']:
            df_spec = create_vintage_dummies(df_spec)

        results[spec_name] = {
            'description': spec['description'],
            'n': len(df_spec),
            'trend_estimate': estimate_trend(df_spec, y_col),
            'mean': df_spec[y_col].mean(),
            'std': df_spec[y_col].std(),
        }

    return results
```

**Output**:
- Summary table comparing key statistics across specifications
- Visualization of coefficient stability
- Pass/fail assessment: "Does substantive conclusion change?"

---

## 4. Code Structure

### 4.1 Module Architecture

```
sdc_2024_replication/scripts/statistical_analysis/
├── module_regime_aware/                    # New module directory
│   ├── __init__.py                         # Exports main functions
│   ├── regime_definitions.py               # Vintage boundaries, regime labels
│   ├── vintage_dummies.py                  # Dummy variable creation
│   ├── piecewise_trends.py                 # Segmented trend estimation
│   ├── covid_intervention.py               # 2020 intervention modeling
│   ├── robust_inference.py                 # HC/HAC SE, WLS
│   └── sensitivity_suite.py                # Coordinated sensitivity runner
├── module_B1_regime_aware_models.py        # Main analysis script
├── results/
│   ├── module_B1_regime_aware_models.json  # Primary results
│   └── module_B1_sensitivity_summary.csv   # Sensitivity comparison
└── figures/
    ├── module_B1_piecewise_trends.png
    ├── module_B1_vintage_effects.png
    ├── module_B1_covid_intervention.png
    ├── module_B1_robust_se_comparison.png
    └── module_B1_sensitivity_summary.png
```

### 4.2 Output Formats

| Output | Format | Purpose |
|--------|--------|---------|
| `module_B1_regime_aware_models.json` | JSON | Machine-readable complete results |
| `module_B1_sensitivity_summary.csv` | CSV | Human-readable comparison table |
| `module_B1_*.png/pdf` | Images | Publication-ready figures |
| `module_B1_robustness_table.tex` | LaTeX | For journal article tables |

---

## 5. Dependencies

### 5.1 Agents This Work Depends On

| Agent | Dependency | Type |
|-------|------------|------|
| **Phase A (Complete)** | Extended n=25 data file with vintage labels | Data |
| **B0a (Complete)** | Versioning strategy for output files | Process |
| **B0b (Complete)** | Sprint workflow structure | Process |

### 5.2 Agents That Depend on This Work

| Agent | What They Need |
|-------|---------------|
| **B2 (Multi-State)** | Regime-aware model specification for panel expansion |
| **B3 (Journal Article)** | Robustness table and sensitivity results |
| **B4 (Bayesian/Panel)** | Regime structure for Bayesian priors |
| **B6 (Testing)** | Test specifications for regime-aware functions |

---

## 6. Risks and Blockers

### 6.1 Statistical Challenges

| Challenge | Risk Level | Mitigation |
|-----------|------------|------------|
| **Small n (n=25 total)** | HIGH | Use robust SE, bootstrap, avoid overfitting |
| **Collinearity between vintage dummies and time** | MEDIUM | Center variables, check VIF |
| **Non-stationarity** | MEDIUM | First-difference if needed (I(1) series) |
| **Negative 2003 value (-545)** | HIGH for log transforms | Use levels, or IHS transform |
| **COVID as single observation** | HIGH | Intervention term vs. exclusion comparison |

### 6.2 Data Availability

| Issue | Status | Resolution |
|-------|--------|------------|
| Extended series (n=25) | Available in Phase A artifacts | Copy to analysis directory |
| Vintage labels | Available | Already in `agent2_nd_migration_data.csv` |
| n=15 baseline | Available | `nd_migration_summary.csv` |

### 6.3 Package Compatibility

| Package | Required | Available | Action |
|---------|----------|-----------|--------|
| `statsmodels` | >=0.14 | Yes | None |
| `arch` | >=6.0 | Per requirements | Verify installation |
| `scipy` | >=1.10 | Yes | None |

---

## 7. Estimated Complexity

| Component | Complexity | Justification |
|-----------|------------|---------------|
| Vintage Dummies | **LOW** | Simple dummy creation and OLS |
| Piecewise Trends | **MEDIUM** | Requires careful parameterization |
| COVID Intervention | **LOW** | Simple intervention dummy |
| Robust Inference | **MEDIUM** | Multiple SE estimators, WLS setup |
| Sensitivity Suite | **MEDIUM** | Coordination across specifications |
| **Overall** | **MEDIUM** | Straightforward statistics, complexity is in coordination |

---

## 8. Handling the 2003 Negative Value

The 2003 observation has `intl_migration = -545`, which creates challenges for:
1. Log transformations (undefined for negative values)
2. Variance-stabilizing transforms
3. Per-capita rate calculations (still negative)

**Recommended approaches**:
1. **Primary analysis in levels**: Use raw counts, not log-transformed
2. **IHS transform** (Inverse Hyperbolic Sine) if transformation needed: `asinh(y) = log(y + sqrt(y^2 + 1))`
3. **Truncate or winsorize**: Note and document treatment
4. **Exclude 2003 as sensitivity check**: Additional specification in sensitivity suite

---

## Summary

This plan provides:

1. **Complete module specification** for regime-aware statistical models
2. **Four core modeling components**: vintage dummies, piecewise trends, COVID intervention, robust inference
3. **Sensitivity analysis framework** with 4+ specifications per external review recommendations
4. **Clear file inventory** of all files to create/modify
5. **Dependency mapping** to other B-series agents

**Decision Required**: Approve this plan to proceed with B1 implementation during execution phase.
