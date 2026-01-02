# Agent B4: Panel/Bayesian VAR Extensions

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B4 |
| Scope | Panel data approaches and Bayesian VAR methods for small-n limitations |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 Existing VAR/Time Series Code

| Module | Location | Relevance |
|--------|----------|-----------|
| `module_2_2_var_cointegration.py` | statistical_analysis/ | **HIGH** - Complete VAR with Johansen, Granger, IRF, FEVD |
| `module_2_1_arima.py` | statistical_analysis/ | **HIGH** - ARIMA forecasting pattern |
| `module_3_1_panel_data.py` | statistical_analysis/ | **CRITICAL** - Panel FE/RE already implemented |
| `module_template.py` | statistical_analysis/ | **HIGH** - ModuleResult pattern |

**Key Finding**: Existing VAR uses `statsmodels.tsa.api.VAR` with MLE estimation. No Bayesian capabilities.

**Documented Limitation** (from code):
> "Short time series (n=15) limits reliability of VAR/cointegration analysis."

### 1.2 Available Packages

| Package | Status | Purpose |
|---------|--------|---------|
| `statsmodels` | **Available** (>=0.14) | Classical VAR, state-space |
| `linearmodels` | **Available** (>=5.0) | Panel FE/RE, already in use |
| `arch` | **Available** (>=6.0) | GARCH, extensions |
| PyMC | **NOT INSTALLED** | Full Bayesian MCMC |
| arviz | **NOT INSTALLED** | Bayesian diagnostics |

**Critical Finding**: No Bayesian packages installed. Adding PyMC requires new dependency.

### 1.3 Existing Panel Data Infrastructure

From `module_3_1_panel_data.py`:

- **Implemented**: Two-way fixed effects, random effects, Hausman test, clustered SE
- **Panel Structure**: 51 entities × 15 periods = 765 observations
- **Package**: `linearmodels.panel.PanelOLS`

### 1.4 Data Files Available

| File | Content |
|------|---------|
| `nd_migration_summary.csv` | n=15 ND (2010-2024) |
| `combined_components_of_change.csv` | 50-state panel (2010-2024) |
| `agent2_nd_migration_data.csv` | n=25 ND with vintage labels |

---

## 2. Panel Data Approach Plan

### 2.1 Panel Fixed Effects for Multi-State Analysis

**Objective**: Leverage 50-state panel to increase effective sample size.

**Model Specification**:
```
intl_migration_{i,t} = alpha_i + gamma_t + beta_1 * X_{i,t} + beta_2 * D_ND * t + epsilon_{i,t}
```

Where:
- `alpha_i` = state fixed effect
- `gamma_t` = time fixed effect
- `D_ND * t` = ND-specific trend

**Specifications to Implement**:

1. **Baseline Panel FE**: All 50 states, entity + time effects
2. **Regional Panel**: Oil/energy states only (10 states)
3. **Synthetic Control-style**: ND vs weighted controls
4. **Random Coefficients**: ND-specific slopes

### 2.2 Dynamic Panel

**Model**:
```
intl_migration_{i,t} = rho * intl_migration_{i,t-1} + alpha_i + gamma_t + epsilon_{i,t}
```

**Implementation**: Arellano-Bond GMM via `linearmodels`

### 2.3 Cross-Sectional Dependence

**Solutions**:
1. Driscoll-Kraay standard errors
2. Common correlated effects (CCE)

---

## 3. Bayesian VAR Plan

### 3.1 Minnesota/Litterman Prior

**Objective**: Apply shrinkage priors to stabilize VAR with n=15-25.

**Prior Specification**:

For VAR(p) coefficients A_k:
```
A_k[i,j] ~ N(m_ij, v_ij)

where:
  m_ii = 1 for k=1, 0 otherwise (random walk prior)
  m_ij = 0 for i != j

  v_ij = (lambda1 / k^lambda3) * (sigma_i / sigma_j) if i = j
       = (lambda1 * lambda2 / k^lambda3) * (sigma_i / sigma_j) if i != j
```

**Hyperparameters**:
- `lambda1` = 0.1 (overall tightness)
- `lambda2` = 0.5 (cross-variable tightness)
- `lambda3` = 1 (lag decay)

### 3.2 Implementation Approach

**Option A: PyMC (Recommended)**

```python
import pymc as pm

with pm.Model() as bvar_model:
    A = pm.Normal("A", mu=prior_mean, sigma=prior_std, shape=(k*p, k))
    Sigma = pm.LKJCholeskyCov("Sigma", n=k, eta=1.0)
    Y_hat = pm.math.dot(X, A)
    pm.MvNormal("Y", mu=Y_hat, chol=Sigma, observed=Y)
    trace = pm.sample(2000, tune=1000, cores=4)
```

**Option B: statsmodels State-Space** (if PyMC unavailable)

**Option C: Manual Conjugate Prior** (fallback)
- Normal-Inverse-Wishart analytical posterior
- No MCMC needed

### 3.3 Posterior Inference

**Outputs**:
1. Posterior distributions for VAR coefficients
2. Credible intervals (90%, 95%) for forecasts
3. IRF posterior distributions
4. Posterior predictive checks

### 3.4 Classical vs Bayesian Comparison

| Aspect | Classical VAR | Bayesian VAR |
|--------|---------------|--------------|
| Estimation | MLE | MCMC posterior |
| Uncertainty | Asymptotic CI | Credible intervals |
| Small-n | Unstable | Regularized |
| Forecast | Point + SE | Full predictive distribution |

---

## 4. Model Comparison Framework

### 4.1 Value Assessment Criteria

| Criterion | Threshold for "Adds Value" |
|-----------|---------------------------|
| Forecast accuracy | RMSE improvement > 10% |
| Uncertainty | Prediction intervals correctly calibrated |
| Coefficient stability | Same sign/significance across methods |
| Robustness | Conclusions unchanged |
| Tractability | Runs in < 10 minutes |

### 4.2 Comparison Metrics

**Forecast Accuracy**:
- RMSE, MAE, MAPE
- Leave-one-out CV

**Uncertainty Quantification**:
- Prediction Interval Coverage Probability (PICP)
- Mean Prediction Interval Width (MPIW)

**Model Selection**:
- WAIC, LOO-IC for Bayesian
- Bayes Factors

### 4.3 Visualizations

1. Fan charts (classical vs Bayesian)
2. Coefficient posterior plots
3. Panel decomposition
4. Sensitivity analysis

---

## 5. Files Inventory

### 5.1 Files to Modify

| File | Modification |
|------|--------------|
| `requirements_statistical.txt` | Add PyMC, arviz |
| `SUBAGENT_COORDINATION.md` | Add B4 documentation |

### 5.2 New Files to Create

```
sdc_2024_replication/scripts/statistical_analysis/
├── module_B4_bayesian_panel/
│   ├── __init__.py
│   ├── minnesota_prior.py
│   ├── bayesian_var.py
│   ├── panel_var.py
│   ├── model_comparison.py
│   └── shrinkage_diagnostics.py
├── module_B4_bayesian_panel_var.py
├── results/
│   ├── module_B4_bayesian_var.json
│   ├── module_B4_panel_var.json
│   ├── module_B4_comparison.json
│   └── module_B4_recommendation.json
└── figures/
    ├── module_B4_posterior_distributions.png
    ├── module_B4_forecast_comparison.png
    ├── module_B4_panel_effects.png
    └── module_B4_sensitivity.png
```

### 5.3 Dependencies

| Agent | What B4 Needs | Status |
|-------|---------------|--------|
| B1 | Regime definitions | Available |
| B2 | 50-state panel | To be generated |

| Agent | What They Need from B4 |
|-------|------------------------|
| B3 | Recommendation on value |
| B6 | Test specifications |

---

## 6. Code Structure

### 6.1 Key Functions

```python
# minnesota_prior.py
def construct_minnesota_prior(
    n_vars: int,
    n_lags: int,
    sigma_estimates: np.ndarray,
    lambda1: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct Minnesota prior matrices."""

# bayesian_var.py
def estimate_bayesian_var(
    data: pd.DataFrame,
    var_cols: list[str],
    n_lags: int = 1,
    prior_type: str = "minnesota",
) -> dict:
    """Estimate BVAR with specified prior."""

# panel_var.py
def estimate_panel_var(
    df: pd.DataFrame,
    target_var: str,
    focal_entity: str = "North Dakota",
) -> dict:
    """Estimate panel VAR with entity effects."""

# model_comparison.py
def compare_models(
    classical_results: dict,
    bayesian_results: dict,
    panel_results: dict,
) -> dict:
    """Compare forecasting accuracy."""
```

---

## 7. Risks and Blockers

### 7.1 Computational

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| PyMC installation | MEDIUM | Fallback to conjugate prior |
| MCMC convergence | MEDIUM | Use NUTS, check Rhat |
| Runtime | LOW | n=25 is trivial |

### 7.2 Package Availability

| Risk | Resolution |
|------|------------|
| PyMC not installed | **BLOCKER** - must add to requirements |
| arviz not installed | **BLOCKER** - required for diagnostics |

**Fallback Strategy**: If PyMC unavailable:
1. Use statsmodels state-space
2. Implement manual conjugate prior
3. Document as limitation

### 7.3 Statistical Challenges

| Challenge | Mitigation |
|-----------|------------|
| n=15 too short | Minnesota prior shrinkage |
| Regime breaks in n=25 | Use B1 regime dummies |
| Panel heterogeneity | ND-specific slopes |

---

## 8. Estimated Complexity

| Component | Complexity |
|-----------|------------|
| Minnesota prior | **LOW** |
| Bayesian VAR (with PyMC) | **MEDIUM** |
| Bayesian VAR (without PyMC) | **HIGH** |
| Panel VAR extension | **MEDIUM** |
| Model comparison | **MEDIUM** |
| **Overall** | **MEDIUM** |

---

## 9. Implementation Sequence

1. **Package Setup**: Add PyMC/arviz, verify installation
2. **Panel Extensions**: Extend module_3_1 to Panel VAR
3. **Bayesian VAR Core**: Minnesota prior + estimation
4. **Comparison Framework**: Metrics and visualizations
5. **Integration**: Wait for B2 panel data, run full analysis

---

## 10. Key Decision Point

**The project must decide whether to add PyMC as a dependency.**

- If approved: Full Bayesian VAR is straightforward
- If not: Manual conjugate prior is feasible but more complex

---

## Summary

This plan provides:

1. **Infrastructure audit** of existing VAR and panel code
2. **Clear identification** that PyMC is NOT installed
3. **Minnesota prior specification** for small-n VAR
4. **Panel extension strategy** using B2's 50-state data
5. **Comparison framework** for assessing value
6. **File inventory** with 10+ new files
7. **Fallback strategy** if PyMC unavailable

**Decision Required**: Approve this plan and resolve PyMC dependency question.
